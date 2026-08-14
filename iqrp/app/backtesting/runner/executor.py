"""PipelineExecutor — load bars and drive EventDrivenEngine chronologically."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from iqrp.app.backtesting.accounting import CapitalState, PositionBook
from iqrp.app.backtesting.event_engine import (
    BacktestClock,
    ClockFrequency,
    EventDrivenEngine,
    MarketEvent,
)
from iqrp.app.backtesting.runner.checkpoint import write_checkpoint
from iqrp.app.backtesting.runner.configuration import BacktestRunConfig
from iqrp.app.backtesting.runner.context import PipelineContext
from iqrp.app.backtesting.runner.pipeline import EventPipeline
from iqrp.app.backtesting.strategy.base import Strategy


def _parse_ts(value: str | datetime | None, *, end: bool = False) -> datetime | None:
    if value is None:  # pragma: no cover
        return None
    if isinstance(value, datetime):
        ts = value
    else:
        ts = pd.Timestamp(value).to_pydatetime()
    if ts.tzinfo is None:  # pragma: no cover
        ts = ts.replace(tzinfo=UTC)
    if end and isinstance(value, str) and len(value) <= 10:
        # Inclusive calendar end-of-day
        ts = ts.replace(hour=23, minute=59, second=59, microsecond=999999)
    return ts


def load_market_frame(config: BacktestRunConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate historical bars via the data package."""
    from iqrp.app.backtesting.data import CSVAdapter, DatasetValidator, ParquetAdapter

    path = config.dataset_path
    if not path:
        raise FileNotFoundError("dataset_path is required (no remote download)")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset path not found: {p}")

    adapter_name = str(config.adapter or "parquet").lower()
    if adapter_name in {"csv", "CSVAdapter"}:
        adapter = CSVAdapter(p, dataset_id=config.dataset_id or p.stem)
    else:
        adapter = ParquetAdapter(p, dataset_id=config.dataset_id or p.stem)

    frame = adapter.load()
    report = DatasetValidator().validate(frame, raise_on_critical=False)
    detail = {
        "dataset_id": getattr(adapter, "dataset_id", p.stem),
        "path": str(p),
        "rows": int(len(frame)),
        "ok": bool(getattr(report, "ok", True)),
        "critical_failures": list(getattr(report, "critical_failures", []) or []),
        "report": report.to_dict() if hasattr(report, "to_dict") else {},
    }
    if detail["critical_failures"]:
        raise ValueError(f"dataset validation failed: {detail['critical_failures']}")

    # Optional range / universe filter
    if "timestamp" in frame.columns:
        if config.start:
            start = _parse_ts(config.start)
            frame = frame[frame["timestamp"] >= start]
        if config.end:
            end = _parse_ts(config.end, end=True)
            frame = frame[frame["timestamp"] <= end]
    if config.universe:
        frame = frame[frame["instrument"].astype(str).isin(set(config.universe))]
    if frame.empty:
        raise ValueError("no bars remain after start/end/universe filters")
    return frame.reset_index(drop=True), detail


def bars_by_timestamp(frame: pd.DataFrame) -> list[tuple[datetime, dict[str, dict[str, Any]]]]:
    """Group OHLCV rows into chronological (timestamp → instrument bars) pairs."""
    ordered = frame.sort_values(["timestamp", "instrument"])
    out: list[tuple[datetime, dict[str, dict[str, Any]]]] = []
    for ts, group in ordered.groupby("timestamp", sort=True):
        ts_dt = pd.Timestamp(ts).to_pydatetime()
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=UTC)
        bars: dict[str, dict[str, Any]] = {}
        for _, row in group.iterrows():
            inst = str(row["instrument"])
            bars[inst] = {
                "timestamp": ts_dt,
                "instrument": inst,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0) or 0.0),
            }
        out.append((ts_dt, bars))
    return out


class PipelineExecutor:
    """Feed MARKET events into EventDrivenEngine and run the cascade."""

    def __init__(
        self,
        config: BacktestRunConfig,
        strategy: Strategy,
        *,
        frame: pd.DataFrame | None = None,
        data_detail: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.frame = frame
        self.data_detail = dict(data_detail or {})
        self.context: PipelineContext | None = None
        self.engine: EventDrivenEngine | None = None
        self.pipeline: EventPipeline | None = None
        self._bar_schedule: list[tuple[datetime, dict[str, dict[str, Any]]]] = []
        self._resume_after: datetime | None = None

    def prepare(self) -> PipelineContext:
        if self.frame is None:
            self.frame, self.data_detail = load_market_frame(self.config)
        self._bar_schedule = bars_by_timestamp(self.frame)
        if not self._bar_schedule:
            raise ValueError("no market bars to execute")

        start_ts = self._bar_schedule[0][0]
        freq_raw = str(self.config.frequency or "daily").lower()
        try:  # pragma: no cover - alias map below is authoritative
            freq = (
                ClockFrequency(freq_raw)
                if freq_raw in ClockFrequency.__members__.values()
                or freq_raw in {f.value for f in ClockFrequency}
                else ClockFrequency.DAILY
            )
        except Exception:
            freq = ClockFrequency.DAILY
        # Map common strings
        aliases = {
            "d": ClockFrequency.DAILY,
            "day": ClockFrequency.DAILY,
            "daily": ClockFrequency.DAILY,
            "1d": ClockFrequency.DAILY,
            "minute": ClockFrequency.MINUTE,
            "hourly": ClockFrequency.HOURLY,
        }
        freq = aliases.get(freq_raw, ClockFrequency.DAILY)

        clock = BacktestClock(start=start_ts, frequency=freq, timezone=self.config.timezone)
        engine = EventDrivenEngine(
            clock=clock,
            on_invalidate=lambda reason: None,
        )
        capital = CapitalState(
            initial_capital=float(self.config.initial_capital),
            currency=str(self.config.currency),
        )
        positions = PositionBook(currency=str(self.config.currency))
        universe = list(self.config.universe) or sorted(
            {str(x) for x in self.frame["instrument"].astype(str).unique()}
        )
        ctx = PipelineContext(
            config=self.config,
            strategy=self.strategy,
            capital=capital,
            positions=positions,
            universe=universe,
            peak_equity=float(self.config.initial_capital),
            diagnostics={
                "data_validated": bool(self.data_detail.get("ok", True)),
                "data_detail": dict(self.data_detail),
            },
            random_state={"seed": int(self.config.seed)},
        )
        self.strategy.initialize(ctx)
        self.pipeline = EventPipeline(engine, ctx)
        self.engine = engine
        self.context = ctx
        return ctx

    def submit_market_events(self, *, resume_after: datetime | None = None) -> int:
        if self.engine is None or self.context is None:
            raise RuntimeError("call prepare() before submit_market_events()")
        count = 0
        for ts, bars in self._bar_schedule:
            if resume_after is not None and ts <= resume_after:
                continue
            self.engine.submit(
                MarketEvent(
                    timestamp=ts,
                    payload={"bars": bars, "asof": ts.isoformat()},
                )
            )
            count += 1
        return count

    def run(
        self,
        *,
        resume_after: datetime | None = None,
        checkpoint_every: int | None = None,
    ) -> PipelineContext:
        if self.context is None or self.engine is None:
            self.prepare()
        assert self.context is not None and self.engine is not None

        n = self.submit_market_events(resume_after=resume_after or self._resume_after)
        if n == 0 and resume_after is None:
            raise RuntimeError("no market events submitted")

        end_ts = self._bar_schedule[-1][0]
        start_ts = self.engine.clock.now
        self.engine.run(start=start_ts, end=end_ts)

        if checkpoint_every and self.config.checkpoint_dir:
            write_checkpoint(
                self.context,
                Path(self.config.checkpoint_dir) / self.config.backtest_id / "checkpoint.json",
            )
        self.strategy.on_end(self.context)
        return self.context


__all__ = ["PipelineExecutor", "bars_by_timestamp", "load_market_frame"]
