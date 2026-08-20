"""Deterministic synthetic OHLCV generators for tests and fixtures.

These series are synthetic fixtures only. They do not represent real markets
and must not be interpreted as investment advice or profitability claims.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from iqrp.app.backtesting.data.dataset import HistoricalDataset
from iqrp.app.backtesting.data.metadata import metadata_from_frame
from iqrp.app.backtesting.data.schema import normalize_frame

__all__ = [
    "generate_synthetic_ohlcv",
    "write_synthetic_ohlcv",
    "create_synthetic_ohlcv",
]


def generate_synthetic_ohlcv(
    *,
    n_days: int = 30,
    instruments: Sequence[str] | None = None,
    seed: int = 1,
    start: str | datetime | pd.Timestamp = "2020-01-01",
    freq: str = "1d",
) -> pd.DataFrame:
    """Generate a deterministic multi-instrument OHLCV DataFrame."""
    if n_days < 1:
        raise ValueError("n_days must be >= 1")
    inst = list(instruments or ["AAA", "BBB"])
    rng = np.random.default_rng(int(seed))
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    if freq.endswith("d") and freq[:-1].isdigit():
        # business days for daily fixtures
        periods = int(n_days)
        timestamps = pd.bdate_range(start=start_ts, periods=periods, tz="UTC")
    else:
        timestamps = pd.date_range(start=start_ts, periods=int(n_days), freq=freq, tz="UTC")

    rows: list[dict[str, object]] = []
    for i, symbol in enumerate(inst):
        # Distinct but deterministic path per instrument
        base = 50.0 + 10.0 * i
        shocks = rng.normal(loc=0.0, scale=0.01, size=len(timestamps))
        closes = base * np.cumprod(1.0 + shocks)
        opens = np.concatenate([[closes[0]], closes[:-1]])
        spreads = np.abs(rng.normal(0.0, 0.005, size=len(timestamps)))
        highs = np.maximum(opens, closes) * (1.0 + spreads)
        lows = np.minimum(opens, closes) * (1.0 - spreads)
        # Enforce OHLC invariants
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        volumes = rng.integers(1_000, 10_000, size=len(timestamps)).astype(float)
        for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes):
            rows.append(
                {
                    "timestamp": ts,
                    "instrument": str(symbol),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v),
                }
            )
    return normalize_frame(pd.DataFrame(rows))


def create_synthetic_ohlcv(
    path: str | Path | None = None,
    *,
    n_days: int = 30,
    instruments: Sequence[str] | None = None,
    seed: int = 1,
    start: str | datetime | pd.Timestamp = "2020-01-01",
    freq: str = "1d",
) -> pd.DataFrame | HistoricalDataset:
    """Generate synthetic OHLCV; optionally write to ``path`` as parquet/CSV."""
    frame = generate_synthetic_ohlcv(
        n_days=n_days,
        instruments=instruments,
        seed=seed,
        start=start,
        freq=freq,
    )
    if path is None:
        return frame
    return write_synthetic_ohlcv(
        path,
        n_days=n_days,
        instruments=instruments,
        seed=seed,
        start=start,
        freq=freq,
        frame=frame,
    )


def write_synthetic_ohlcv(
    path: str | Path,
    *,
    n_days: int = 30,
    instruments: Sequence[str] | None = None,
    seed: int = 1,
    start: str | datetime | pd.Timestamp = "2020-01-01",
    freq: str = "1d",
    frame: pd.DataFrame | None = None,
    dataset_id: str | None = None,
) -> HistoricalDataset:
    """Write synthetic OHLCV to ``path`` and return a :class:`HistoricalDataset`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = frame if frame is not None else generate_synthetic_ohlcv(
        n_days=n_days,
        instruments=instruments,
        seed=seed,
        start=start,
        freq=freq,
    )
    data = normalize_frame(data)
    suffix = out.suffix.lower()
    if suffix == ".csv":
        data.to_csv(out, index=False)
    elif suffix in {".feather", ".arrow"}:
        data.to_feather(out)
    else:
        # default parquet (also for .parquet / .pq / no suffix)
        if suffix not in {".parquet", ".pq"}:
            out = out.with_suffix(".parquet")
        data.to_parquet(out, index=False)

    meta = metadata_from_frame(
        data,
        dataset_id=dataset_id or out.stem,
        version="1.0.0",
        source="synthetic",
        path=str(out),
        known_limitations=[
            "Synthetic fixture data for testing only; not real market prices.",
        ],
    )
    return HistoricalDataset(frame=data, metadata=meta)
