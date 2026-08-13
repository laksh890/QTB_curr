"""Query API for labels: get/list/describe + compute-and-store."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from iqrp.app.labels.base.pipeline import LabelPipeline, LabelPipelineBenchmarks
from iqrp.app.labels.base.registry import ensure_labels_loaded, get_registry
from iqrp.app.labels.config import LabelSettings
from iqrp.app.labels.custom import ensure_custom_examples_registered
from iqrp.app.labels.store.label_store import LabelStore
from iqrp.app.labels.validation import LabelValidator
from iqrp.app.labels.visualization import LabelVisualizer


class LabelQueryService:
    def __init__(
        self,
        store: LabelStore | None = None,
        *,
        store_root: Path | None = None,
        pipeline: LabelPipeline | None = None,
        settings: LabelSettings | None = None,
    ) -> None:
        ensure_labels_loaded()
        ensure_custom_examples_registered()
        self.settings = settings or LabelSettings.default()
        self.registry = get_registry()
        self.store = store or LabelStore(store_root or Path(self.settings.store_dir))
        self.pipeline = pipeline or LabelPipeline(max_workers=self.settings.n_jobs)
        self.validator = LabelValidator(self.settings)
        self.visualizer = LabelVisualizer(self.settings)

    def list_labels(self, *, category: str | None = None) -> list[str]:
        return self.registry.list_names(category=category)

    def describe_label(self, name: str) -> dict[str, Any]:
        return self.registry.describe(name).to_dict()

    def get_label(
        self,
        name: str,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        version: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        meta = self.registry.describe(name)
        frame = self.store.read(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            label_name=name,
            version=version or meta.version,
            start=start,
            end=end,
        )
        cols = [c for c in ("open_time", *meta.output_columns) if c in frame.columns]
        return frame.select(cols) if cols else frame

    def get_labels(
        self,
        names: list[str],
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for name in names:
            part = self.get_label(
                name,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
            )
            if not part.is_empty():
                frames.append(part)
        if not frames:
            return pl.DataFrame()
        out = frames[0]
        for frm in frames[1:]:
            if "open_time" in out.columns and "open_time" in frm.columns:
                new_cols = [c for c in frm.columns if c == "open_time" or c not in out.columns]
                out = out.join(frm.select(new_cols), on="open_time", how="full", coalesce=True)
            else:
                add = [c for c in frm.columns if c not in out.columns]
                if add:
                    out = out.hstack(frm.select(add))
        return out.sort("open_time") if "open_time" in out.columns else out

    def compute_and_store(
        self,
        market_frame: pl.DataFrame,
        label_names: list[str] | None,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        incremental: bool = False,
        write_reports: bool = True,
    ) -> tuple[pl.DataFrame, LabelPipelineBenchmarks]:
        result, bench = self.pipeline.compute(market_frame, label_names)
        names = label_names or self.registry.list_names()
        for name in names:
            meta = self.registry.describe(name)
            cols = [c for c in ("open_time", *meta.output_columns) if c in result.columns]
            if not cols:
                continue
            part = result.select(cols)
            if incremental:
                self.store.update_incremental(
                    part,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    label_name=name,
                    version=meta.version,
                )
            else:
                self.store.write(
                    part,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    label_name=name,
                    version=meta.version,
                )

        if write_reports:
            label_cols = [
                c
                for c in result.columns
                if c not in {"open_time", "open", "high", "low", "close", "volume"}
            ]
            out_dir = Path(self.settings.output_dir)
            self.visualizer.write_all(out_dir / "charts", result, label_columns=label_cols)
            report = self.validator.validate(result, label_cols)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "label_quality_report.json").write_text(
                json.dumps(report.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        return result, bench


def get_label(
    name: str,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    store_root: Path | None = None,
    version: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    return LabelQueryService(store_root=store_root).get_label(
        name,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        version=version,
        start=start,
        end=end,
    )


def get_labels(
    names: list[str],
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    store_root: Path | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    return LabelQueryService(store_root=store_root).get_labels(
        names, exchange=exchange, symbol=symbol, timeframe=timeframe, start=start, end=end
    )


def list_labels(*, category: str | None = None) -> list[str]:
    return LabelQueryService().list_labels(category=category)


def describe_label(name: str) -> dict[str, Any]:
    return LabelQueryService().describe_label(name)
