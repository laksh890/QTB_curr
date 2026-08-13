"""Query API for the feature store / registry."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from iqrp.app.features.base.pipeline import FeaturePipeline, PipelineBenchmarks
from iqrp.app.features.base.registry import ensure_features_loaded, get_registry
from iqrp.app.features.store.feature_store import FeatureStore


class FeatureQueryService:
    """Central access point for feature metadata and persisted matrices."""

    def __init__(
        self,
        store: FeatureStore | None = None,
        *,
        store_root: Path | None = None,
        pipeline: FeaturePipeline | None = None,
    ) -> None:
        ensure_features_loaded()
        self.registry = get_registry()
        self.store = store or FeatureStore(store_root or Path("data/features"))
        self.pipeline = pipeline or FeaturePipeline()

    def list_features(self, *, category: str | None = None) -> list[str]:
        return self.registry.list_names(category=category)

    def describe_feature(self, name: str) -> dict[str, Any]:
        return self.registry.describe(name).to_dict()

    def feature_dependencies(self, name: str) -> tuple[str, ...]:
        return self.registry.dependencies(name)

    def get_feature(
        self,
        name: str,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        feature_group: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        meta = self.registry.describe(name)
        frame = self.store.read(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            feature_group=feature_group,
            start=start,
            end=end,
        )
        cols = [c for c in ("open_time", *meta.output_columns) if c in frame.columns]
        return frame.select(cols) if cols else frame

    def get_features(
        self,
        names: list[str],
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        feature_group: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for name in names:
            part = self.get_feature(
                name,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                feature_group=feature_group,
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
        feature_names: list[str] | None,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        feature_group: str = "all",
        incremental_since: datetime | None = None,
    ) -> tuple[pl.DataFrame, PipelineBenchmarks]:
        result, bench = self.pipeline.compute(market_frame, feature_names, since=incremental_since)
        if incremental_since is not None:
            self.store.update_incremental(
                result,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                feature_group=feature_group,
            )
        else:
            self.store.write(
                result,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                feature_group=feature_group,
            )
        return result, bench


def get_feature(
    name: str,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    store_root: Path | None = None,
    feature_group: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    return FeatureQueryService(store_root=store_root).get_feature(
        name,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        feature_group=feature_group,
        start=start,
        end=end,
    )


def get_features(
    names: list[str],
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    store_root: Path | None = None,
    feature_group: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    return FeatureQueryService(store_root=store_root).get_features(
        names,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        feature_group=feature_group,
        start=start,
        end=end,
    )


def list_features(*, category: str | None = None) -> list[str]:
    return FeatureQueryService().list_features(category=category)


def describe_feature(name: str) -> dict[str, Any]:
    return FeatureQueryService().describe_feature(name)


def feature_dependencies(name: str) -> tuple[str, ...]:
    return FeatureQueryService().feature_dependencies(name)
