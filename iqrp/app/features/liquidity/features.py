"""Liquidity features (order-book aware when columns present)."""

from __future__ import annotations

import polars as pl

from iqrp.app.features._polars_utils import safe_div, with_open_time
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.registry import register_feature


def _ensure_book_cols(frame: pl.DataFrame) -> pl.DataFrame:
    """Synthesize book columns from OHLCV when absent (research fallback)."""
    out = frame
    if "best_bid" not in out.columns:
        out = out.with_columns((pl.col("low")).alias("best_bid"))
    if "best_ask" not in out.columns:
        out = out.with_columns((pl.col("high")).alias("best_ask"))
    if "bid_size" not in out.columns:
        out = out.with_columns((pl.col("volume") * 0.5).alias("bid_size"))
    if "ask_size" not in out.columns:
        out = out.with_columns((pl.col("volume") * 0.5).alias("ask_size"))
    return out


@register_feature
class BidAskSpread(Feature):
    meta = FeatureMeta(
        name="bid_ask_spread",
        version="1.0.0",
        description="Absolute bid-ask spread",
        category="liquidity",
        required_columns=("close",),
        output_columns=("bid_ask_spread",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _ensure_book_cols(frame)
        return with_open_time(f, (pl.col("best_ask") - pl.col("best_bid")).alias("bid_ask_spread"))


@register_feature
class EffectiveSpread(Feature):
    meta = FeatureMeta(
        name="effective_spread",
        version="1.0.0",
        description="2 * |close - mid|",
        category="liquidity",
        required_columns=("close",),
        output_columns=("effective_spread",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _ensure_book_cols(frame)
        mid = (pl.col("best_bid") + pl.col("best_ask")) / 2.0
        return with_open_time(f, (2.0 * (pl.col("close") - mid).abs()).alias("effective_spread"))


@register_feature
class QuotedSpread(Feature):
    meta = FeatureMeta(
        name="quoted_spread",
        version="1.0.0",
        description="(ask - bid) / mid",
        category="liquidity",
        required_columns=("close",),
        output_columns=("quoted_spread",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _ensure_book_cols(frame)
        mid = (pl.col("best_bid") + pl.col("best_ask")) / 2.0
        return with_open_time(
            f, safe_div(pl.col("best_ask") - pl.col("best_bid"), mid).alias("quoted_spread")
        )


@register_feature
class DepthImbalance(Feature):
    meta = FeatureMeta(
        name="depth_imbalance",
        version="1.0.0",
        description="(bid_size - ask_size) / (bid_size + ask_size)",
        category="liquidity",
        required_columns=("volume",),
        output_columns=("depth_imbalance",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _ensure_book_cols(frame)
        return with_open_time(
            f,
            safe_div(
                pl.col("bid_size") - pl.col("ask_size"),
                pl.col("bid_size") + pl.col("ask_size"),
            ).alias("depth_imbalance"),
        )


@register_feature
class MarketDepth(Feature):
    meta = FeatureMeta(
        name="market_depth",
        version="1.0.0",
        description="bid_size + ask_size",
        category="liquidity",
        required_columns=("volume",),
        output_columns=("market_depth",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _ensure_book_cols(frame)
        return with_open_time(f, (pl.col("bid_size") + pl.col("ask_size")).alias("market_depth"))


@register_feature
class LiquidityScore(Feature):
    meta = FeatureMeta(
        name="liquidity_score",
        version="1.0.0",
        description="depth / (1 + quoted_spread)",
        category="liquidity",
        required_columns=("close", "volume"),
        output_columns=("liquidity_score",),
        dependencies=("quoted_spread", "market_depth"),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _ensure_book_cols(frame)
        mid = (pl.col("best_bid") + pl.col("best_ask")) / 2.0
        qs = safe_div(pl.col("best_ask") - pl.col("best_bid"), mid)
        depth = pl.col("bid_size") + pl.col("ask_size")
        return with_open_time(f, safe_div(depth, 1.0 + qs).alias("liquidity_score"))


@register_feature
class Turnover(Feature):
    meta = FeatureMeta(
        name="turnover",
        version="1.0.0",
        description="close * volume",
        category="liquidity",
        required_columns=("close", "volume"),
        output_columns=("turnover",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return with_open_time(frame, (pl.col("close") * pl.col("volume")).alias("turnover"))
