"""Calendar features from open_time."""

from __future__ import annotations

import polars as pl

from iqrp.app.features._polars_utils import with_open_time
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.registry import register_feature


@register_feature
class HourOfDay(Feature):
    meta = FeatureMeta(
        name="hour",
        version="1.0.0",
        description="Hour of day (UTC)",
        category="calendar",
        required_columns=("open_time",),
        output_columns=("hour",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return with_open_time(frame, pl.col("open_time").dt.hour().alias("hour"))


@register_feature
class DayOfMonth(Feature):
    meta = FeatureMeta(
        name="day",
        version="1.0.0",
        description="Day of month",
        category="calendar",
        required_columns=("open_time",),
        output_columns=("day",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return with_open_time(frame, pl.col("open_time").dt.day().alias("day"))


@register_feature
class WeekOfYear(Feature):
    meta = FeatureMeta(
        name="week",
        version="1.0.0",
        description="ISO week number",
        category="calendar",
        required_columns=("open_time",),
        output_columns=("week",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return with_open_time(frame, pl.col("open_time").dt.week().alias("week"))


@register_feature
class MonthOfYear(Feature):
    meta = FeatureMeta(
        name="month",
        version="1.0.0",
        description="Month of year",
        category="calendar",
        required_columns=("open_time",),
        output_columns=("month",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return with_open_time(frame, pl.col("open_time").dt.month().alias("month"))


@register_feature
class QuarterOfYear(Feature):
    meta = FeatureMeta(
        name="quarter",
        version="1.0.0",
        description="Quarter of year",
        category="calendar",
        required_columns=("open_time",),
        output_columns=("quarter",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return with_open_time(frame, pl.col("open_time").dt.quarter().alias("quarter"))


@register_feature
class YearFeature(Feature):
    meta = FeatureMeta(
        name="year",
        version="1.0.0",
        description="Calendar year",
        category="calendar",
        required_columns=("open_time",),
        output_columns=("year",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        return with_open_time(frame, pl.col("open_time").dt.year().alias("year"))


@register_feature
class WeekendFlag(Feature):
    meta = FeatureMeta(
        name="weekend",
        version="1.0.0",
        description="1 if Saturday/Sunday else 0",
        category="calendar",
        required_columns=("open_time",),
        output_columns=("weekend",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        # Polars weekday: Monday=1 ... Sunday=7
        wd = pl.col("open_time").dt.weekday()
        return with_open_time(frame, ((wd >= 6).cast(pl.Float64)).alias("weekend"))


@register_feature
class SessionFlag(Feature):
    meta = FeatureMeta(
        name="session",
        version="1.0.0",
        description="0=Asia, 1=Europe, 2=US by UTC hour",
        category="calendar",
        required_columns=("open_time",),
        output_columns=("session",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = pl.col("open_time").dt.hour()
        session = pl.when(h < 8).then(0).when(h < 14).then(1).otherwise(2).cast(pl.Float64)
        return with_open_time(frame, session.alias("session"))


@register_feature
class HolidayFlag(Feature):
    meta = FeatureMeta(
        name="holiday_flag",
        version="1.0.0",
        description="Weekend proxy holiday flag (extensible)",
        category="calendar",
        required_columns=("open_time",),
        output_columns=("holiday_flag",),
        dependencies=("weekend",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        wd = pl.col("open_time").dt.weekday()
        return with_open_time(frame, ((wd >= 6).cast(pl.Float64)).alias("holiday_flag"))
