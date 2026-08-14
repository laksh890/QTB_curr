"""Causal research feature registry and deterministic OHLCV feature computation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.normalize import (
    causal_rank,
    causal_rolling_zscore,
    causal_vol_normalize,
)


@dataclass
class FeatureSpec:
    feature_id: str
    version: str = "1.0.0"
    description: str = ""
    inputs: tuple[str, ...] = ("close",)
    output_schema: tuple[str, ...] = ()
    timeframe: str = "native"
    lookback: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    availability: str = "always"
    warmup: int = 0
    family: str = "generic"
    nan_behavior: str = "leading_nan_during_warmup"
    normalization: str | None = None

    def __post_init__(self) -> None:
        if not self.output_schema:
            self.output_schema = (self.feature_id,)
        if self.warmup <= 0:
            self.warmup = max(int(self.lookback), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "version": self.version,
            "description": self.description,
            "inputs": list(self.inputs),
            "output_schema": list(self.output_schema),
            "timeframe": self.timeframe,
            "lookback": self.lookback,
            "parameters": dict(self.parameters),
            "dependencies": list(self.dependencies),
            "availability": self.availability,
            "warmup": self.warmup,
            "family": self.family,
            "nan_behavior": self.nan_behavior,
            "normalization": self.normalization,
            "disclaimer": "RESEARCH FEATURE — not a profitability claim.",
        }


FeatureFn = Callable[[pd.DataFrame, FeatureSpec], pd.Series]


class FeatureRegistry:
    """Register research features by feature_id@version."""

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], FeatureSpec] = {}
        self._fns: dict[tuple[str, str], FeatureFn] = {}

    def register(
        self,
        spec: FeatureSpec,
        fn: FeatureFn,
        *,
        overwrite: bool = False,
    ) -> None:
        key = (spec.feature_id, spec.version)
        if key in self._specs and not overwrite:
            raise ValueError(f"feature already registered: {key}")
        self._specs[key] = spec
        self._fns[key] = fn

    def get(self, feature_id: str, version: str | None = None) -> tuple[FeatureSpec, FeatureFn]:
        if version:
            key = (feature_id, version)
            if key not in self._specs:
                raise KeyError(f"unknown feature {feature_id}@{version}")
            return self._specs[key], self._fns[key]
        matches = [(k, s) for k, s in self._specs.items() if k[0] == feature_id]
        if not matches:
            raise KeyError(f"unknown feature {feature_id}")
        if len(matches) > 1:
            vers = sorted(k[1] for k, _ in matches)
            raise KeyError(f"feature {feature_id} has multiple versions {vers}; pass version")
        k = matches[0][0]
        return self._specs[k], self._fns[k]

    def list(self) -> list[FeatureSpec]:
        return [self._specs[k] for k in sorted(self._specs)]

    def compute(
        self,
        frame: pd.DataFrame,
        feature_id: str,
        *,
        version: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Compute a causal feature series with metadata."""
        spec, fn = self.get(feature_id, version)
        # bind parameter overrides into a shallow copy
        use = FeatureSpec(
            feature_id=spec.feature_id,
            version=spec.version,
            description=spec.description,
            inputs=spec.inputs,
            output_schema=spec.output_schema,
            timeframe=spec.timeframe,
            lookback=int(parameters.get("lookback", spec.lookback)) if parameters else spec.lookback,
            parameters={**spec.parameters, **dict(parameters or {})},
            dependencies=spec.dependencies,
            availability=spec.availability,
            warmup=int(parameters.get("warmup", spec.warmup)) if parameters else spec.warmup,
            family=spec.family,
            nan_behavior=spec.nan_behavior,
            normalization=(
                parameters.get("normalization", spec.normalization) if parameters else spec.normalization
            ),
        )
        if use.warmup < use.lookback:
            use.warmup = use.lookback
        series = fn(frame, use)
        series = pd.Series(series, index=frame.index, dtype=np.float64)
        # enforce warmup NaNs
        if use.warmup > 0 and len(series) >= use.warmup:
            series.iloc[: use.warmup] = np.nan
        if use.normalization == "rolling_zscore":
            series = causal_rolling_zscore(series, window=max(use.lookback, 20))
        elif use.normalization == "rank":
            series = causal_rank(series, window=max(use.lookback, 20))
        elif use.normalization == "vol":
            series = causal_vol_normalize(series, window=max(use.lookback, 20))
        meta = {
            **use.to_dict(),
            "source_columns": list(use.inputs),
            "n": int(series.notna().sum()),
            "implementation_version": use.version,
            "causal": True,
            "disclaimer": "Feature values are research diagnostics only.",
        }
        return series, meta


_GLOBAL = FeatureRegistry()


def get_feature_registry() -> FeatureRegistry:
    if not _GLOBAL.list():
        from iqrp.app.backtesting.alpha_research import reference_features as _rf

        _rf.register_reference_features(_GLOBAL)
    return _GLOBAL


__all__ = ["FeatureFn", "FeatureRegistry", "FeatureSpec", "get_feature_registry"]
