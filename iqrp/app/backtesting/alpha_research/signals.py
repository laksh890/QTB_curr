"""Signal registry — signals reference features; produce LONG/SHORT/FLAT."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.features import FeatureRegistry, get_feature_registry
from iqrp.app.backtesting.alpha_research.types import SignalKind


@dataclass
class SignalSpec:
    signal_id: str
    version: str = "1.0.0"
    description: str = ""
    feature_ids: tuple[str, ...] = ()
    kind: SignalKind = SignalKind.CATEGORICAL
    parameters: dict[str, Any] = field(default_factory=dict)
    family: str = "generic"
    holding_bars: int = 1
    allow_short: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "version": self.version,
            "description": self.description,
            "feature_ids": list(self.feature_ids),
            "kind": self.kind.value,
            "parameters": dict(self.parameters),
            "family": self.family,
            "holding_bars": self.holding_bars,
            "allow_short": self.allow_short,
            "disclaimer": "RESEARCH SIGNAL — not a profitability claim.",
        }


SignalFn = Callable[[pd.DataFrame, Mapping[str, pd.Series], SignalSpec], pd.Series]


class SignalRegistry:
    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], SignalSpec] = {}
        self._fns: dict[tuple[str, str], SignalFn] = {}

    def register(self, spec: SignalSpec, fn: SignalFn, *, overwrite: bool = False) -> None:
        key = (spec.signal_id, spec.version)
        if key in self._specs and not overwrite:
            raise ValueError(f"signal already registered: {key}")
        self._specs[key] = spec
        self._fns[key] = fn

    def get(self, signal_id: str, version: str | None = None) -> tuple[SignalSpec, SignalFn]:
        if version:
            key = (signal_id, version)
            if key not in self._specs:
                raise KeyError(key)
            return self._specs[key], self._fns[key]
        matches = [k for k in self._specs if k[0] == signal_id]
        if not matches:
            raise KeyError(signal_id)
        if len(matches) > 1:
            raise KeyError(f"multiple versions for {signal_id}")
        k = matches[0]
        return self._specs[k], self._fns[k]

    def list(self) -> list[SignalSpec]:
        return [self._specs[k] for k in sorted(self._specs)]

    def generate(
        self,
        frame: pd.DataFrame,
        signal_id: str,
        *,
        version: str | None = None,
        feature_registry: FeatureRegistry | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> tuple[pd.Series, dict[str, Any], dict[str, pd.Series]]:
        spec, fn = self.get(signal_id, version)
        use = SignalSpec(
            signal_id=spec.signal_id,
            version=spec.version,
            description=spec.description,
            feature_ids=spec.feature_ids,
            kind=spec.kind,
            parameters={**spec.parameters, **dict(parameters or {})},
            family=spec.family,
            holding_bars=int((parameters or {}).get("holding_bars", spec.holding_bars)),
            allow_short=bool((parameters or {}).get("allow_short", spec.allow_short)),
        )
        freg = feature_registry or get_feature_registry()
        features: dict[str, pd.Series] = {}
        feature_meta: dict[str, Any] = {}
        for fid in use.feature_ids:
            lookback = int(use.parameters.get("lookback", use.parameters.get(f"{fid}_lookback", 20)))
            series, meta = freg.compute(frame, fid, parameters={"lookback": lookback})
            features[fid] = series
            feature_meta[fid] = meta
        raw = fn(frame, features, use)
        sig = pd.Series(raw, index=frame.index, dtype=np.float64)
        if not use.allow_short:
            sig = sig.clip(lower=0.0)
        meta = {
            **use.to_dict(),
            "feature_meta": feature_meta,
            "values_unique": sorted({float(x) for x in sig.dropna().unique().tolist()[:20]}),
        }
        return sig, meta, features


def apply_holding(signal: pd.Series, holding_bars: int) -> pd.Series:
    """Hold non-zero signal for N bars; allow reverse/flat."""
    h = max(int(holding_bars), 1)
    arr = signal.to_numpy(dtype=np.float64)
    pos = np.zeros_like(arr)
    i = 0
    n = arr.size
    while i < n:
        s = arr[i]
        if not np.isfinite(s) or abs(s) < 1e-15:
            i += 1
            continue
        end = min(i + h, n)
        pos[i:end] = np.sign(s)
        i = end
    return pd.Series(pos, index=signal.index, dtype=np.float64)


_GLOBAL = SignalRegistry()


def get_signal_registry() -> SignalRegistry:
    if not _GLOBAL.list():
        from iqrp.app.backtesting.alpha_research import reference_signals as _rs

        _rs.register_reference_signals(_GLOBAL)
    return _GLOBAL


__all__ = [
    "SignalFn",
    "SignalRegistry",
    "SignalSpec",
    "apply_holding",
    "get_signal_registry",
]
