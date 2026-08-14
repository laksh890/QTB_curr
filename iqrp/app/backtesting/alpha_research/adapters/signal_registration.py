"""Register model-adapter signals into the EXISTING SignalRegistry.

Does not replace reference signals. Call explicitly — not auto-wired into
get_signal_registry() — so Prompt 35 reference campaigns stay reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from iqrp.app.backtesting.alpha_research.adapters.model_registry import (
    get_adapter,
    list_adapters,
    register_default_adapters,
)
from iqrp.app.backtesting.alpha_research.adapters.pipeline import run_adapter
from iqrp.app.backtesting.alpha_research.signals import SignalFn, SignalRegistry, SignalSpec
from iqrp.app.backtesting.alpha_research.types import SignalKind

_CACHE: dict[str, pd.Series] = {}


def clear_model_signal_cache() -> None:
    _CACHE.clear()


def _make_fn(adapter_id: str) -> SignalFn:
    def _fn(
        frame: pd.DataFrame,
        features: Mapping[str, pd.Series],
        spec: SignalSpec,
    ) -> pd.Series:
        _ = features
        col = f"__model_signal__{adapter_id}"
        if col in frame.columns:
            return pd.Series(frame[col], index=frame.index, dtype=float)
        cache_key = f"{adapter_id}:{len(frame)}:{frame['timestamp'].iloc[0]}:{frame['timestamp'].iloc[-1]}"
        if cache_key in _CACHE and bool(spec.parameters.get("use_cache", True)):
            return _CACHE[cache_key].reindex(frame.index)
        train_frac = float(spec.parameters.get("train_frac", 0.5))
        result = run_adapter(adapter_id, frame, train_frac=train_frac)
        if result.get("status") != "PASS" or result.get("signal") is None:
            reason = result.get("reason", "adapter failed")
            raise RuntimeError(f"model adapter {adapter_id} unavailable: {reason}")
        sig = pd.Series(result["signal"], index=frame.index, dtype=float)
        _CACHE[cache_key] = sig
        return sig

    return _fn


def register_model_adapter_signals(
    registry: SignalRegistry,
    *,
    overwrite: bool = True,
    adapter_ids: list[str] | None = None,
) -> list[SignalSpec]:
    """Register adapter-backed signals into an existing SignalRegistry instance."""
    register_default_adapters(overwrite=True)
    specs_out: list[SignalSpec] = []
    adapters = list_adapters()
    if adapter_ids is not None:
        wanted = set(adapter_ids)
        adapters = [a for a in adapters if a.adapter_id in wanted]
    for a in adapters:
        sig_spec = SignalSpec(
            signal_id=a.signal_id,
            version=a.model_version,
            description=f"Model adapter signal from {a.model_family}/{a.model_id}",
            feature_ids=(),
            kind=SignalKind.CATEGORICAL,
            parameters={
                "adapter_id": a.adapter_id,
                "model_id": a.model_id,
                "model_family": a.model_family,
                "model_version": a.model_version,
                "timeframe": a.timeframe,
                "horizon": a.horizon,
                "holding_bars": a.holding_bars,
                "train_frac": 0.5,
                "use_cache": True,
                "source": "model_adapter",
            },
            family="model_adapter",
            holding_bars=a.holding_bars,
            allow_short=a.signal_mapping.allow_short,
        )
        registry.register(sig_spec, _make_fn(a.adapter_id), overwrite=overwrite)
        specs_out.append(sig_spec)
    return specs_out


def attach_precomputed_signal(frame: pd.DataFrame, adapter_id: str, signal: pd.Series) -> pd.DataFrame:
    """Attach a precomputed model signal column for registry generate() without re-fitting."""
    out = frame.copy()
    out[f"__model_signal__{adapter_id}"] = signal.reindex(out.index).to_numpy()
    return out


def ensure_adapter_registered(adapter_id: str) -> Any:
    register_default_adapters(overwrite=True)
    return get_adapter(adapter_id)


__all__ = [
    "attach_precomputed_signal",
    "clear_model_signal_cache",
    "ensure_adapter_registered",
    "register_model_adapter_signals",
]
