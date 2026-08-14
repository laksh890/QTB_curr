"""Regime output → signal mapping (informational unless explicitly mapped)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.adapters.types import SignalMappingConfig


def regime_states_to_signal(
    state_ids: np.ndarray | pd.Series,
    mapping: SignalMappingConfig,
    *,
    state_names: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    """Map regime state ids/labels to LONG/SHORT/FLAT via configured regime_map."""
    ids = np.asarray(state_ids).reshape(-1)
    out = np.zeros(ids.size, dtype=np.float64)
    rmap = dict(mapping.regime_map or {})
    for i, sid in enumerate(ids):
        key = str(sid)
        if state_names is not None:
            try:
                idx = int(sid)
                if 0 <= idx < len(state_names):
                    key = str(state_names[idx])
            except Exception:  # noqa: BLE001
                key = str(sid)
        if key in rmap:
            out[i] = float(rmap[key])
        elif str(int(sid)) if str(sid).lstrip("-").isdigit() else key in rmap:
            out[i] = float(rmap.get(str(int(sid)), 0.0))
        elif mapping.flat_on_unknown_regime:
            out[i] = 0.0
        else:
            out[i] = np.nan
        if not mapping.allow_short and out[i] < 0:
            out[i] = 0.0
    return out


def regime_result_to_signal_series(
    state_ids: np.ndarray,
    index: pd.Index,
    mapping: SignalMappingConfig,
    *,
    state_names: list[str] | None = None,
) -> pd.Series:
    arr = regime_states_to_signal(state_ids, mapping, state_names=state_names)
    if arr.size != len(index):
        # align by truncation/pad
        out = np.full(len(index), np.nan if mapping.flat_on_unknown_regime else 0.0)
        n = min(arr.size, len(index))
        out[:n] = arr[:n]
        arr = out
    return pd.Series(arr, index=index, dtype=np.float64)


def regime_probabilities_to_confidence(proba: np.ndarray) -> np.ndarray:
    """Per-bar confidence = max state probability (diagnostic, not a trade rule)."""
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim == 1:
        return p
    return np.max(p, axis=1)


__all__ = [
    "regime_probabilities_to_confidence",
    "regime_result_to_signal_series",
    "regime_states_to_signal",
]
