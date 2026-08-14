"""Attribution utilities wrapping explainability interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.explainability.importance import (
    ExplanationResult,
    explain_model,
    integrated_gradients_interface,
    shap_interface,
)


def attribute(
    model: Any,
    frame: pl.DataFrame,
    feature_columns: list[str],
    *,
    method: str = "shap",
) -> ExplanationResult:
    return explain_model(model, frame, feature_columns, method=method)


def top_k_features(result: ExplanationResult, k: int = 5) -> list[tuple[str, float]]:
    items = sorted(result.importances.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return items[: max(int(k), 0)]


def attribution_matrix(result: ExplanationResult) -> np.ndarray:
    if result.attributions is not None:
        return np.asarray(result.attributions, dtype=np.float64)
    # synthesize from importances
    cols = list(result.importances)
    return np.asarray([result.importances[c] for c in cols], dtype=np.float64).reshape(1, -1)


def compare_attributions(a: ExplanationResult, b: ExplanationResult) -> dict[str, float]:
    keys = sorted(set(a.importances) | set(b.importances))
    return {k: float(a.importances.get(k, 0.0) - b.importances.get(k, 0.0)) for k in keys}


# re-export hooks for convenience
__all__ = [
    "attribute",
    "attribution_matrix",
    "compare_attributions",
    "integrated_gradients_interface",
    "shap_interface",
    "top_k_features",
]
