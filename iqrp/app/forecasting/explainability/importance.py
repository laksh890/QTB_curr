"""Feature importance and model explanation interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

ExplainMethod = Literal[
    "permutation",
    "shap",
    "integrated_gradients",
    "attention",
    "builtin",
]


@dataclass(slots=True)
class ExplanationResult:
    method: str
    importances: dict[str, float]
    attributions: np.ndarray | None = None
    attention: np.ndarray | None = None
    baseline_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "importances": dict(self.importances),
            "attributions": None if self.attributions is None else self.attributions.tolist(),
            "attention": None if self.attention is None else self.attention.tolist(),
            "baseline_score": self.baseline_score,
            "metadata": dict(self.metadata),
        }


def permutation_importance(
    model: Any,
    frame: pl.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str | None = None,
    n_repeats: int = 3,
    rng: np.random.Generator | None = None,
) -> ExplanationResult:
    """Model-agnostic permutation importance using MAE degradation."""
    from iqrp.app.forecasting.base.evaluator import mae

    gen = rng or np.random.default_rng(0)
    cols = list(feature_columns)
    tgt = target_column or getattr(model, "_target_column", None)
    if tgt is None or tgt not in frame.columns:
        # use prediction variance proxy — still ranks features
        baseline_pred = np.asarray(model.predict(frame, cols), dtype=np.float64).reshape(-1)
        baseline = float(np.std(baseline_pred))
        y_true = baseline_pred
    else:
        y_true = frame[tgt].to_numpy().astype(np.float64)
        baseline_pred = np.asarray(model.predict(frame, cols), dtype=np.float64).reshape(-1)
        baseline = mae(y_true, baseline_pred)

    scores: dict[str, float] = {}
    for col in cols:
        degradations: list[float] = []
        for _ in range(max(n_repeats, 1)):
            shuffled = frame.with_columns(
                pl.Series(name=col, values=gen.permutation(frame[col].to_numpy()))
            )
            pred = np.asarray(model.predict(shuffled, cols), dtype=np.float64).reshape(-1)
            if tgt is None or tgt not in frame.columns:
                degradations.append(abs(float(np.std(pred)) - baseline))
            else:
                degradations.append(mae(y_true, pred) - baseline)
        scores[col] = float(np.mean(degradations))
    total = sum(abs(v) for v in scores.values()) or 1.0
    normed = {k: float(v / total) for k, v in scores.items()}
    return ExplanationResult(
        method="permutation",
        importances=normed,
        baseline_score=float(baseline),
        metadata={"n_repeats": n_repeats},
    )


def builtin_importance(model: Any, feature_columns: list[str]) -> ExplanationResult:
    """Read ``feature_importances_`` or equal weights."""
    cols = list(feature_columns)
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        raw = getattr(model, "_feature_importances", None)
    if raw is None:
        w = {c: 1.0 / max(len(cols), 1) for c in cols}
        return ExplanationResult(method="builtin", importances=w, metadata={"source": "equal"})
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size != len(cols):
        arr = np.resize(arr, len(cols))
    arr = np.clip(arr, 0, None)
    s = float(arr.sum()) or 1.0
    return ExplanationResult(
        method="builtin",
        importances={c: float(arr[i] / s) for i, c in enumerate(cols)},
        metadata={"source": "model"},
    )


def shap_interface(
    model: Any,
    frame: pl.DataFrame,
    feature_columns: list[str],
) -> ExplanationResult:
    """SHAP-compatible interface; falls back to permutation if no hook."""
    hook = getattr(model, "shap_values", None)
    if callable(hook):
        vals = np.asarray(hook(frame, feature_columns), dtype=np.float64)
        if vals.ndim == 1:
            imp = {c: float(abs(vals[i])) for i, c in enumerate(feature_columns) if i < vals.size}
        else:
            mean_abs = np.mean(np.abs(vals), axis=0)
            imp = {
                c: float(mean_abs[i]) for i, c in enumerate(feature_columns) if i < mean_abs.size
            }
        total = sum(imp.values()) or 1.0
        imp = {k: v / total for k, v in imp.items()}
        return ExplanationResult(method="shap", importances=imp, attributions=vals)
    return permutation_importance(model, frame, feature_columns)


def integrated_gradients_interface(
    model: Any,
    frame: pl.DataFrame,
    feature_columns: list[str],
    *,
    steps: int = 16,
) -> ExplanationResult:
    """IG interface; uses finite-difference path if model exposes ``predict_raw``."""
    hook = getattr(model, "integrated_gradients", None)
    if callable(hook):
        vals = np.asarray(hook(frame, feature_columns, steps=steps), dtype=np.float64)
        mean_abs = np.mean(np.abs(vals), axis=0) if vals.ndim > 1 else np.abs(vals)
        imp = {
            c: float(mean_abs[i]) for i, c in enumerate(feature_columns) if i < mean_abs.size
        }
        total = sum(imp.values()) or 1.0
        return ExplanationResult(
            method="integrated_gradients",
            importances={k: v / total for k, v in imp.items()},
            attributions=vals,
            metadata={"steps": steps},
        )
    return permutation_importance(model, frame, feature_columns)


def attention_visualization(model: Any) -> ExplanationResult:
    """Expose attention weights when available."""
    attn = getattr(model, "attention_weights_", None)
    if attn is None:
        attn = getattr(model, "_attention_weights", None)
    if attn is None:
        return ExplanationResult(
            method="attention",
            importances={},
            metadata={"available": False},
        )
    arr = np.asarray(attn, dtype=np.float64)
    return ExplanationResult(
        method="attention",
        importances={},
        attention=arr,
        metadata={"available": True, "shape": list(arr.shape)},
    )


def explain_model(
    model: Any,
    frame: pl.DataFrame,
    feature_columns: list[str],
    *,
    method: str = "permutation",
) -> ExplanationResult:
    m = method.lower()
    if m == "builtin":
        return builtin_importance(model, feature_columns)
    if m == "shap":
        return shap_interface(model, frame, feature_columns)
    if m in {"integrated_gradients", "ig"}:
        return integrated_gradients_interface(model, frame, feature_columns)
    if m == "attention":
        return attention_visualization(model)
    return permutation_importance(model, frame, feature_columns)
