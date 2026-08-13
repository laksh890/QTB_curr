"""Feature importance, SHAP approximations, PDP / ICE."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.forecasting.tree_models.base.backends import (
    estimator_feature_importances,
    estimator_predict,
)


def compute_feature_importance(
    estimator: Any,
    feature_names: list[str],
    *,
    kind: Literal["gain", "split", "permutation", "shap"] = "gain",
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    model: Any = None,
) -> dict[str, float]:
    names = list(feature_names)
    n = len(names)
    if kind in {"gain", "split"}:
        imp = estimator_feature_importances(estimator, n)
        # split count proxy: rank of gain
        if kind == "split":
            ranks = np.argsort(np.argsort(-imp)) + 1
            imp = ranks.astype(np.float64)
            imp = imp / (imp.sum() or 1.0)
        return {names[i]: float(imp[i]) for i in range(n)}
    if kind == "permutation" and X is not None and y is not None:
        base = float(np.mean((y - estimator_predict(estimator, X)) ** 2))
        rng = np.random.default_rng(0)
        scores = {}
        for j, name in enumerate(names):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            err = float(np.mean((y - estimator_predict(estimator, Xp)) ** 2))
            scores[name] = max(err - base, 0.0)
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}
    if kind == "shap" and X is not None:
        sv = shap_values(estimator, X, feature_names=names)
        means = np.mean(np.abs(sv), axis=0)
        total = float(means.sum()) or 1.0
        return {names[i]: float(means[i] / total) for i in range(n)}
    imp = estimator_feature_importances(estimator, n)
    return {names[i]: float(imp[i]) for i in range(n)}


def shap_values(
    estimator: Any,
    X: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Interventional SHAP approximation via Kernel-SHAP style sampling / Tree path fallback."""
    X = np.asarray(X, dtype=np.float64)
    # try shap library
    try:
        import shap  # type: ignore

        name = type(estimator).__name__
        if hasattr(estimator, "get_booster") or name.startswith(
            ("XGB", "LGBM", "CatBoost", "HistGradient", "RandomForest", "ExtraTrees")
        ):
            explainer = shap.TreeExplainer(estimator)
            vals = explainer.shap_values(X)
            if isinstance(vals, list):
                vals = vals[-1]
            return np.asarray(vals, dtype=np.float64)
    except Exception:  # noqa: BLE001
        pass
    return _kernel_shap_approx(estimator, X, background=background)



def shap_interaction_values(estimator: Any, X: np.ndarray, *, max_rows: int = 50) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)[:max_rows]
    n, p = X.shape
    base = shap_values(estimator, X)
    interactions = np.zeros((n, p, p))
    # approximate interactions as outer product of SHAP with sign
    for i in range(n):
        interactions[i] = np.outer(base[i], base[i]) / max(np.linalg.norm(base[i]), 1e-8)
    return interactions


def partial_dependence(
    estimator: Any,
    X: np.ndarray,
    feature_index: int,
    *,
    grid_size: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    grid = np.linspace(np.nanmin(X[:, feature_index]), np.nanmax(X[:, feature_index]), grid_size)
    preds = []
    for v in grid:
        Xp = X.copy()
        Xp[:, feature_index] = v
        preds.append(float(np.mean(estimator_predict(estimator, Xp))))
    return grid, np.asarray(preds, dtype=np.float64)


def ice_curves(
    estimator: Any,
    X: np.ndarray,
    feature_index: int,
    *,
    grid_size: int = 15,
    max_curves: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)[:max_curves]
    grid = np.linspace(np.nanmin(X[:, feature_index]), np.nanmax(X[:, feature_index]), grid_size)
    curves = np.zeros((X.shape[0], grid_size))
    for i in range(X.shape[0]):
        for j, v in enumerate(grid):
            row = X[i : i + 1].copy()
            row[0, feature_index] = v
            curves[i, j] = float(estimator_predict(estimator, row)[0])
    return grid, curves


def decision_paths(estimator: Any, X: np.ndarray, *, max_rows: int = 5) -> list[dict[str, Any]]:
    """Best-effort decision path extraction for tree estimators."""
    X = np.asarray(X, dtype=np.float64)[:max_rows]
    paths: list[dict[str, Any]] = []
    if hasattr(estimator, "decision_path") and hasattr(estimator, "estimators_"):
        try:
            # RF-like: use first tree
            tree = estimator.estimators_[0]
            node_indicator = tree.decision_path(X)
            for i in range(X.shape[0]):
                nodes = node_indicator[i].indices.tolist()
                paths.append({"row": i, "nodes": nodes, "prediction": float(estimator_predict(estimator, X[i : i + 1])[0])})
            return paths
        except Exception:  # noqa: BLE001
            pass
    for i in range(X.shape[0]):
        paths.append(
            {
                "row": i,
                "nodes": [],
                "prediction": float(estimator_predict(estimator, X[i : i + 1])[0]),
                "contribution": estimator_feature_importances(estimator, X.shape[1]).tolist(),
            }
        )
    return paths


def _kernel_shap_approx(
    estimator: Any,
    X: np.ndarray,
    *,
    background: np.ndarray | None = None,
    n_samples: int = 64,
) -> np.ndarray:
    """Sampling-based interventional attribution (KernelSHAP-like)."""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    bg = background if background is not None else X[: min(50, n)]
    bg_mean = np.mean(bg, axis=0)
    base = float(np.mean(estimator_predict(estimator, np.broadcast_to(bg_mean, (1, p)))))
    rng = np.random.default_rng(0)
    phi = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            # with/without feature j
            diffs = []
            for _ in range(max(n_samples // max(p, 1), 4)):
                mask = rng.random(p) > 0.5
                mask[j] = False
                x0 = bg_mean.copy()
                x1 = bg_mean.copy()
                x0[mask] = X[i, mask]
                x1[mask] = X[i, mask]
                x1[j] = X[i, j]
                y0 = float(estimator_predict(estimator, x0.reshape(1, -1))[0])
                y1 = float(estimator_predict(estimator, x1.reshape(1, -1))[0])
                diffs.append(y1 - y0)
            phi[i, j] = float(np.mean(diffs))
        # adjust to efficiency
        pred = float(estimator_predict(estimator, X[i : i + 1])[0])
        gap = pred - base - float(np.sum(phi[i]))
        phi[i] += gap / max(p, 1)
    return phi
