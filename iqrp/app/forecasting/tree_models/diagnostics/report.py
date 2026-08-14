"""Tree model diagnostics: learning curves, calibration, drift, residuals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.forecasting.tree_models.base.backends import create_estimator, estimator_predict


@dataclass(slots=True)
class TreeDiagnosticReport:
    residual_mean: float
    residual_std: float
    residual_skew: float
    learning_curve: dict[str, list[float]]
    validation_curve: dict[str, list[float]]
    calibration_curve: dict[str, list[float]]
    feature_stability: dict[str, float]
    prediction_drift: float
    bias: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "residual_skew": self.residual_skew,
            "learning_curve": {k: list(v) for k, v in self.learning_curve.items()},
            "validation_curve": {k: list(v) for k, v in self.validation_curve.items()},
            "calibration_curve": {k: list(v) for k, v in self.calibration_curve.items()},
            "feature_stability": dict(self.feature_stability),
            "prediction_drift": self.prediction_drift,
            "bias": self.bias,
            "metadata": dict(self.metadata),
        }


def run_tree_diagnostics(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    backend: str = "hist_gradient_boosting",
    task: str = "regression",
    params: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
) -> TreeDiagnosticReport:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = estimator_predict(estimator, X)
    resid = y - pred
    mean = float(np.mean(resid))
    std = float(np.std(resid))
    skew = float(np.mean(((resid - mean) / std) ** 3)) if std > 1e-12 else 0.0
    lc = learning_curve(backend, X, y, task=task, params=params or {})
    vc = validation_curve(backend, X, y, task=task, params=params or {})
    cc = calibration_curve(y, pred)
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    stability = feature_stability(backend, X, y, names, task=task, params=params or {})
    # drift: first half vs second half prediction mean shift
    mid = X.shape[0] // 2
    drift = float(abs(np.mean(pred[mid:]) - np.mean(pred[:mid]))) if mid else 0.0
    return TreeDiagnosticReport(
        residual_mean=mean,
        residual_std=std,
        residual_skew=skew,
        learning_curve=lc,
        validation_curve=vc,
        calibration_curve=cc,
        feature_stability=stability,
        prediction_drift=drift,
        bias=mean,
        metadata={"n": int(y.size), "backend": backend},
    )


def learning_curve(
    backend: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    params: dict[str, Any],
    train_sizes: list[float] | None = None,
) -> dict[str, list[float]]:
    sizes = train_sizes or [0.3, 0.5, 0.7, 0.9]
    n = X.shape[0]
    split = int(n * 0.8)
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    train_scores, val_scores, xs = [], [], []
    for frac in sizes:
        m = max(int(Xtr.shape[0] * frac), 10)
        xs.append(float(m))
        try:
            est = create_estimator(backend, task=task, params=params)  # type: ignore[arg-type]
            est.fit(Xtr[:m], ytr[:m])
            train_scores.append(
                float(np.sqrt(np.mean((ytr[:m] - estimator_predict(est, Xtr[:m])) ** 2)))
            )
            val_scores.append(float(np.sqrt(np.mean((yte - estimator_predict(est, Xte)) ** 2))))
        except Exception:
            train_scores.append(float("nan"))
            val_scores.append(float("nan"))
    return {"train_sizes": xs, "train_rmse": train_scores, "val_rmse": val_scores}


def validation_curve(
    backend: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    params: dict[str, Any],
    param_name: str = "max_depth",
    param_range: list[Any] | None = None,
) -> dict[str, list[float]]:
    values = param_range or [2, 3, 4, 6, 8]
    split = int(X.shape[0] * 0.8)
    scores = []
    for v in values:
        p = dict(params)
        p[param_name] = v
        try:
            est = create_estimator(backend, task=task, params=p)  # type: ignore[arg-type]
            est.fit(X[:split], y[:split])
            pred = estimator_predict(est, X[split:])
            scores.append(float(np.sqrt(np.mean((y[split:] - pred) ** 2))))
        except Exception:
            scores.append(float("nan"))
    return {"param": param_name, "values": [float(v) for v in values], "val_rmse": scores}


def calibration_curve(
    y: np.ndarray, pred: np.ndarray, *, n_bins: int = 10
) -> dict[str, list[float]]:
    # for regression: reliability of probabilistic bins of ranked predictions
    order = np.argsort(pred)
    y_s, p_s = y[order], pred[order]
    bins = np.array_split(np.arange(y_s.size), n_bins)
    frac_pos, mean_pred = [], []
    for b in bins:
        if b.size == 0:
            continue
        frac_pos.append(float(np.mean(y_s[b])))
        mean_pred.append(float(np.mean(p_s[b])))
    return {"mean_predicted": mean_pred, "fraction_positive": frac_pos}


def feature_stability(
    backend: str,
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    task: str,
    params: dict[str, Any],
    n_boots: int = 5,
) -> dict[str, float]:
    from iqrp.app.forecasting.tree_models.base.backends import estimator_feature_importances

    rng = np.random.default_rng(0)
    mat = []
    n = X.shape[0]
    for _ in range(n_boots):
        idx = rng.integers(0, n, size=n)
        try:
            est = create_estimator(backend, task=task, params=params)  # type: ignore[arg-type]
            est.fit(X[idx], y[idx])
            mat.append(estimator_feature_importances(est, X.shape[1]))
        except Exception:
            mat.append(np.ones(X.shape[1]) / X.shape[1])
    arr = np.stack(mat)
    stab = 1.0 - np.std(arr, axis=0) / (np.mean(arr, axis=0) + 1e-12)
    return {names[i]: float(np.clip(stab[i], 0, 1)) for i in range(len(names))}
