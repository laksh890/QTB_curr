"""Visualization helpers for tree forecasting (matplotlib optional)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:  # noqa: BLE001
        return None


def plot_feature_importance(importances: dict[str, float], *, title: str = "Feature Importance") -> dict[str, Any]:
    items = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)
    payload = {"names": [k for k, _ in items], "values": [float(v) for _, v in items], "title": title}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(payload["names"][::-1], payload["values"][::-1])
    ax.set_title(title)
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_shap_summary(shap_vals: np.ndarray, feature_names: list[str] | None = None) -> dict[str, Any]:

    sv = np.asarray(shap_vals, dtype=np.float64)
    means = np.mean(np.abs(sv), axis=0)
    names = feature_names or [f"f{i}" for i in range(means.size)]
    payload = {"mean_abs_shap": means.tolist(), "names": list(names)}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(7, 4))
    order = np.argsort(means)
    ax.barh([names[i] for i in order], means[order])
    ax.set_title("SHAP Summary")
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_dependence(grid: np.ndarray, values: np.ndarray, *, title: str = "Partial Dependence") -> dict[str, Any]:
    payload = {"grid": np.asarray(grid).tolist(), "values": np.asarray(values).tolist(), "title": title}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(grid, values)
    ax.set_title(title)
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_prediction_error(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    payload = {"y_true": yt.tolist(), "y_pred": yp.tolist()}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, s=10, alpha=0.6)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "k--")
    ax.set_title("Prediction Error")
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_calibration(mean_predicted: list[float], fraction_positive: list[float]) -> dict[str, Any]:
    payload = {"mean_predicted": list(mean_predicted), "fraction_positive": list(fraction_positive)}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_predicted, fraction_positive, "o-")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("Calibration")
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_learning_curve(curve: dict[str, list[float]]) -> dict[str, Any]:
    payload = dict(curve)
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(curve.get("train_sizes", []), curve.get("train_rmse", []), label="train")
    ax.plot(curve.get("train_sizes", []), curve.get("val_rmse", []), label="val")
    ax.legend()
    ax.set_title("Learning Curve")
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_residual_distribution(residuals: np.ndarray) -> dict[str, Any]:
    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    payload = {"residuals": r.tolist()}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(r, bins=30, density=True)
    ax.set_title("Residual Distribution")
    payload["figure"] = fig
    plt.close(fig)
    return payload
