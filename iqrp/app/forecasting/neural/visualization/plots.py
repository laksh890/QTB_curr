"""Visualization helpers for neural forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:  # noqa: BLE001  # pragma: no cover
        return None


def plot_forecast(y_true: np.ndarray | None, y_pred: np.ndarray, *, intervals: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "y_pred": np.asarray(y_pred).reshape(-1).tolist(),
        "y_true": None if y_true is None else np.asarray(y_true).reshape(-1).tolist(),
    }
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(payload["y_pred"], label="forecast")
    if y_true is not None:
        ax.plot(np.asarray(y_true).reshape(-1), label="actual", alpha=0.7)
    if intervals is not None:
        lo, hi = intervals
        ax.fill_between(np.arange(len(lo)), lo, hi, alpha=0.2)
    ax.legend()
    ax.set_title("Neural Forecast")
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_training_curves(history: dict[str, list[float]]) -> dict[str, Any]:
    payload = dict(history)
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(7, 3))
    if history.get("train_loss"):
        ax.plot(history["train_loss"], label="train")
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="val")
    ax.legend()
    ax.set_title("Training Curves")
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_attention(attn: np.ndarray) -> dict[str, Any]:
    a = np.asarray(attn, dtype=np.float64)
    payload = {"attention": a.tolist()}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(a if a.ndim == 2 else a.mean(0), aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title("Attention Map")
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_attribution(attr: np.ndarray, feature_names: list[str] | None = None) -> dict[str, Any]:
    a = np.asarray(attr, dtype=np.float64)
    if a.ndim == 3:
        scores = np.mean(np.abs(a), axis=(0, 1))
    elif a.ndim == 2:
        scores = np.mean(np.abs(a), axis=0)
    else:
        scores = np.abs(a)
    names = feature_names or [f"f{i}" for i in range(scores.size)]
    payload = {"names": list(names[: scores.size]), "scores": scores.tolist()}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(payload["names"][::-1], payload["scores"][::-1])
    ax.set_title("Feature Attribution")
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


def plot_loss_curve(losses: list[float], *, title: str = "Loss") -> dict[str, Any]:
    payload = {"losses": list(losses)}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses)
    ax.set_title(title)
    payload["figure"] = fig
    plt.close(fig)
    return payload
