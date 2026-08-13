"""Visualization helpers for transformer forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:  # noqa: BLE001  # pragma: no cover
        return None


def plot_attention_map(attn: np.ndarray) -> dict[str, Any]:
    a = np.asarray(attn, dtype=np.float64)
    if a.ndim > 2:
        a = a.mean(axis=tuple(range(a.ndim - 2)))
    payload: dict[str, Any] = {"attention": a.tolist()}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(a, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title("Attention Map")
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_forecast(y_pred: np.ndarray, *, y_true: np.ndarray | None = None, bands: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"y_pred": np.asarray(y_pred).reshape(-1).tolist()}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(payload["y_pred"], label="forecast")
    if y_true is not None:
        ax.plot(np.asarray(y_true).reshape(-1), label="actual", alpha=0.7)
    if bands is not None:
        lo, hi = bands
        ax.fill_between(np.arange(len(lo)), lo, hi, alpha=0.2)
    ax.legend()
    ax.set_title("Transformer Forecast")
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_embedding_projection(emb: np.ndarray) -> dict[str, Any]:
    e = np.asarray(emb, dtype=np.float64)
    if e.ndim > 2:
        e = e.reshape(e.shape[0], -1)
    # PCA-2 via SVD
    e = e - e.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(e, full_matrices=False)
    proj = e @ vt[:2].T if vt.shape[0] >= 2 else e[:, :1]
    payload: dict[str, Any] = {"projection": proj.tolist()}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(proj[:, 0], proj[:, 1] if proj.shape[1] > 1 else np.zeros(proj.shape[0]), s=12)
    ax.set_title("Embedding Projection")
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_residual_distribution(residuals: np.ndarray) -> dict[str, Any]:
    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    payload: dict[str, Any] = {"residuals": r.tolist()}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(r, bins=30, density=True)
    ax.set_title("Residual Distribution")
    payload["figure"] = fig
    plt.close(fig)
    return payload


def plot_calibration_curve(levels: list[float], coverage: list[float]) -> dict[str, Any]:
    payload: dict[str, Any] = {"levels": list(levels), "coverage": list(coverage)}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(levels, coverage, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Calibration")
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
