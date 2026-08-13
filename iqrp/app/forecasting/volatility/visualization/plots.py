"""Volatility visualization helpers (matplotlib optional)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:  # noqa: BLE001
        return None


def _pyplot():
    return _import_pyplot()


def _pyplot_available() -> bool:
    """Expose import success for tests without requiring a display backend."""
    return _pyplot() is not None


def plot_volatility_forecast(
    sigma: np.ndarray,
    *,
    forecast: np.ndarray | None = None,
    title: str = "Volatility Forecast",
    max_points: int = 500,
) -> dict[str, Any]:
    s = np.asarray(sigma, dtype=np.float64).reshape(-1)
    if s.size > max_points:
        idx = np.linspace(0, s.size - 1, max_points).astype(int)
        s = s[idx]
    payload = {
        "title": title,
        "in_sample": s.tolist(),
        "forecast": None if forecast is None else np.asarray(forecast).reshape(-1).tolist(),
    }
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(s, label="conditional vol")
    if forecast is not None:
        f = np.asarray(forecast, dtype=np.float64).reshape(-1)
        ax.plot(np.arange(s.size, s.size + f.size), f, label="forecast")
    ax.set_title(title)
    ax.legend()
    payload["figure"] = fig
    return payload


def plot_conditional_variance(variance: np.ndarray, *, max_points: int = 500) -> dict[str, Any]:
    v = np.asarray(variance, dtype=np.float64).reshape(-1)
    if v.size > max_points:
        v = v[np.linspace(0, v.size - 1, max_points).astype(int)]
    payload = {"variance": v.tolist()}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(v)
    ax.set_title("Conditional Variance")
    payload["figure"] = fig
    return payload


def plot_residuals(z: np.ndarray, *, max_points: int = 500) -> dict[str, Any]:
    x = np.asarray(z, dtype=np.float64).reshape(-1)
    if x.size > max_points:
        x = x[np.linspace(0, x.size - 1, max_points).astype(int)]
    payload = {"residuals": x.tolist()}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(x)
    axes[0].set_title("Standardized Residuals")
    axes[1].hist(x, bins=30, density=True)
    axes[1].set_title("Residual Density")
    payload["figure"] = fig
    return payload


def plot_persistence(persistence: float, half_life: float) -> dict[str, Any]:
    payload = {"persistence": float(persistence), "half_life": float(half_life)}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(5, 3))
    xs = np.arange(0, 50)
    decay = persistence**xs
    ax.plot(xs, decay)
    ax.axhline(0.5, ls="--", color="gray")
    ax.set_title(f"Shock Decay (half-life≈{half_life:.1f})")
    payload["figure"] = fig
    return payload


def plot_correlation_evolution(corr: np.ndarray, *, i: int = 0, j: int = 1) -> dict[str, Any]:
    c = np.asarray(corr, dtype=np.float64)
    if c.ndim == 3:
        series = c[:, i, j]
    else:
        series = c.reshape(-1)
    payload = {"correlation": series.tolist()}
    plt = _pyplot()
    if plt is None:
        return payload
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(series)
    ax.set_title("Correlation Evolution")
    payload["figure"] = fig
    return payload
