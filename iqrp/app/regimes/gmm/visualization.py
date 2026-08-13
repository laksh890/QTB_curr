"""SVG visualizations for Gaussian mixture regime detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.regimes.gmm.config import GMMSettings


def _ensure(path: Path, settings: GMMSettings) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return False
    return True


def plot_likelihood_curve(
    history: Any,
    path: Path,
    settings: GMMSettings | None = None,
    *,
    title: str = "Log-Likelihood Curve",
) -> Path:
    settings = settings or GMMSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    v = np.asarray(history, dtype=np.float64).reshape(-1)
    v = v[: settings.visualization.max_points]
    width, height = 640, 220
    if v.size == 0:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
        )
        return path
    lo, hi = float(v.min()), float(v.max())
    span = max(hi - lo, 1e-9)
    pts = []
    for i, val in enumerate(v):
        x = 40 + (width - 60) * i / max(v.size - 1, 1)
        y = height - 30 - (height - 50) * (val - lo) / span
        pts.append(f"{x:.1f},{y:.1f}")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
        f'<polyline fill="none" stroke="#1f4e79" stroke-width="1.5" points="{" ".join(pts)}"/>',
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def plot_cluster_scatter(
    x: Any,
    labels: Any,
    path: Path,
    settings: GMMSettings | None = None,
    *,
    title: str = "Cluster Scatter",
) -> Path:
    settings = settings or GMMSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = np.column_stack([np.arange(y.size), y])
    labs = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = min(y.shape[0], settings.visualization.max_points)
    y, labs = y[:n], labs[:n]
    width, height = 520, 400
    x0, x1 = float(y[:, 0].min()), float(y[:, 0].max())
    y0, y1 = float(y[:, 1].min()) if y.shape[1] > 1 else 0.0, (
        float(y[:, 1].max()) if y.shape[1] > 1 else 1.0
    )
    sx = max(x1 - x0, 1e-9)
    sy = max(y1 - y0, 1e-9)
    colors = ["#1d3557", "#e63946", "#2a9d8f", "#457b9d", "#f4a261", "#9b5de5"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i in range(n):
        px = 40 + (width - 60) * (y[i, 0] - x0) / sx
        py = height - 30 - (height - 50) * ((y[i, 1] if y.shape[1] > 1 else 0.0) - y0) / sy
        col = colors[int(labs[i]) % len(colors)]
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{col}" opacity="0.8"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_probability_heatmap(
    responsibilities: Any,
    path: Path,
    settings: GMMSettings | None = None,
    *,
    title: str = "Posterior Probability Heatmap",
) -> Path:
    settings = settings or GMMSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    p = np.asarray(responsibilities, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    t = min(p.shape[0], settings.visualization.max_points)
    k = p.shape[1]
    p = p[:t]
    cell_w = max(2.0, 680 / max(t, 1))
    cell_h = max(12.0, 200 / max(k, 1))
    width = int(40 + cell_w * t)
    height = int(40 + cell_h * k)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i in range(t):
        for j in range(k):
            v = float(np.clip(p[i, j], 0.0, 1.0))
            g = int(255 * (1.0 - v))
            parts.append(
                f'<rect x="{40 + i * cell_w:.2f}" y="{30 + j * cell_h:.2f}" '
                f'width="{cell_w:.2f}" height="{cell_h:.2f}" fill="rgb({g},{g},255)"/>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_regime_timeline(
    labels: Any,
    path: Path,
    settings: GMMSettings | None = None,
    *,
    title: str = "Regime Timeline",
) -> Path:
    settings = settings or GMMSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    s = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = s[: settings.visualization.max_points]
    width, height = 680, 160
    colors = ["#1d3557", "#e63946", "#2a9d8f", "#457b9d", "#f4a261"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    if s.size:
        w = (width - 40) / s.size
        for i, lab in enumerate(s):
            parts.append(
                f'<rect x="{40 + i * w:.2f}" y="40" width="{max(w, 1):.2f}" height="80" '
                f'fill="{colors[int(lab) % len(colors)]}"/>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_covariance_ellipses(
    means: Any,
    covars: Any,
    path: Path,
    settings: GMMSettings | None = None,
    *,
    title: str = "Covariance Ellipses",
) -> Path:
    settings = settings or GMMSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    mu = np.asarray(means, dtype=np.float64)
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
    width, height = 520, 400
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    if mu.shape[1] < 2:
        for i, m in enumerate(mu):
            parts.append(
                f'<circle cx="{80 + i * 60}" cy="{height/2}" r="20" '
                f'fill="none" stroke="#1d3557" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{70 + i * 60}" y="{height/2 + 40}" font-size="10">μ={m[0]:.2f}</text>'
            )
    else:
        xs, ys = mu[:, 0], mu[:, 1]
        x0, x1 = float(xs.min() - 1), float(xs.max() + 1)
        y0, y1 = float(ys.min() - 1), float(ys.max() + 1)
        for m in mu:
            px = 40 + (width - 60) * (m[0] - x0) / max(x1 - x0, 1e-9)
            py = height - 30 - (height - 50) * (m[1] - y0) / max(y1 - y0, 1e-9)
            parts.append(
                f'<ellipse cx="{px:.1f}" cy="{py:.1f}" rx="30" ry="20" '
                f'fill="none" stroke="#e63946" stroke-width="2"/>'
            )
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="#1d3557"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_component_weights(
    weights: Any,
    path: Path,
    settings: GMMSettings | None = None,
    *,
    title: str = "Component Weights",
) -> Path:
    settings = settings or GMMSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    width, height = 480, 220
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    if w.size:
        bar_w = (width - 60) / w.size
        for i, val in enumerate(w):
            bh = (height - 50) * float(val)
            parts.append(
                f'<rect x="{40 + i * bar_w:.2f}" y="{height - 30 - bh:.2f}" '
                f'width="{max(bar_w - 4, 1):.2f}" height="{bh:.2f}" fill="#2a6f97"/>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
