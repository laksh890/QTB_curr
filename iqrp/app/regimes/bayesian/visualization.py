"""SVG visualizations for Bayesian regime-switching inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.regimes.bayesian.config import BayesianSettings


def _ensure(path: Path, settings: BayesianSettings) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return False
    return True


def plot_trace(
    values: Any,
    path: Path,
    settings: BayesianSettings | None = None,
    *,
    title: str = "Trace Plot",
) -> Path:
    settings = settings or BayesianSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    v = np.asarray(values, dtype=np.float64).reshape(-1)
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


def plot_posterior_histogram(
    samples: Any,
    path: Path,
    settings: BayesianSettings | None = None,
    *,
    title: str = "Posterior Distribution",
    bins: int = 30,
) -> Path:
    settings = settings or BayesianSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    v = np.asarray(samples, dtype=np.float64).reshape(-1)
    width, height = 480, 240
    if v.size == 0:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
        )
        return path
    hist, _edges = np.histogram(v, bins=bins)
    m = max(int(hist.max()), 1)
    bar_w = (width - 60) / max(len(hist), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i, h in enumerate(hist):
        bh = (height - 50) * float(h) / m
        parts.append(
            f'<rect x="{40 + i * bar_w:.2f}" y="{height - 30 - bh:.2f}" '
            f'width="{max(bar_w - 1, 1):.2f}" height="{bh:.2f}" fill="#2a6f97"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_transition_uncertainty(
    mean_tm: Any,
    low: Any,
    high: Any,
    path: Path,
    settings: BayesianSettings | None = None,
    *,
    title: str = "Transition Uncertainty",
) -> Path:
    settings = settings or BayesianSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    m = np.asarray(mean_tm, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    k = m.shape[0]
    cell = 50
    width, height = 40 + cell * k, 40 + cell * k
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i in range(k):
        for j in range(k):
            v = float(np.clip(m[i, j], 0, 1))
            g = int(255 * (1 - v))
            width_band = float(hi[i, j] - lo[i, j])
            parts.append(
                f'<rect x="{40 + j * cell}" y="{30 + i * cell}" width="{cell - 2}" '
                f'height="{cell - 2}" fill="rgb({g},{g},255)" stroke="#333"/>'
            )
            parts.append(
                f'<text x="{40 + j * cell + 4}" y="{30 + i * cell + 20}" font-size="10">'
                f"{v:.2f}±{width_band/2:.2f}</text>"
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_credible_intervals(
    mean: Any,
    low: Any,
    high: Any,
    path: Path,
    settings: BayesianSettings | None = None,
    *,
    title: str = "Credible Intervals",
) -> Path:
    settings = settings or BayesianSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    lo = np.asarray(low, dtype=np.float64).reshape(-1)
    hi = np.asarray(high, dtype=np.float64).reshape(-1)
    width, height = 520, 220
    n = mu.size
    if n == 0:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
        )
        return path
    ymin, ymax = float(min(lo.min(), mu.min())), float(max(hi.max(), mu.max()))
    span = max(ymax - ymin, 1e-9)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    gap = (width - 60) / max(n, 1)
    for i in range(n):
        x = 50 + i * gap
        y_lo = height - 30 - (height - 50) * (lo[i] - ymin) / span
        y_hi = height - 30 - (height - 50) * (hi[i] - ymin) / span
        y_mu = height - 30 - (height - 50) * (mu[i] - ymin) / span
        parts.append(
            f'<line x1="{x}" y1="{y_lo}" x2="{x}" y2="{y_hi}" stroke="#333" stroke-width="2"/>'
        )
        parts.append(f'<circle cx="{x}" cy="{y_mu}" r="4" fill="#c1121f"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_regime_timeline(
    probabilities: Any,
    path: Path,
    settings: BayesianSettings | None = None,
    *,
    title: str = "Regime Probability Timeline",
) -> Path:
    settings = settings or BayesianSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    t = min(p.shape[0], settings.visualization.max_points)
    k = p.shape[1]
    p = p[:t]
    width, height = 680, 220
    colors = ["#1d3557", "#457b9d", "#a8dadc", "#e63946", "#2a9d8f"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for j in range(k):
        pts = []
        for i in range(t):
            x = 40 + (width - 60) * i / max(t - 1, 1)
            y = height - 30 - (height - 50) * float(p[i, j])
            pts.append(f"{x:.1f},{y:.1f}")
        parts.append(
            f'<polyline fill="none" stroke="{colors[j % len(colors)]}" '
            f'stroke-width="1.5" points="{" ".join(pts)}"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_posterior_predictive_check(
    observed: Any,
    predictive: Any,
    path: Path,
    settings: BayesianSettings | None = None,
    *,
    title: str = "Posterior Predictive Check",
) -> Path:
    settings = settings or BayesianSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    pred = np.asarray(predictive, dtype=np.float64)
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[0], -1)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    q_low = np.percentile(pred, 5, axis=0)
    q_hi = np.percentile(pred, 95, axis=0)
    q_mid = np.percentile(pred, 50, axis=0)
    n = min(y.size, q_mid.size, settings.visualization.max_points)
    width, height = 680, 240
    all_v = np.concatenate([y[:n], q_low[:n], q_hi[:n]])
    lo, hi = float(all_v.min()), float(all_v.max())
    span = max(hi - lo, 1e-9)

    def _y(val: float) -> float:
        return height - 30 - (height - 50) * (val - lo) / span

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    band = []
    for i in range(n):
        x = 40 + (width - 60) * i / max(n - 1, 1)
        band.append((x, _y(q_hi[i]), _y(q_low[i])))
    for i in range(n - 1):
        x0, y_hi0, y_lo0 = band[i]
        x1, y_hi1, y_lo1 = band[i + 1]
        parts.append(
            f'<polygon points="{x0},{y_hi0} {x1},{y_hi1} {x1},{y_lo1} {x0},{y_lo0}" '
            f'fill="#a8dadc" opacity="0.5"/>'
        )
    obs_pts = []
    mid_pts = []
    for i in range(n):
        x = 40 + (width - 60) * i / max(n - 1, 1)
        obs_pts.append(f"{x:.1f},{_y(float(y[i])):.1f}")
        mid_pts.append(f"{x:.1f},{_y(float(q_mid[i])):.1f}")
    parts.append(
        f'<polyline fill="none" stroke="#1d3557" stroke-width="1.5" points="{" ".join(obs_pts)}"/>'
    )
    parts.append(
        f'<polyline fill="none" stroke="#e63946" stroke-width="1.2" '
        f'stroke-dasharray="4 2" points="{" ".join(mid_pts)}"/>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
