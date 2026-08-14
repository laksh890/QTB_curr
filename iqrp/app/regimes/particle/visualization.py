"""SVG visualizations for particle filters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.regimes.particle.config import ParticleSettings


def _ensure(path: Path, settings: ParticleSettings) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return False
    return True


def _line_plot(
    series: list[tuple[np.ndarray, str]],
    path: Path,
    *,
    title: str,
    settings: ParticleSettings,
    bands: tuple[np.ndarray, np.ndarray] | None = None,
) -> Path:
    path = Path(path)
    if not _ensure(path, settings):
        return path
    width, height = 720, 260
    colors = ["#1d3557", "#e63946", "#2a9d8f", "#457b9d", "#f4a261"]
    max_n = settings.visualization.max_points
    cleaned = [(np.asarray(a, dtype=np.float64).reshape(-1)[:max_n], name) for a, name in series]
    nonempty = [v for v, _ in cleaned if v.size]
    all_vals = np.concatenate(nonempty) if nonempty else np.array([], dtype=np.float64)
    if bands is not None:
        lo_b = np.asarray(bands[0], dtype=np.float64).reshape(-1)[:max_n]
        hi_b = np.asarray(bands[1], dtype=np.float64).reshape(-1)[:max_n]
        if lo_b.size and hi_b.size:
            all_vals = (
                np.concatenate([all_vals, lo_b, hi_b])
                if all_vals.size
                else np.concatenate([lo_b, hi_b])
            )
    if all_vals.size == 0:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="10" y="18" font-size="14">{title}</text></svg>',
            encoding="utf-8",
        )
        return path
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    span = max(hi - lo, 1e-9)
    n = max((cleaned[0][0].size if cleaned else 1), 1)

    def _xy(i: int, val: float) -> tuple[float, float]:
        x = 40 + (width - 60) * i / max(n - 1, 1)
        y = height - 30 - (height - 50) * (val - lo) / span
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    if bands is not None:
        lo_b = np.asarray(bands[0], dtype=np.float64).reshape(-1)[:max_n]
        hi_b = np.asarray(bands[1], dtype=np.float64).reshape(-1)[:max_n]
        poly = []
        for i, val in enumerate(hi_b):
            x, y = _xy(i, float(val))
            poly.append(f"{x:.1f},{y:.1f}")
        for i in range(lo_b.size - 1, -1, -1):
            x, y = _xy(i, float(lo_b[i]))
            poly.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polygon fill="#a8dadc" opacity="0.45" points="{" ".join(poly)}"/>')
    for idx, (v, name) in enumerate(cleaned):
        if not v.size:
            continue
        pts = [
            f"{_xy(i, float(val))[0]:.1f},{_xy(i, float(val))[1]:.1f}" for i, val in enumerate(v)
        ]
        col = colors[idx % len(colors)]
        parts.append(
            f'<polyline fill="none" stroke="{col}" stroke-width="1.5" points="{" ".join(pts)}"/>'
        )
        parts.append(
            f'<text x="{width - 120}" y="{30 + 14 * idx}" font-size="11" fill="{col}">{name}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_particle_cloud(
    states: Any,
    weights: Any,
    path: Path,
    settings: ParticleSettings | None = None,
    *,
    title: str = "Particle Cloud",
) -> Path:
    settings = settings or ParticleSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    x = np.asarray(states, dtype=np.float64)
    if x.ndim == 1:
        x = np.column_stack([np.arange(x.size), x]) if x.size else np.zeros((0, 2))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    width, height = 520, 400
    if x.size == 0 or x.shape[0] == 0:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="10" y="18" font-size="14">{title}</text></svg>',
            encoding="utf-8",
        )
        return path
    n = min(x.shape[0], settings.visualization.max_particles_plot)
    x, w = x[:n], w[:n]
    x0, x1 = float(x[:, 0].min()), float(x[:, 0].max())
    y0, y1 = float(x[:, 1].min()) if x.shape[1] > 1 else 0.0, (
        float(x[:, 1].max()) if x.shape[1] > 1 else 1.0
    )
    sx, sy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    wmax = max(float(w.max()), 1e-12)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i in range(n):
        px = 40 + (width - 60) * (x[i, 0] - x0) / sx
        py = height - 30 - (height - 50) * ((x[i, 1] if x.shape[1] > 1 else 0.0) - y0) / sy
        r = 2 + 6 * (w[i] / wmax)
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}" fill="#1d3557" opacity="0.55"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_weight_histogram(
    weights: Any,
    path: Path,
    settings: ParticleSettings | None = None,
    *,
    title: str = "Weight Histogram",
    bins: int = 20,
) -> Path:
    settings = settings or ParticleSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    width, height = 520, 260
    if w.size == 0:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
        )
        return path
    hist, edges = np.histogram(w, bins=bins)
    hmax = max(int(hist.max()), 1)
    bar_w = (width - 60) / bins
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i, h in enumerate(hist):
        bh = (height - 50) * float(h) / hmax
        x = 40 + i * bar_w
        y = height - 30 - bh
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.9:.1f}" height="{bh:.1f}" fill="#457b9d"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_state_trajectory(
    means: Any,
    path: Path,
    settings: ParticleSettings | None = None,
    *,
    observations: Any | None = None,
    title: str = "State Trajectory",
) -> Path:
    settings = settings or ParticleSettings.default()
    m = np.asarray(means, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    series: list[tuple[np.ndarray, str]] = [(m[:, 0], "posterior")]
    if observations is not None:
        series.append((np.asarray(observations, dtype=np.float64).reshape(-1), "obs"))
    return _line_plot(series, Path(path), title=title, settings=settings)


def plot_credible_intervals(
    means: Any,
    lower: Any,
    upper: Any,
    path: Path,
    settings: ParticleSettings | None = None,
    *,
    title: str = "Credible Intervals",
) -> Path:
    settings = settings or ParticleSettings.default()
    m = np.asarray(means, dtype=np.float64).reshape(-1)
    return _line_plot(
        [(m, "mean")],
        Path(path),
        title=title,
        settings=settings,
        bands=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
    )


def plot_ess_timeline(
    ess: Any,
    path: Path,
    settings: ParticleSettings | None = None,
    *,
    title: str = "ESS Timeline",
) -> Path:
    settings = settings or ParticleSettings.default()
    return _line_plot(
        [(np.asarray(ess, dtype=np.float64).reshape(-1), "ESS")],
        Path(path),
        title=title,
        settings=settings,
    )


def plot_resampling_timeline(
    resampled: Any,
    path: Path,
    settings: ParticleSettings | None = None,
    *,
    title: str = "Resampling Timeline",
) -> Path:
    settings = settings or ParticleSettings.default()
    v = np.asarray(resampled, dtype=np.float64).reshape(-1)
    return _line_plot([(v, "resampled")], Path(path), title=title, settings=settings)


def plot_posterior_evolution(
    means: Any,
    path: Path,
    settings: ParticleSettings | None = None,
    *,
    title: str = "Posterior Evolution",
) -> Path:
    return plot_state_trajectory(means, path, settings, title=title)
