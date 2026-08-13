"""SVG visualizations for Kalman filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.regimes.kalman.config import KalmanSettings


def _ensure(path: Path, settings: KalmanSettings) -> bool:
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
    settings: KalmanSettings,
    bands: tuple[np.ndarray, np.ndarray] | None = None,
) -> Path:
    path = Path(path)
    if not _ensure(path, settings):
        return path
    width, height = 720, 260
    colors = ["#1d3557", "#e63946", "#2a9d8f", "#457b9d", "#f4a261"]
    max_n = settings.visualization.max_points
    cleaned: list[tuple[np.ndarray, str]] = []
    for arr, name in series:
        v = np.asarray(arr, dtype=np.float64).reshape(-1)[:max_n]
        cleaned.append((v, name))
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
            f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="260">'
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
        parts.append(
            f'<polygon fill="#a8dadc" opacity="0.45" points="{" ".join(poly)}"/>'
        )
    for idx, (v, name) in enumerate(cleaned):
        pts = []
        for i, val in enumerate(v):
            x, y = _xy(i, float(val))
            pts.append(f"{x:.1f},{y:.1f}")
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


def plot_filtered_state(
    means: Any,
    path: Path,
    settings: KalmanSettings | None = None,
    *,
    observations: Any | None = None,
    title: str = "Filtered State",
) -> Path:
    settings = settings or KalmanSettings.default()
    m = np.asarray(means, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    series: list[tuple[np.ndarray, str]] = [(m[:, 0], "state_0")]
    if observations is not None:
        series.append((np.asarray(observations, dtype=np.float64).reshape(-1), "obs"))
    return _line_plot(series, Path(path), title=title, settings=settings)


def plot_smoothed_state(
    means: Any,
    path: Path,
    settings: KalmanSettings | None = None,
    *,
    title: str = "Smoothed State",
) -> Path:
    settings = settings or KalmanSettings.default()
    m = np.asarray(means, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    return _line_plot([(m[:, 0], "smoothed")], Path(path), title=title, settings=settings)


def plot_prediction_bands(
    means: Any,
    lower: Any,
    upper: Any,
    path: Path,
    settings: KalmanSettings | None = None,
    *,
    title: str = "Prediction Bands",
) -> Path:
    settings = settings or KalmanSettings.default()
    m = np.asarray(means, dtype=np.float64).reshape(-1)
    return _line_plot(
        [(m, "mean")],
        Path(path),
        title=title,
        settings=settings,
        bands=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
    )


def plot_innovations(
    innovations: Any,
    path: Path,
    settings: KalmanSettings | None = None,
    *,
    title: str = "Innovations",
) -> Path:
    settings = settings or KalmanSettings.default()
    v = np.asarray(innovations, dtype=np.float64)
    if v.ndim > 1:
        v = v[:, 0]
    return _line_plot([(v, "innovation")], Path(path), title=title, settings=settings)


def plot_covariance_evolution(
    covs: Any,
    path: Path,
    settings: KalmanSettings | None = None,
    *,
    title: str = "Covariance Trace",
) -> Path:
    settings = settings or KalmanSettings.default()
    c = np.asarray(covs, dtype=np.float64)
    traces = np.array([float(np.trace(c[t])) for t in range(c.shape[0])], dtype=np.float64)
    return _line_plot([(traces, "tr(P)")], Path(path), title=title, settings=settings)


def plot_kalman_gain(
    gains: Any,
    path: Path,
    settings: KalmanSettings | None = None,
    *,
    title: str = "Kalman Gain Norm",
) -> Path:
    settings = settings or KalmanSettings.default()
    g = np.asarray(gains, dtype=np.float64)
    norms = np.array([float(np.linalg.norm(g[t])) for t in range(g.shape[0])], dtype=np.float64)
    return _line_plot([(norms, "||K||")], Path(path), title=title, settings=settings)
