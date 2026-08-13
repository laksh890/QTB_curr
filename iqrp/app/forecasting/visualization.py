"""SVG visualizations for forecasting diagnostics and outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.forecasting.config import ForecastingSettings


def _ensure(path: Path, settings: ForecastingSettings) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return False
    return True


def plot_forecast(
    actual: Any,
    predicted: Any,
    path: Path,
    *,
    lower: Any | None = None,
    upper: Any | None = None,
    settings: ForecastingSettings | None = None,
    title: str = "Forecast Chart",
) -> Path:
    settings = settings or ForecastingSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    y = np.asarray(actual, dtype=np.float64).reshape(-1)[: settings.visualization.max_points]
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)[: settings.visualization.max_points]
    series = [(y, "actual"), (p, "predicted")]
    if lower is not None and upper is not None:
        series.append((np.asarray(lower, dtype=np.float64).reshape(-1), "lower"))
        series.append((np.asarray(upper, dtype=np.float64).reshape(-1), "upper"))
    return _line_plot(series, path, title=title, settings=settings)


def plot_residuals(
    residuals: Any,
    path: Path,
    settings: ForecastingSettings | None = None,
    *,
    title: str = "Residual Plot",
) -> Path:
    settings = settings or ForecastingSettings.default()
    return _line_plot(
        [(np.asarray(residuals, dtype=np.float64).reshape(-1), "residual")],
        Path(path),
        title=title,
        settings=settings,
    )


def plot_rolling_accuracy(
    scores: Any,
    path: Path,
    settings: ForecastingSettings | None = None,
    *,
    title: str = "Rolling Accuracy",
) -> Path:
    settings = settings or ForecastingSettings.default()
    return _line_plot(
        [(np.asarray(scores, dtype=np.float64).reshape(-1), "score")],
        Path(path),
        title=title,
        settings=settings,
    )


def plot_feature_importance(
    importances: dict[str, float],
    path: Path,
    settings: ForecastingSettings | None = None,
    *,
    title: str = "Feature Importance",
) -> Path:
    settings = settings or ForecastingSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    items = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
    width, height = 640, max(120, 28 * len(items) + 40)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    max_v = max((abs(v) for _, v in items), default=1.0) or 1.0
    for i, (name, val) in enumerate(items):
        y = 36 + i * 26
        bar_w = (width - 180) * abs(val) / max_v
        parts.append(f'<text x="10" y="{y + 12}" font-size="11">{name}</text>')
        parts.append(
            f'<rect x="140" y="{y}" width="{bar_w:.1f}" height="16" fill="#1d3557"/>'
        )
        parts.append(f'<text x="{145 + bar_w:.1f}" y="{y + 12}" font-size="10">{val:.3f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_horizon_comparison(
    errors_by_horizon: dict[int, float],
    path: Path,
    settings: ForecastingSettings | None = None,
    *,
    title: str = "Forecast Horizon Comparison",
) -> Path:
    settings = settings or ForecastingSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    items = sorted(errors_by_horizon.items())
    width, height = 640, 260
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    if not items:
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path
    max_v = max(v for _, v in items) or 1.0
    n = len(items)
    bar_w = (width - 80) / max(n, 1)
    for i, (h, v) in enumerate(items):
        x = 40 + i * bar_w
        bh = (height - 60) * (v / max_v)
        y = height - 30 - bh
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.7:.1f}" height="{bh:.1f}" fill="#e63946"/>'
        )
        parts.append(f'<text x="{x:.1f}" y="{height - 10}" font-size="10">h{h}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _line_plot(
    series: list[tuple[np.ndarray, str]],
    path: Path,
    *,
    title: str,
    settings: ForecastingSettings,
) -> Path:
    path = Path(path)
    if not _ensure(path, settings):
        return path
    width, height = 720, 260
    colors = ["#1d3557", "#e63946", "#2a9d8f", "#457b9d", "#f4a261"]
    max_n = settings.visualization.max_points
    cleaned = [(np.asarray(a, dtype=np.float64).reshape(-1)[:max_n], name) for a, name in series]
    nonempty = [v for v, _ in cleaned if v.size]
    if not nonempty:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="10" y="18" font-size="14">{title}</text></svg>',
            encoding="utf-8",
        )
        return path
    all_vals = np.concatenate(nonempty)
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    span = max(hi - lo, 1e-9)
    n = max(cleaned[0][0].size, 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for idx, (v, name) in enumerate(cleaned):
        if not v.size:
            continue
        pts = []
        for i, val in enumerate(v):
            x = 40 + (width - 60) * i / max(n - 1, 1)
            y = height - 30 - (height - 50) * (float(val) - lo) / span
            pts.append(f"{x:.1f},{y:.1f}")
        col = colors[idx % len(colors)]
        parts.append(
            f'<polyline fill="none" stroke="{col}" stroke-width="1.5" points="{" ".join(pts)}"/>'
        )
        parts.append(
            f'<text x="{width - 140}" y="{30 + 14 * idx}" font-size="11" fill="{col}">{name}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
