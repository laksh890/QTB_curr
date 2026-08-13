"""SVG visualizations for statistical forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.forecasting.statistical.config import StatisticalSettings


def _ensure(path: Path, settings: StatisticalSettings) -> bool:
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
    settings: StatisticalSettings | None = None,
    title: str = "Statistical Forecast",
) -> Path:
    settings = settings or StatisticalSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    series = [
        (np.asarray(actual, dtype=np.float64).reshape(-1), "actual"),
        (np.asarray(predicted, dtype=np.float64).reshape(-1), "forecast"),
    ]
    if lower is not None and upper is not None:
        series.append((np.asarray(lower, dtype=np.float64).reshape(-1), "lower"))
        series.append((np.asarray(upper, dtype=np.float64).reshape(-1), "upper"))
    return _line_plot(series, path, title=title, settings=settings)


def plot_residuals(resid: Any, path: Path, settings: StatisticalSettings | None = None) -> Path:
    settings = settings or StatisticalSettings.default()
    return _line_plot(
        [(np.asarray(resid, dtype=np.float64).reshape(-1), "residual")],
        Path(path),
        title="Residual Analysis",
        settings=settings,
    )


def plot_acf(acf_vals: list[float] | np.ndarray, path: Path, settings: StatisticalSettings | None = None) -> Path:
    settings = settings or StatisticalSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    vals = np.asarray(acf_vals, dtype=np.float64).reshape(-1)
    width, height = 640, 240
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="10" y="18" font-size="14">Residual ACF</text>',
        f'<line x1="40" y1="{height/2}" x2="{width-20}" y2="{height/2}" stroke="#999"/>',
    ]
    n = max(vals.size, 1)
    for i, v in enumerate(vals):
        x = 40 + i * (width - 60) / n
        y2 = height / 2 - v * (height / 2 - 30)
        parts.append(f'<line x1="{x:.1f}" y1="{height/2}" x2="{x:.1f}" y2="{y2:.1f}" stroke="#1d3557" stroke-width="2"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_qq(resid: Any, path: Path, settings: StatisticalSettings | None = None) -> Path:
    settings = settings or StatisticalSettings.default()
    e = np.sort(np.asarray(resid, dtype=np.float64).reshape(-1))
    if e.size == 0:
        path = Path(path)
        _ensure(path, settings)
        return path
    # theoretical normal quantiles
    probs = (np.arange(1, e.size + 1) - 0.5) / e.size
    from scipy.stats import norm

    theo = norm.ppf(probs)
    return _line_plot(
        [(theo, "theoretical"), (e, "sample")],
        Path(path),
        title="QQ Plot",
        settings=settings or StatisticalSettings.default(),
    )


def plot_irf(irf: np.ndarray, path: Path, settings: StatisticalSettings | None = None) -> Path:
    settings = settings or StatisticalSettings.default()
    arr = np.asarray(irf, dtype=np.float64)
    # plot response of var 0 to shock 0
    series = arr[:, 0, 0] if arr.ndim == 3 else arr.reshape(-1)
    return _line_plot(
        [(series, "irf_0_0")],
        Path(path),
        title="Impulse Response",
        settings=settings,
    )


def plot_seasonal_decomposition(
    y: Any,
    period: int,
    path: Path,
    settings: StatisticalSettings | None = None,
) -> Path:
    settings = settings or StatisticalSettings.default()
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    s = max(int(period), 2)
    if x.size < 2 * s:
        trend = np.full_like(x, np.mean(x) if x.size else 0.0)
        seasonal = np.zeros_like(x)
    else:
        kernel = np.ones(s) / s
        trend = np.convolve(x, kernel, mode="same")
        detr = x - trend
        seasonal = np.zeros_like(x)
        for i in range(s):
            seasonal[i::s] = np.mean(detr[i::s]) if detr[i::s].size else 0.0
    return _line_plot(
        [(x, "observed"), (trend, "trend"), (seasonal, "seasonal")],
        Path(path),
        title="Seasonal Decomposition",
        settings=settings,
    )


def plot_rolling_comparison(
    actual: Any,
    forecasts: dict[str, Any],
    path: Path,
    settings: StatisticalSettings | None = None,
) -> Path:
    settings = settings or StatisticalSettings.default()
    series = [(np.asarray(actual, dtype=np.float64).reshape(-1), "actual")]
    for name, fc in forecasts.items():
        series.append((np.asarray(fc, dtype=np.float64).reshape(-1), name))
    return _line_plot(series, Path(path), title="Rolling Forecast Comparison", settings=settings)


def _line_plot(
    series: list[tuple[np.ndarray, str]],
    path: Path,
    *,
    title: str,
    settings: StatisticalSettings,
) -> Path:
    path = Path(path)
    if not _ensure(path, settings):
        return path
    width, height = 720, 260
    colors = ["#1d3557", "#e63946", "#2a9d8f", "#457b9d", "#f4a261", "#9b5de5"]
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
    n = max(len(cleaned[0][0]), 1)
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
