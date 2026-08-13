"""SVG visualizations for ensemble regime intelligence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.regimes.ensemble.config import EnsembleSettings


def _ensure(path: Path, settings: EnsembleSettings) -> bool:
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
    settings: EnsembleSettings,
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


def plot_regime_timeline(
    states: Any,
    path: Path,
    settings: EnsembleSettings | None = None,
    *,
    title: str = "Unified Regime Timeline",
) -> Path:
    settings = settings or EnsembleSettings.default()
    return _line_plot(
        [(np.asarray(states, dtype=np.float64).reshape(-1), "regime")],
        Path(path),
        title=title,
        settings=settings,
    )


def plot_member_timelines(
    member_states: dict[str, Any],
    path: Path,
    settings: EnsembleSettings | None = None,
    *,
    title: str = "Member Timelines",
) -> Path:
    settings = settings or EnsembleSettings.default()
    series = [(np.asarray(v, dtype=np.float64).reshape(-1), k) for k, v in member_states.items()]
    return _line_plot(series, Path(path), title=title, settings=settings)


def plot_agreement_heatmap(
    matrix: Any,
    names: list[str],
    path: Path,
    settings: EnsembleSettings | None = None,
    *,
    title: str = "Agreement Heatmap",
) -> Path:
    settings = settings or EnsembleSettings.default()
    path = Path(path)
    if not _ensure(path, settings):
        return path
    mat = np.asarray(matrix, dtype=np.float64)
    n = mat.shape[0]
    cell = 36
    width = 80 + cell * n
    height = 60 + cell * n
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i in range(n):
        for j in range(n):
            val = float(mat[i, j])
            shade = int(255 * (1.0 - val))
            color = f"rgb({shade},{shade},255)"
            x = 60 + j * cell
            y = 40 + i * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x + 8}" y="{y + 20}" font-size="10">{val:.2f}</text>'
            )
        parts.append(
            f'<text x="5" y="{40 + i * cell + 20}" font-size="10">{names[i] if i < len(names) else i}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_confidence_timeline(
    confidence: Any,
    path: Path,
    settings: EnsembleSettings | None = None,
    *,
    title: str = "Confidence Timeline",
) -> Path:
    settings = settings or EnsembleSettings.default()
    return _line_plot(
        [(np.asarray(confidence, dtype=np.float64).reshape(-1), "confidence")],
        Path(path),
        title=title,
        settings=settings,
    )


def plot_weight_evolution(
    history: list[dict[str, float]],
    path: Path,
    settings: EnsembleSettings | None = None,
    *,
    title: str = "Weight Evolution",
) -> Path:
    settings = settings or EnsembleSettings.default()
    if not history:
        return _line_plot([], Path(path), title=title, settings=settings)
    names = sorted({k for h in history for k in h})
    series = []
    for name in names:
        series.append(
            (np.asarray([float(h.get(name, 0.0)) for h in history], dtype=np.float64), name)
        )
    return _line_plot(series, Path(path), title=title, settings=settings)


def plot_probability_dashboard(
    proba: Any,
    state_names: tuple[str, ...] | list[str],
    path: Path,
    settings: EnsembleSettings | None = None,
    *,
    title: str = "Regime Probability Dashboard",
) -> Path:
    settings = settings or EnsembleSettings.default()
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    series = [
        (p[:, i], state_names[i] if i < len(state_names) else f"r{i}")
        for i in range(p.shape[1])
    ]
    return _line_plot(series, Path(path), title=title, settings=settings)


def plot_transition_chart(
    transition: Any,
    path: Path,
    settings: EnsembleSettings | None = None,
    *,
    title: str = "Transition Probabilities",
) -> Path:
    settings = settings or EnsembleSettings.default()
    tm = np.asarray(transition, dtype=np.float64)
    names = [f"s{i}" for i in range(tm.shape[0])]
    return plot_agreement_heatmap(tm, names, path, settings, title=title)
