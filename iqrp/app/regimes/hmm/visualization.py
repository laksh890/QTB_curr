"""SVG visualization for HMM analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.regimes.hmm.config import HMMSettings


def plot_hidden_state_timeline(
    states: Any,
    path: Path,
    settings: HMMSettings | None = None,
    *,
    title: str = "Hidden State Timeline",
) -> Path:
    settings = settings or HMMSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    ids = np.asarray(states, dtype=np.int64).reshape(-1)
    n = min(len(ids), settings.visualization.max_points)
    ids = ids[:n]
    width, height = 720, 140
    colors = ["#e45756", "#bab0ac", "#54a24b", "#4c78a8", "#f58518", "#b279a2"]
    bar_w = (width - 40) / max(n, 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i, sid in enumerate(ids):
        color = colors[int(sid) % len(colors)]
        parts.append(
            f'<rect x="{40 + i * bar_w:.2f}" y="40" width="{max(bar_w, 0.5):.2f}" '
            f'height="60" fill="{color}"><title>state={int(sid)}</title></rect>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_posterior_heatmap(
    probabilities: Any,
    path: Path,
    settings: HMMSettings | None = None,
    *,
    title: str = "Posterior Probability Heatmap",
) -> Path:
    settings = settings or HMMSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2:
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
                f'width="{max(cell_w, 0.5):.2f}" height="{max(cell_h - 1, 1):.2f}" '
                f'fill="rgb({g},{g},255)"><title>t={i} s={j} p={v:.3f}</title></rect>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_transition_heatmap(
    transition: Any,
    path: Path,
    settings: HMMSettings | None = None,
    *,
    title: str = "Transition Matrix Heatmap",
    state_names: tuple[str, ...] = (),
) -> Path:
    settings = settings or HMMSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    p = np.asarray(transition, dtype=np.float64)
    k = p.shape[0]
    cell = 36
    width = 80 + cell * k
    height = 60 + cell * k
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i in range(k):
        name = state_names[i] if i < len(state_names) else f"s{i}"
        parts.append(f'<text x="8" y="{55 + i * cell + cell / 2:.1f}" font-size="10">{name}</text>')
        for j in range(k):
            v = float(np.clip(p[i, j], 0.0, 1.0))
            g = int(255 * (1.0 - v))
            parts.append(
                f'<rect x="{60 + j * cell}" y="{40 + i * cell}" width="{cell - 2}" '
                f'height="{cell - 2}" fill="rgb({g},{g},255)"><title>{v:.3f}</title></rect>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_likelihood_curve(
    history: list[float],
    path: Path,
    settings: HMMSettings | None = None,
    *,
    title: str = "Log-Likelihood Curve",
) -> Path:
    settings = settings or HMMSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled or not history:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    h = np.asarray(history, dtype=np.float64)
    width, height = 640, 220
    xs = np.linspace(50, width - 30, len(h))
    lo, hi = float(h.min()), float(h.max())
    span = max(hi - lo, 1e-9)
    ys = 180 - 140 * (h - lo) / span
    pts = " ".join(f"{xs[i]:.1f},{ys[i]:.1f}" for i in range(len(h)))
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
        f'<text x="10" y="18" font-size="14">{title}</text>\n'
        f'<polyline fill="none" stroke="#4c78a8" stroke-width="2" points="{pts}"/>\n'
        f"</svg>"
    )
    path.write_text(svg, encoding="utf-8")
    return path


def plot_viterbi_path(
    states: Any,
    path: Path,
    settings: HMMSettings | None = None,
    *,
    title: str = "Viterbi Path",
) -> Path:
    return plot_hidden_state_timeline(states, path, settings, title=title)


def plot_state_duration_histogram(
    run_lengths: list[int] | dict[str, list[int]],
    path: Path,
    settings: HMMSettings | None = None,
    *,
    title: str = "State Duration Histogram",
) -> Path:
    settings = settings or HMMSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(run_lengths, dict):
        lengths = [x for xs in run_lengths.values() for x in xs]
    else:
        lengths = list(run_lengths)
    if not settings.visualization.enabled or not lengths:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    arr = np.asarray(lengths, dtype=np.int64)
    max_l = int(arr.max())
    counts = np.bincount(arr, minlength=max_l + 1).astype(np.float64)
    width, height = 640, 220
    bar_w = (width - 60) / max(max_l, 1)
    peak = max(float(counts.max()), 1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for length in range(1, max_l + 1):
        h = 140 * (counts[length] / peak)
        parts.append(
            f'<rect x="{40 + (length - 1) * bar_w:.2f}" y="{180 - h:.2f}" '
            f'width="{max(bar_w - 1, 0.5):.2f}" height="{h:.2f}" fill="#54a24b"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_emission_means(
    means: Any,
    path: Path,
    settings: HMMSettings | None = None,
    *,
    title: str = "Emission Means",
) -> Path:
    settings = settings or HMMSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    m = np.asarray(means, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    k, _d = m.shape
    width, height = 480, 220
    bar_w = (width - 60) / max(k, 1)
    vals = m[:, 0]
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-9)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i in range(k):
        h = 140 * (float(vals[i]) - lo) / span + 10
        parts.append(
            f'<rect x="{40 + i * bar_w:.2f}" y="{180 - h:.2f}" '
            f'width="{max(bar_w - 4, 0.5):.2f}" height="{h:.2f}" fill="#f58518"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
