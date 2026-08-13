"""SVG visualization for Markov chain analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.regimes.markov.config import MarkovSettings


def plot_transition_heatmap(
    transition: Any,
    path: Path,
    settings: MarkovSettings | None = None,
    *,
    title: str = "Transition Heatmap",
    state_names: tuple[str, ...] = (),
) -> Path:
    settings = settings or MarkovSettings.default()
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
                f'height="{cell - 2}" fill="rgb({g},{g},255)">'
                f"<title>{v:.3f}</title></rect>"
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_transition_graph(
    transition: Any,
    path: Path,
    settings: MarkovSettings | None = None,
    *,
    title: str = "Directed Transition Graph",
    state_names: tuple[str, ...] = (),
) -> Path:
    settings = settings or MarkovSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    tm = np.asarray(transition, dtype=np.float64)
    k = tm.shape[0]
    width, height = 480, 360
    cx, cy, radius = width / 2, height / 2 + 10, 120
    positions = [
        (
            cx + radius * np.cos(2 * np.pi * i / max(k, 1) - np.pi / 2),
            cy + radius * np.sin(2 * np.pi * i / max(k, 1) - np.pi / 2),
        )
        for i in range(k)
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i in range(k):
        for j in range(k):
            prob = float(tm[i, j])
            if prob < 0.05:
                continue
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            stroke = max(0.5, 4.0 * prob)
            if i == j:
                parts.append(
                    f'<circle cx="{x1:.1f}" cy="{y1 - 28:.1f}" r="14" fill="none" '
                    f'stroke="#4c78a8" stroke-width="{stroke:.2f}" opacity="0.7"/>'
                )
            else:
                parts.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="#4c78a8" stroke-width="{stroke:.2f}" opacity="0.55"/>'
                )
    for i, (x, y) in enumerate(positions):
        name = state_names[i] if i < len(state_names) else f"s{i}"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="18" fill="#54a24b"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="11" fill="#fff">'
            f"{name}</text>"
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_occupancy_timeline(
    states: Any,
    path: Path,
    settings: MarkovSettings | None = None,
    *,
    title: str = "State Occupancy Timeline",
) -> Path:
    settings = settings or MarkovSettings.default()
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


def plot_persistence_histogram(
    run_lengths: list[int] | dict[str, list[int]],
    path: Path,
    settings: MarkovSettings | None = None,
    *,
    title: str = "Persistence Histogram",
) -> Path:
    settings = settings or MarkovSettings.default()
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
            f'width="{max(bar_w - 1, 0.5):.2f}" height="{h:.2f}" fill="#4c78a8"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_forecast_probabilities(
    step_distributions: Any,
    path: Path,
    settings: MarkovSettings | None = None,
    *,
    title: str = "Forecast Probabilities",
) -> Path:
    settings = settings or MarkovSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    steps = np.asarray(step_distributions, dtype=np.float64)
    if steps.ndim == 1:
        steps = steps.reshape(1, -1)
    h, k = steps.shape
    width, height = 640, 240
    colors = ["#e45756", "#54a24b", "#4c78a8", "#f58518", "#b279a2", "#bab0ac"]
    xs = np.linspace(50, width - 30, h)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
        f'<line x1="40" y1="200" x2="{width - 20}" y2="200" stroke="#333"/>',
        '<line x1="40" y1="40" x2="40" y2="200" stroke="#333"/>',
    ]
    for j in range(k):
        pts = " ".join(f"{xs[i]:.1f},{200 - 150 * steps[i, j]:.1f}" for i in range(h))
        parts.append(
            f'<polyline fill="none" stroke="{colors[j % len(colors)]}" '
            f'stroke-width="2" points="{pts}"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_stationary_distribution(
    stationary: Any,
    path: Path,
    settings: MarkovSettings | None = None,
    *,
    title: str = "Stationary Distribution",
    state_names: tuple[str, ...] = (),
) -> Path:
    settings = settings or MarkovSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    pi = np.asarray(stationary, dtype=np.float64).reshape(-1)
    k = pi.size
    width, height = 480, 220
    bar_w = (width - 60) / max(k, 1)
    peak = max(float(pi.max()), 1e-12)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i in range(k):
        h = 140 * (float(pi[i]) / peak)
        name = state_names[i] if i < len(state_names) else f"s{i}"
        parts.append(
            f'<rect x="{40 + i * bar_w:.2f}" y="{180 - h:.2f}" '
            f'width="{max(bar_w - 4, 0.5):.2f}" height="{h:.2f}" fill="#54a24b"/>'
        )
        parts.append(
            f'<text x="{40 + i * bar_w + bar_w / 2:.1f}" y="200" text-anchor="middle" '
            f'font-size="10">{name}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
