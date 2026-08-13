"""Transition graph SVG."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqrp.app.state_space.config import StateSpaceSettings


def plot_transition_graph(
    transition_matrix: np.ndarray,
    path: Path,
    settings: StateSpaceSettings | None = None,
    *,
    title: str = "Transition Graph",
    state_names: tuple[str, ...] = (),
) -> Path:
    settings = settings or StateSpaceSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    tm = np.asarray(transition_matrix, dtype=np.float64)
    k = tm.shape[0]
    width, height = 480, 360
    cx, cy, radius = width / 2, height / 2 + 10, 120
    positions = [
        (
            cx + radius * np.cos(2 * np.pi * i / k - np.pi / 2),
            cy + radius * np.sin(2 * np.pi * i / k - np.pi / 2),
        )
        for i in range(k)
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
    ]
    for i in range(k):
        for j in range(k):
            p = float(tm[i, j])
            if p < 0.05:
                continue
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            stroke = max(0.5, 4.0 * p)
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


def plot_persistence_distribution(
    run_lengths: list[int],
    path: Path,
    settings: StateSpaceSettings | None = None,
    *,
    title: str = "Persistence Distribution",
) -> Path:
    settings = settings or StateSpaceSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.visualization.enabled or not run_lengths:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    lengths = np.asarray(run_lengths, dtype=np.int64)
    max_l = int(lengths.max())
    counts = np.bincount(lengths, minlength=max_l + 1).astype(np.float64)
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
