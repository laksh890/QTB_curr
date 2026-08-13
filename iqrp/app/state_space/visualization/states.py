"""State timeline SVG chart."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqrp.app.state_space.config import StateSpaceSettings


def plot_state_timeline(
    states: np.ndarray,
    path: Path,
    settings: StateSpaceSettings | None = None,
    *,
    title: str = "Latent State Timeline",
) -> Path:
    settings = settings or StateSpaceSettings.default()
    if not settings.visualization.enabled:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        return path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
