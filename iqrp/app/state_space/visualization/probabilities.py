"""Probability heatmap SVG."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqrp.app.state_space.config import StateSpaceSettings


def plot_probability_heatmap(
    probabilities: np.ndarray,
    path: Path,
    settings: StateSpaceSettings | None = None,
    *,
    title: str = "State Probability Heatmap",
) -> Path:
    settings = settings or StateSpaceSettings.default()
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
            gray = int(255 * (1.0 - v))
            color = f"rgb({gray},{gray},255)"
            parts.append(
                f'<rect x="{40 + i * cell_w:.2f}" y="{30 + j * cell_h:.2f}" '
                f'width="{max(cell_w, 0.5):.2f}" height="{max(cell_h - 1, 1):.2f}" '
                f'fill="{color}"><title>t={i} s={j} p={v:.3f}</title></rect>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
