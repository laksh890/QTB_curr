"""Forecast uncertainty SVG chart."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqrp.app.state_space.config import StateSpaceSettings


def plot_forecast_uncertainty(
    step_distributions: np.ndarray,
    path: Path,
    settings: StateSpaceSettings | None = None,
    *,
    title: str = "Forecast Uncertainty",
) -> Path:
    settings = settings or StateSpaceSettings.default()
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
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="18" font-size="14">{title}</text>',
        f'<line x1="40" y1="200" x2="{width - 20}" y2="200" stroke="#333"/>',
        '<line x1="40" y1="40" x2="40" y2="200" stroke="#333"/>',
    ]
    colors = ["#e45756", "#54a24b", "#4c78a8", "#f58518", "#b279a2", "#bab0ac"]
    xs = np.linspace(50, width - 30, h)
    for j in range(k):
        pts = " ".join(f"{xs[i]:.1f},{200 - 150 * steps[i, j]:.1f}" for i in range(h))
        color = colors[j % len(colors)]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
    # max-prob band
    max_p = steps.max(axis=1)
    band = " ".join(f"{xs[i]:.1f},{200 - 150 * max_p[i]:.1f}" for i in range(h))
    parts.append(f'<polyline fill="none" stroke="#111" stroke-dasharray="4 3" points="{band}"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
