"""State probability chart visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqrp.app.regimes.base.regime import RegimeResult
from iqrp.app.regimes.config import RegimeSettings


def plot_probabilities(
    result: RegimeResult,
    path: Path,
    settings: RegimeSettings | None = None,
) -> Path:
    settings = settings or RegimeSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    proba = np.asarray(result.state_probabilities, dtype=np.float64)
    if proba.ndim != 2:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40">'
            "<text x='10' y='20'>No probabilities</text></svg>",
            encoding="utf-8",
        )
        return path
    n = min(proba.shape[0], settings.visualization.max_points)
    k = proba.shape[1]
    width, height = 720, 280
    colors = ["#e45756", "#bab0ac", "#54a24b", "#4c78a8", "#f58518"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="10" y="18" font-size="14">State Probabilities</text>',
    ]
    xs = np.linspace(40, width - 10, n)
    for j in range(k):
        series = proba[:n, j]
        ys = height - 30 - np.nan_to_num(series, nan=0.0) * (height - 60)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=False))
        parts.append(
            f'<polyline fill="none" stroke="{colors[j % len(colors)]}" '
            f'stroke-width="1.4" points="{pts}"><title>state_{j}</title></polyline>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
