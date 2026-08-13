"""Persistence chart visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqrp.app.regimes.base.regime import RegimeResult
from iqrp.app.regimes.config import RegimeSettings


def plot_persistence(
    result: RegimeResult,
    path: Path,
    settings: RegimeSettings | None = None,
) -> Path:
    settings = settings or RegimeSettings.default()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if result.persistence is None:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40">'
            "<text x='10' y='20'>No persistence</text></svg>",
            encoding="utf-8",
        )
        return path
    roll = np.asarray(result.persistence.rolling_persistence, dtype=np.float64)
    n = min(len(roll), settings.visualization.max_points)
    series = roll[:n]
    width, height = 720, 240
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="10" y="18" font-size="14">Rolling Persistence</text>',
    ]
    if n >= 2 and np.isfinite(series).any():
        xs = np.linspace(40, width - 10, n)
        ymin, ymax = 0.0, 1.0
        ys = (
            height
            - 30
            - (np.nan_to_num(series, nan=0.0) - ymin) / (ymax - ymin + 1e-12) * (height - 60)
        )
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=False))
        parts.append(f'<polyline fill="none" stroke="#4c78a8" stroke-width="1.5" points="{pts}"/>')
    # Expected duration bars
    exp = result.persistence.expected_duration
    if exp:
        max_e = max(exp.values()) or 1.0
        for i, (sid, dur) in enumerate(sorted(exp.items())):
            y = 40 + i * 18
            w = 200.0 * float(dur) / max_e
            parts.append(f'<text x="10" y="{y + 12}" font-size="10">E[dur|S{sid}]</text>')
            parts.append(
                f'<rect x="90" y="{y}" width="{w:.1f}" height="12" fill="#54a24b">'
                f"<title>{dur:.2f}</title></rect>"
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
