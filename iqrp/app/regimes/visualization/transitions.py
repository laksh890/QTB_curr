"""Transition graph / matrix visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqrp.app.regimes.base.regime import RegimeResult
from iqrp.app.regimes.config import RegimeSettings


def plot_transitions(
    result: RegimeResult,
    path: Path,
    settings: RegimeSettings | None = None,
) -> Path:
    _ = settings
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tm = np.asarray(result.transition_matrix, dtype=np.float64)
    k = tm.shape[0]
    cell = 36
    margin = 60
    width = margin + k * cell + 20
    height = margin + k * cell + 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="10" y="18" font-size="14">Transition Matrix</text>',
    ]
    for i in range(k):
        parts.append(
            f'<text x="{margin - 8}" y="{margin + i * cell + 22}" '
            f'font-size="10" text-anchor="end">S{i}</text>'
        )
        parts.append(
            f'<text x="{margin + i * cell + 10}" y="{margin - 8}" font-size="10">S{i}</text>'
        )
        for j in range(k):
            val = float(tm[i, j])
            intensity = int(255 * (1 - val))
            color = f"rgb({intensity},{intensity},255)"
            parts.append(
                f'<rect x="{margin + j * cell}" y="{margin + i * cell}" '
                f'width="{cell - 2}" height="{cell - 2}" fill="{color}">'
                f"<title>P({i}->{j})={val:.3f}</title></rect>"
            )
            parts.append(
                f'<text x="{margin + j * cell + 8}" y="{margin + i * cell + 22}" '
                f'font-size="9">{val:.2f}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
