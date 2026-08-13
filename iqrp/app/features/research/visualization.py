"""SVG visualizations for feature research reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import polars as pl

from iqrp.app.features.research.config import ResearchSettings


class ResearchVisualizer:
    def __init__(self, settings: ResearchSettings | None = None) -> None:
        self.settings = settings or ResearchSettings.default()

    def write_all(
        self,
        output_dir: Path,
        *,
        corr_pearson: pl.DataFrame | None,
        rolling_ic: dict[str, list[float]],
        distributions: dict[str, np.ndarray],
        drift_psi: dict[str, float],
        mi_ranking: list[tuple[str, float]],
        importance: dict[str, float],
        stability: dict[str, float],
    ) -> dict[str, Path]:
        if not self.settings.visualization.enabled:
            return {}
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        max_n = self.settings.visualization.max_features_in_charts
        if corr_pearson is not None and not corr_pearson.is_empty():
            paths["correlation_heatmap"] = self.correlation_heatmap(
                corr_pearson, output_dir / "correlation_heatmap.svg", max_n=max_n
            )
        if rolling_ic:
            paths["rolling_ic"] = self.rolling_ic_chart(
                rolling_ic, output_dir / "rolling_ic.svg", max_n=max_n
            )
        if distributions:
            paths["feature_distribution"] = self.distribution_chart(
                distributions, output_dir / "feature_distribution.svg", max_n=min(6, max_n)
            )
            paths["qq_plot"] = self.qq_plot(
                next(iter(distributions.values())), output_dir / "qq_plot.svg"
            )
        if drift_psi:
            paths["feature_drift"] = self.bar_chart(
                sorted(drift_psi.items(), key=lambda kv: -abs(kv[1]))[:max_n],
                output_dir / "feature_drift.svg",
                title="Population Drift (PSI)",
            )
        if mi_ranking:
            paths["mutual_information"] = self.bar_chart(
                mi_ranking[:max_n],
                output_dir / "mutual_information.svg",
                title="Mutual Information Ranking",
            )
        if importance:
            paths["importance"] = self.bar_chart(
                sorted(importance.items(), key=lambda kv: -abs(kv[1]))[:max_n],
                output_dir / "importance.svg",
                title="Permutation Importance",
            )
        if stability:
            paths["stability"] = self.bar_chart(
                sorted(stability.items(), key=lambda kv: -kv[1])[:max_n],
                output_dir / "stability.svg",
                title="Stability Scores",
            )
        return paths

    def correlation_heatmap(self, matrix: pl.DataFrame, path: Path, *, max_n: int) -> Path:
        names = matrix["feature"].to_list()[:max_n]
        n = len(names)
        cell = 18
        margin = 80
        width = margin + n * cell + 20
        height = margin + n * cell + 20
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<text x="10" y="20" font-size="14">Pearson Correlation</text>',
        ]
        for i, a in enumerate(names):
            parts.append(
                f'<text x="{margin - 5}" y="{margin + i * cell + 12}" '
                f'font-size="8" text-anchor="end">{escape(str(a)[:16])}</text>'
            )
            parts.append(
                f'<text x="{margin + i * cell + 2}" y="{margin - 5}" '
                f'font-size="8" transform="rotate(-60 {margin + i * cell},{margin})">'
                f"{escape(str(a)[:16])}</text>"
            )
            for j, b in enumerate(names):
                val = float(matrix.filter(pl.col("feature") == a).select(b).item())
                color = _corr_color(val)
                parts.append(
                    f'<rect x="{margin + j * cell}" y="{margin + i * cell}" '
                    f'width="{cell - 1}" height="{cell - 1}" fill="{color}">'
                    f"<title>{escape(a)}/{escape(b)}: {val:.3f}</title></rect>"
                )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

    def rolling_ic_chart(
        self, rolling_ic: dict[str, list[float]], path: Path, *, max_n: int
    ) -> Path:
        items = list(rolling_ic.items())[:max_n]
        width, height = 720, 280
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<text x="10" y="20" font-size="14">Rolling Information Coefficient</text>',
            f'<line x1="40" y1="{height/2}" x2="{width-10}" y2="{height/2}" '
            f'stroke="#999" stroke-dasharray="4"/>',
        ]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for idx, (name, series) in enumerate(items):
            arr = np.asarray(series, dtype=np.float64)
            if len(arr) < 2:
                continue
            xs = np.linspace(40, width - 10, len(arr))
            ys = height / 2 - np.nan_to_num(arr, nan=0.0) * (height / 2 - 30)
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=False))
            color = colors[idx % len(colors)]
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}">'
                f"<title>{escape(name)}</title></polyline>"
            )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

    def distribution_chart(
        self, distributions: dict[str, np.ndarray], path: Path, *, max_n: int
    ) -> Path:
        items = list(distributions.items())[:max_n]
        width, height = 720, 240
        panel_w = width // max(len(items), 1)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<text x="10" y="18" font-size="14">Feature Distributions</text>',
        ]
        for i, (name, arr) in enumerate(items):
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            hist, edges = np.histogram(finite, bins=12)
            max_h = max(int(hist.max()), 1)
            base_x = i * panel_w + 20
            parts.append(f'<text x="{base_x}" y="36" font-size="10">{escape(name[:18])}</text>')
            bar_w = max(4, (panel_w - 40) // len(hist))
            for j, h in enumerate(hist):
                bh = 140.0 * float(h) / max_h
                parts.append(
                    f'<rect x="{base_x + j * bar_w}" y="{200 - bh}" '
                    f'width="{bar_w - 1}" height="{bh}" fill="#4c78a8"/>'
                )
            _ = edges
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

    def qq_plot(self, values: np.ndarray, path: Path) -> Path:
        finite = np.sort(values[np.isfinite(values)])
        width, height = 360, 360
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<text x="10" y="20" font-size="14">QQ Plot vs Normal</text>',
        ]
        if len(finite) >= 5:
            from scipy import stats  # type: ignore[import-untyped]

            theor = stats.norm.ppf((np.arange(1, len(finite) + 1) - 0.5) / len(finite))
            x = (theor - theor.min()) / (theor.max() - theor.min() + 1e-12)
            y = (finite - finite.min()) / (finite.max() - finite.min() + 1e-12)
            pts = " ".join(
                f"{40 + 280 * xi:.1f},{320 - 280 * yi:.1f}" for xi, yi in zip(x, y, strict=False)
            )
            parts.append(
                f'<polyline fill="none" stroke="#e45756" stroke-width="1.2" points="{pts}"/>'
            )
            parts.append(
                '<line x1="40" y1="320" x2="320" y2="40" stroke="#999" stroke-dasharray="3"/>'
            )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

    def bar_chart(self, items: list[tuple[str, float]], path: Path, *, title: str) -> Path:
        n = max(len(items), 1)
        width = 720
        height = 40 + n * 22
        max_abs = max((abs(v) for _, v in items if np.isfinite(v)), default=1.0) or 1.0
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<text x="10" y="18" font-size="14">{escape(title)}</text>',
        ]
        for i, (name, val) in enumerate(items):
            y = 30 + i * 22
            w = 400.0 * abs(val) / max_abs if np.isfinite(val) else 0.0
            color = "#4c78a8" if val >= 0 else "#e45756"
            parts.append(
                f'<text x="10" y="{y + 12}" font-size="10">{escape(str(name)[:24])}</text>'
            )
            parts.append(
                f'<rect x="180" y="{y}" width="{w:.1f}" height="14" fill="{color}">'
                f"<title>{escape(str(name))}: {val}</title></rect>"
            )
            parts.append(f'<text x="{190 + w:.1f}" y="{y + 12}" font-size="9">' f"{val:.4f}</text>")
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path


def _corr_color(r: float) -> str:
    if not np.isfinite(r):
        return "#eeeeee"
    # blue (-) white (0) red (+)
    r = float(np.clip(r, -1, 1))
    if r >= 0:
        g = int(255 * (1 - r))
        return f"rgb(255,{g},{g})"
    g = int(255 * (1 + r))
    return f"rgb({g},{g},255)"


def chart_manifest(paths: dict[str, Path]) -> dict[str, Any]:
    return {k: str(v) for k, v in paths.items()}
