"""SVG visualizations for label research."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import polars as pl

from iqrp.app.labels.config import LabelSettings


class LabelVisualizer:
    def __init__(self, settings: LabelSettings | None = None) -> None:
        self.settings = settings or LabelSettings.default()

    def write_all(
        self,
        output_dir: Path,
        frame: pl.DataFrame,
        *,
        label_columns: list[str],
    ) -> dict[str, Path]:
        if not self.settings.visualization.enabled:
            return {}
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        paths["class_distribution"] = self.class_distribution(
            frame, label_columns, output_dir / "class_distribution.svg"
        )
        paths["rolling_label_distribution"] = self.rolling_distribution(
            frame, label_columns, output_dir / "rolling_label_distribution.svg"
        )
        paths["label_drift"] = self.label_drift(
            frame, label_columns, output_dir / "label_drift.svg"
        )
        if "tb_upper" in frame.columns and "tb_lower" in frame.columns:
            paths["barrier_chart"] = self.barrier_chart(frame, output_dir / "barrier_chart.svg")
        if "bull_bear_sideways" in frame.columns:
            paths["regime_timeline"] = self.regime_timeline(
                frame, output_dir / "regime_timeline.svg"
            )
        return paths

    def class_distribution(self, frame: pl.DataFrame, columns: list[str], path: Path) -> Path:
        width, height = 720, 40 + 22 * max(len(columns), 1)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<text x="10" y="18" font-size="14">Label Class Distribution (entropy proxy)</text>',
        ]
        for i, col in enumerate(columns[:40]):
            arr = frame[col].cast(pl.Float64).to_numpy()
            finite = arr[np.isfinite(arr)]
            y = 30 + i * 22
            if finite.size == 0:
                continue
            uniq, counts = np.unique(np.round(finite, 6), return_counts=True)
            total = counts.sum()
            x = 180.0
            colors = ["#4c78a8", "#f58518", "#e45756", "#54a24b", "#b279a2"]
            parts.append(f'<text x="10" y="{y + 12}" font-size="10">{escape(col[:22])}</text>')
            for j, (u, c) in enumerate(zip(uniq[:8], counts[:8], strict=False)):
                w = 400.0 * float(c) / total
                parts.append(
                    f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="14" '
                    f'fill="{colors[j % len(colors)]}"><title>{u}: {c}</title></rect>'
                )
                x += w
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

    def rolling_distribution(self, frame: pl.DataFrame, columns: list[str], path: Path) -> Path:
        width, height = 720, 280
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<text x="10" y="18" font-size="14">Rolling Mean of Labels</text>',
        ]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        max_points = self.settings.visualization.max_points
        for idx, col in enumerate(columns[:4]):
            arr = frame[col].cast(pl.Float64).to_numpy()
            if len(arr) < 5:
                continue
            window = max(5, len(arr) // 20)
            roll = np.convolve(np.nan_to_num(arr, nan=0.0), np.ones(window) / window, mode="same")
            step = max(1, len(roll) // max_points)
            xs = np.linspace(40, width - 10, len(roll[::step]))
            ys = (
                height
                - 40
                - (roll[::step] - np.nanmin(roll))
                / (np.nanmax(roll) - np.nanmin(roll) + 1e-12)
                * (height - 80)
            )
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=False))
            parts.append(
                f'<polyline fill="none" stroke="{colors[idx % len(colors)]}" '
                f'stroke-width="1.5" points="{pts}"><title>{escape(col)}</title></polyline>'
            )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

    def label_drift(self, frame: pl.DataFrame, columns: list[str], path: Path) -> Path:
        width, height = 720, 40 + 22 * max(len(columns), 1)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<text x="10" y="18" font-size="14">Label Drift (early vs late mean shift)</text>',
        ]
        for i, col in enumerate(columns[:40]):
            arr = frame[col].cast(pl.Float64).to_numpy()
            finite_idx = np.where(np.isfinite(arr))[0]
            y = 30 + i * 22
            if finite_idx.size < 10:
                continue
            mid = len(arr) // 2
            early = arr[:mid][np.isfinite(arr[:mid])]
            late = arr[mid:][np.isfinite(arr[mid:])]
            if early.size == 0 or late.size == 0:
                continue
            e = float(np.mean(early))
            late_mean = float(np.mean(late))
            delta = late_mean - e
            w = min(400.0, abs(delta) * 2000)
            color = "#e45756" if delta >= 0 else "#4c78a8"
            parts.append(f'<text x="10" y="{y + 12}" font-size="10">{escape(col[:22])}</text>')
            parts.append(
                f'<rect x="180" y="{y}" width="{w:.1f}" height="14" fill="{color}">'
                f"<title>delta={delta:.6f}</title></rect>"
            )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

    def barrier_chart(self, frame: pl.DataFrame, path: Path) -> Path:
        n = min(frame.height, self.settings.visualization.max_points)
        close = frame["close"][:n].to_numpy()
        upper = frame["tb_upper"][:n].to_numpy()
        lower = frame["tb_lower"][:n].to_numpy()
        width, height = 720, 320
        ymin = float(np.nanmin([np.nanmin(close), np.nanmin(lower)]))
        ymax = float(np.nanmax([np.nanmax(close), np.nanmax(upper)]))
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<text x="10" y="18" font-size="14">Triple Barrier Chart</text>',
        ]

        def ymap(v: float) -> float:
            return height - 30 - (v - ymin) / (ymax - ymin + 1e-12) * (height - 60)

        xs = np.linspace(40, width - 10, n)
        for name, series, color in (
            ("close", close, "#333"),
            ("upper", upper, "#e45756"),
            ("lower", lower, "#4c78a8"),
        ):
            pts = " ".join(
                f"{x:.1f},{ymap(float(v)):.1f}"
                for x, v in zip(xs, series, strict=False)
                if np.isfinite(v)
            )
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.2" points="{pts}">'
                f"<title>{name}</title></polyline>"
            )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

    def regime_timeline(self, frame: pl.DataFrame, path: Path) -> Path:
        n = min(frame.height, self.settings.visualization.max_points)
        reg = frame["bull_bear_sideways"][:n].to_numpy()
        width, height = 720, 120
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<text x="10" y="18" font-size="14">Regime Timeline</text>',
        ]
        colors = {0.0: "#e45756", 1.0: "#bab0ac", 2.0: "#54a24b"}
        bar_w = (width - 40) / max(n, 1)
        for i, v in enumerate(reg):
            if not np.isfinite(v):
                continue
            parts.append(
                f'<rect x="{40 + i * bar_w:.2f}" y="40" width="{max(bar_w, 0.5):.2f}" '
                f'height="50" fill="{colors.get(float(v), "#999")}"/>'
            )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path
