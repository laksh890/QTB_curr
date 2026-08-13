"""Multi-method correlation analysis for feature matrices."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]

from iqrp.app.features.research._numeric import (
    distance_correlation,
    kendall,
    mutual_information,
    pearson,
    spearman,
    try_mic,
)
from iqrp.app.features.research.config import ResearchSettings


@dataclass
class CorrelationReport:
    pearson: pl.DataFrame
    spearman: pl.DataFrame
    kendall: pl.DataFrame
    distance: pl.DataFrame
    mutual_information: pl.DataFrame
    mic: pl.DataFrame | None
    cross_correlation: dict[str, list[dict[str, float]]]
    rolling_correlation: dict[str, pl.DataFrame]
    clusters: dict[int, list[str]]
    high_correlation_groups: list[list[str]]
    network_edges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pearson": self.pearson.to_dicts(),
            "spearman": self.spearman.to_dicts(),
            "kendall": self.kendall.to_dicts(),
            "distance": self.distance.to_dicts(),
            "mutual_information": self.mutual_information.to_dicts(),
            "mic": None if self.mic is None else self.mic.to_dicts(),
            "cross_correlation": self.cross_correlation,
            "clusters": {str(k): v for k, v in self.clusters.items()},
            "high_correlation_groups": self.high_correlation_groups,
            "network_edges": self.network_edges,
        }


def _matrix_frame(names: list[str], mat: np.ndarray) -> pl.DataFrame:
    data: dict[str, Any] = {"feature": names}
    for j, name in enumerate(names):
        data[name] = mat[:, j].tolist()
    return pl.DataFrame(data)


class CorrelationAnalyzer:
    def __init__(self, settings: ResearchSettings | None = None) -> None:
        self.settings = settings or ResearchSettings.default()

    def analyze(self, frame: pl.DataFrame, columns: list[str]) -> CorrelationReport:
        cfg = self.settings.correlation
        n = len(columns)
        if n == 0:
            empty = pl.DataFrame()
            return CorrelationReport(
                pearson=empty,
                spearman=empty,
                kendall=empty,
                distance=empty,
                mutual_information=empty,
                mic=None,
                cross_correlation={},
                rolling_correlation={},
                clusters={},
                high_correlation_groups=[],
            )

        arrays = {c: frame[c].cast(pl.Float64).to_numpy() for c in columns}
        pearson_m = np.eye(n)
        spearman_m = np.eye(n)
        kendall_m = np.eye(n)
        dist_m = np.eye(n)
        mi_m = np.zeros((n, n))
        mic_m = np.full((n, n), np.nan)
        mic_available = False

        def pair_work(i: int, j: int) -> tuple[int, int, dict[str, float | None]]:
            x, y = arrays[columns[i]], arrays[columns[j]]
            out: dict[str, float | None] = {}
            if cfg.pearson:
                out["pearson"] = pearson(x, y)
            if cfg.spearman:
                out["spearman"] = spearman(x, y)
            if cfg.kendall:
                out["kendall"] = kendall(x, y)
            if cfg.distance:
                out["distance"] = distance_correlation(x, y)
            if cfg.mutual_information:
                out["mi"] = mutual_information(x, y, bins=cfg.mi_bins)
            if cfg.mic:
                out["mic"] = try_mic(x, y)
            return i, j, out

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        with ThreadPoolExecutor(max_workers=max(1, self.settings.n_jobs)) as pool:
            results = list(pool.map(lambda ij: pair_work(ij[0], ij[1]), pairs))

        for i, j, out in results:
            if "pearson" in out and out["pearson"] is not None:
                pearson_m[i, j] = pearson_m[j, i] = float(out["pearson"])
            if "spearman" in out and out["spearman"] is not None:
                spearman_m[i, j] = spearman_m[j, i] = float(out["spearman"])
            if "kendall" in out and out["kendall"] is not None:
                kendall_m[i, j] = kendall_m[j, i] = float(out["kendall"])
            if "distance" in out and out["distance"] is not None:
                dist_m[i, j] = dist_m[j, i] = float(out["distance"])
            if "mi" in out and out["mi"] is not None:
                mi_m[i, j] = mi_m[j, i] = float(out["mi"])
            if "mic" in out and out["mic"] is not None:
                mic_available = True
                mic_m[i, j] = mic_m[j, i] = float(out["mic"])

        # Cross-correlation vs first feature as reference (lagged)
        cross: dict[str, list[dict[str, float]]] = {}
        max_lag = cfg.cross_correlation_max_lag
        if n >= 1:
            ref = arrays[columns[0]]
            for name in columns[1:]:
                series = arrays[name]
                lags: list[dict[str, float]] = []
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        r = pearson(ref[-lag:], series[: len(series) + lag])
                    elif lag > 0:
                        r = pearson(ref[: len(ref) - lag], series[lag:])
                    else:
                        r = pearson(ref, series)
                    lags.append({"lag": float(lag), "corr": r})
                cross[name] = lags

        # Rolling correlation vs reference
        rolling: dict[str, pl.DataFrame] = {}
        w = cfg.rolling_window
        if n >= 2 and w > 5:
            ref = arrays[columns[0]]
            for name in columns[1:]:
                y = arrays[name]
                vals = np.full(len(ref), np.nan)
                for t in range(w - 1, len(ref)):
                    vals[t] = pearson(ref[t - w + 1 : t + 1], y[t - w + 1 : t + 1])
                rolling[name] = pl.DataFrame({"index": np.arange(len(ref)), "rolling_corr": vals})

        # Hierarchical clustering on |pearson| distance
        clusters: dict[int, list[str]] = {}
        if n >= 2:
            dist = np.clip(1.0 - np.abs(np.nan_to_num(pearson_m, nan=0.0)), 0.0, 1.0)
            np.fill_diagonal(dist, 0.0)
            condensed = squareform(dist, checks=False)
            condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0, neginf=1.0)
            z = linkage(condensed, method=cfg.clustering_linkage)
            labels = fcluster(z, t=cfg.clustering_distance_threshold, criterion="distance")
            for lab, name in zip(labels, columns, strict=False):
                clusters.setdefault(int(lab), []).append(name)

        thr = cfg.high_correlation_threshold
        groups: list[list[str]] = []
        visited: set[str] = set()
        for i, a in enumerate(columns):
            if a in visited:
                continue
            group = [a]
            for j, b in enumerate(columns):
                if i == j:
                    continue
                if abs(pearson_m[i, j]) >= thr:
                    group.append(b)
            if len(group) > 1:
                for g in group:
                    visited.add(g)
                groups.append(sorted(set(group)))

        edges = []
        for i, a in enumerate(columns):
            for j in range(i + 1, n):
                r = float(pearson_m[i, j])
                if abs(r) >= thr:
                    edges.append({"source": a, "target": columns[j], "weight": r})

        return CorrelationReport(
            pearson=_matrix_frame(columns, pearson_m),
            spearman=_matrix_frame(columns, spearman_m),
            kendall=_matrix_frame(columns, kendall_m),
            distance=_matrix_frame(columns, dist_m),
            mutual_information=_matrix_frame(columns, mi_m),
            mic=_matrix_frame(columns, mic_m) if mic_available else None,
            cross_correlation=cross,
            rolling_correlation=rolling,
            clusters=clusters,
            high_correlation_groups=groups,
            network_edges=edges,
        )
