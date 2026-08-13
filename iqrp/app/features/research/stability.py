"""Feature stability across time (rolling IC, variance, decay, drift)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.features.research._numeric import (
    information_coefficient,
    mutual_information,
    pearson,
    safe_nanmean,
)
from iqrp.app.features.research.config import ResearchSettings
from iqrp.app.features.research.targets import build_targets


@dataclass
class FeatureStabilityReport:
    feature: str
    rolling_mean_stability: float
    rolling_variance_stability: float
    rolling_ic_mean: float
    rolling_ic_std: float
    rolling_mi_mean: float
    rolling_correlation_mean: float
    feature_decay: float
    parameter_drift: float
    stability_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StabilityAnalyzer:
    def __init__(self, settings: ResearchSettings | None = None) -> None:
        self.settings = settings or ResearchSettings.default()

    def analyze(self, frame: pl.DataFrame, columns: list[str]) -> dict[str, FeatureStabilityReport]:
        targets = build_targets(frame, self.settings)
        y = targets["future_return"].cast(pl.Float64).to_numpy()
        cfg = self.settings.stability
        out: dict[str, FeatureStabilityReport] = {}
        for name in columns:
            x = frame[name].cast(pl.Float64).to_numpy()
            out[name] = self._one(name, x, y, cfg.rolling_window, cfg.step, cfg.ic_min_obs)
        return out

    def _one(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        window: int,
        step: int,
        min_obs: int,
    ) -> FeatureStabilityReport:
        means: list[float] = []
        vars_: list[float] = []
        ics: list[float] = []
        mis: list[float] = []
        corrs: list[float] = []
        for start in range(0, max(0, len(x) - window + 1), step):
            sl = slice(start, start + window)
            xs, ys = x[sl], y[sl]
            if np.isfinite(xs).sum() < min_obs:
                continue
            finite_x = xs[np.isfinite(xs)]
            if finite_x.size == 0:
                continue
            means.append(float(np.mean(finite_x)))
            vars_.append(float(np.var(finite_x)))
            ics.append(information_coefficient(xs, ys))
            mis.append(mutual_information(xs, ys, bins=self.settings.predictive.mi_bins))
            corrs.append(pearson(xs, ys))

        mean_stab = _stability_from_series(np.asarray(means, dtype=np.float64))
        var_stab = _stability_from_series(np.asarray(vars_, dtype=np.float64))
        ic_arr = np.asarray(ics, dtype=np.float64)
        mi_arr = np.asarray(mis, dtype=np.float64)
        corr_arr = np.asarray(corrs, dtype=np.float64)

        # Feature decay: half-life style — ratio of late vs early |IC|
        decay = _decay(ic_arr, self.settings.stability.decay_half_life_bars // max(step, 1))
        # Parameter drift: std of rolling mean / (|mean| + eps)
        pw = self.settings.stability.parameter_drift_window
        param_drift = _parameter_drift(x, pw)

        stab_components = [mean_stab, var_stab]
        finite_ic = ic_arr[np.isfinite(ic_arr)]
        if finite_ic.size:
            abs_ic = np.abs(finite_ic)
            stab_components.append(
                float(np.clip(1.0 - (np.std(abs_ic) / (np.mean(abs_ic) + 1e-9)), 0, 1))
            )
        stability_score = float(np.mean(stab_components)) * 100.0

        return FeatureStabilityReport(
            feature=name,
            rolling_mean_stability=mean_stab,
            rolling_variance_stability=var_stab,
            rolling_ic_mean=safe_nanmean(ic_arr),
            rolling_ic_std=float(np.std(finite_ic)) if finite_ic.size else float("nan"),
            rolling_mi_mean=safe_nanmean(mi_arr),
            rolling_correlation_mean=safe_nanmean(corr_arr),
            feature_decay=decay,
            parameter_drift=param_drift,
            stability_score=stability_score,
        )


def _stability_from_series(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if len(v) < 2:
        return 0.0
    cv = float(np.std(v) / (abs(np.mean(v)) + 1e-9))
    return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))


def _decay(ic: np.ndarray, half_segments: int) -> float:
    v = ic[np.isfinite(ic)]
    if len(v) < 4:
        return float("nan")
    mid = len(v) // 2
    early = np.mean(np.abs(v[:mid]))
    late = np.mean(np.abs(v[mid:]))
    _ = half_segments
    if early <= 1e-12:
        return 0.0
    return float(np.clip(late / early, 0.0, 2.0))


def _parameter_drift(x: np.ndarray, window: int) -> float:
    if len(x) < window * 2:
        return float("nan")
    means = []
    for i in range(0, len(x) - window + 1, max(1, window // 2)):
        chunk = x[i : i + window]
        finite = chunk[np.isfinite(chunk)]
        if finite.size == 0:
            continue
        means.append(float(np.mean(finite)))
    arr = np.asarray(means, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr) / (abs(np.mean(arr)) + 1e-9))
