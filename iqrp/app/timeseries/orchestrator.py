"""Institutional Time-Series Analytics Engine — analytical discovery API.

Produces measurements and evidence. Does NOT generate trading signals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from iqrp.app.timeseries.alignment.dtw import dtw_distance
from iqrp.app.timeseries.alignment.soft_dtw import soft_dtw
from iqrp.app.timeseries.anomaly.isolation_forest import isolation_forest_anomalies
from iqrp.app.timeseries.anomaly.matrix_profile import matrix_profile_anomalies
from iqrp.app.timeseries.anomaly.robust import robust_zscore_anomalies
from iqrp.app.timeseries.anomaly.statistical import zscore_anomalies
from iqrp.app.timeseries.autocorrelation.acf import acf
from iqrp.app.timeseries.autocorrelation.cross_correlation import ccf
from iqrp.app.timeseries.autocorrelation.pacf import pacf
from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.change_points.bayesian import bayesian_online_changepoint
from iqrp.app.timeseries.change_points.binary_segmentation import binseg_detect
from iqrp.app.timeseries.change_points.cusum import cusum_detect
from iqrp.app.timeseries.change_points.online import online_cusum
from iqrp.app.timeseries.change_points.pelt import pelt_detect
from iqrp.app.timeseries.config import TimeSeriesSettings
from iqrp.app.timeseries.decomposition.classical import classical_decompose
from iqrp.app.timeseries.decomposition.mstl import mstl_decompose
from iqrp.app.timeseries.decomposition.stl import stl_decompose
from iqrp.app.timeseries.dependence.cointegration import engle_granger, johansen_trace
from iqrp.app.timeseries.dependence.distance_correlation import distance_correlation
from iqrp.app.timeseries.dependence.granger import granger_causality
from iqrp.app.timeseries.dependence.mutual_information import mutual_information
from iqrp.app.timeseries.dependence.tail_dependence import empirical_tail_dependence
from iqrp.app.timeseries.diagnostics import full_diagnostics
from iqrp.app.timeseries.features import extract_features
from iqrp.app.timeseries.motifs.discovery import find_motifs
from iqrp.app.timeseries.motifs.discord import find_discords
from iqrp.app.timeseries.multiple_testing import adjust_pvalues
from iqrp.app.timeseries.nonlinear.approximate_entropy import approximate_entropy
from iqrp.app.timeseries.nonlinear.entropy import shannon_entropy
from iqrp.app.timeseries.nonlinear.hurst import hurst_exponent
from iqrp.app.timeseries.nonlinear.permutation_entropy import permutation_entropy
from iqrp.app.timeseries.nonlinear.sample_entropy import sample_entropy
from iqrp.app.timeseries.registry import ensure_timeseries_loaded, list_methods
from iqrp.app.timeseries.serializer import TimeSeriesSerializer
from iqrp.app.timeseries.spectral.fft import dominant_frequencies, fft_spectrum
from iqrp.app.timeseries.spectral.periodogram import periodogram
from iqrp.app.timeseries.spectral.welch import welch_psd
from iqrp.app.timeseries.stationarity.adf import adf
from iqrp.app.timeseries.stationarity.kpss import kpss
from iqrp.app.timeseries.stationarity.phillips_perron import phillips_perron
from iqrp.app.timeseries.stationarity.variance_ratio import variance_ratio
from iqrp.app.timeseries.transforms import TimeSeriesTransformer
from iqrp.app.timeseries.visualization import (
    acf_chart,
    anomaly_chart,
    change_point_chart,
    decomposition_chart,
    spectrum_chart,
)
from iqrp.app.timeseries.wavelets.continuous import cwt_morlet
from iqrp.app.timeseries.wavelets.denoising import wavelet_denoise
from iqrp.app.timeseries.wavelets.discrete import dwt_haar


class TimeSeriesAnalyticsEngine:
    """Unified façade for institutional time-series analytical discovery."""

    def __init__(self, settings: TimeSeriesSettings | None = None) -> None:
        self.settings = settings or TimeSeriesSettings.default()
        self._transformer = TimeSeriesTransformer(
            method=self.settings.transform.method,
            window=self.settings.transform.window,
            temporal_mode=self.settings.transform.temporal_mode,
        )
        self._last: dict[str, Any] = {}
        self._fitted = False
        self._serializer = TimeSeriesSerializer()
        ensure_timeseries_loaded()

    # ---- transform API (leakage-safe) ---------------------------------
    def fit(self, x: np.ndarray | list[float]) -> TimeSeriesAnalyticsEngine:
        self._transformer.fit(x)
        self._fitted = True
        return self

    def transform(self, x: np.ndarray | list[float]) -> np.ndarray:
        return self._transformer.transform(x)

    def fit_transform(self, x: np.ndarray | list[float]) -> np.ndarray:
        out = self._transformer.fit_transform(x)
        self._fitted = True
        return out

    # ---- analysis API -------------------------------------------------
    def analyze(self, x: np.ndarray | list[float]) -> dict[str, Any]:
        """Run a standard analytical battery (discovery, not prediction)."""
        arr = as_float_array(x)
        report = {
            "stationarity": self.stationarity(arr),
            "decomposition": self.decompose(arr).to_dict(),
            "acf": self.correlate(arr).to_dict(),
            "change_points": self.change_points(arr).to_dict(),
            "spectral": self.spectral_analysis(arr),
            "hurst": self.hurst(arr).to_dict(),
            "entropy": self.entropy(arr),
            "anomalies": self.anomalies(arr).to_dict(),
            "diagnostics": {k: v.to_dict() for k, v in self.diagnostics(arr).items()},
            "features": self.features(arr).to_dict(),
            "disclaimer": "Analytical measurements only — not trading signals.",
        }
        self._last = report
        return report

    def detect(self, x: np.ndarray | list[float], *, what: str = "change_points") -> Any:
        if what == "change_points":
            return self.change_points(x)
        if what == "anomalies":
            return self.anomalies(x)
        if what == "motifs":
            return self.motifs(x)
        if what == "discords":
            return find_discords(x, window=self.settings.motif.window)
        raise ValueError(f"Unknown detect target: {what}")

    def decompose(
        self,
        x: np.ndarray | list[float],
        *,
        method: str | None = None,
        period: int | None = None,
        model: str | None = None,
    ) -> Any:
        m = method or self.settings.decomposition.method
        p = period or self.settings.decomposition.period
        mod = model or self.settings.decomposition.model
        if m == "classical":
            return classical_decompose(x, period=p, model=mod)
        if m == "mstl":
            return mstl_decompose(x, periods=(p, max(p * 7, p + 1)), robust=self.settings.decomposition.robust)
        return stl_decompose(x, period=p, robust=self.settings.decomposition.robust)

    def correlate(
        self,
        x: np.ndarray | list[float],
        y: np.ndarray | list[float] | None = None,
        *,
        kind: Literal["acf", "pacf", "ccf"] = "acf",
        nlags: int | None = None,
    ) -> AnalysisResult:
        if kind == "pacf":
            return pacf(x, nlags=nlags)
        if kind == "ccf":
            if y is None:
                raise ValueError("ccf requires y")
            return ccf(x, y, nlags=nlags)
        return acf(x, nlags=nlags)

    def stationarity(self, x: np.ndarray | list[float]) -> dict[str, Any]:
        alpha = self.settings.stationarity.alpha
        results = {
            "adf": adf(x, max_lag=self.settings.stationarity.max_lag, alpha=alpha),
            "kpss": kpss(x, alpha=alpha),
            "phillips_perron": phillips_perron(x, alpha=alpha),
            "variance_ratio": variance_ratio(x, alpha=alpha),
        }
        pvals = [r.pvalue for r in results.values() if r.pvalue is not None and np.isfinite(r.pvalue)]
        adj = adjust_pvalues(pvals, method=self.settings.multiple_testing.method, alpha=self.settings.multiple_testing.alpha)
        return {
            "tests": {k: v.to_dict() for k, v in results.items()},
            "multiple_testing": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in adj.items()},
            "note": "Do not treat unadjusted significance as a profitable feature.",
        }

    def change_points(
        self,
        x: np.ndarray | list[float],
        *,
        method: str | None = None,
    ) -> Any:
        m = method or self.settings.change_points.method
        pen = self.settings.change_points.penalty
        ms = self.settings.change_points.min_size
        if m == "cusum":
            return cusum_detect(x)
        if m == "binseg":
            return binseg_detect(x, min_size=ms)
        if m == "bayesian":
            return bayesian_online_changepoint(x)
        if m == "online":
            return online_cusum(x)
        return pelt_detect(x, penalty=pen, min_size=ms)

    def spectral_analysis(self, x: np.ndarray | list[float]) -> dict[str, Any]:
        fft_r = fft_spectrum(x, detrend=self.settings.spectral.detrend)
        welch_r = welch_psd(x, nperseg=self.settings.spectral.nperseg)
        peri = periodogram(x)
        dom = dominant_frequencies(x, top_k=3)
        return {
            "fft": fft_r.to_dict(),
            "welch": welch_r.to_dict(),
            "periodogram": peri.to_dict(),
            "dominant": dom.to_dict(),
        }

    def wavelet_analysis(self, x: np.ndarray | list[float]) -> dict[str, Any]:
        dwt = dwt_haar(x, level=self.settings.wavelet.level)
        cwt = cwt_morlet(x)
        den = wavelet_denoise(x, threshold=self.settings.wavelet.threshold)
        return {"dwt": dwt.to_dict(), "cwt": cwt.to_dict(), "denoised": den.to_dict()}

    def entropy(self, x: np.ndarray | list[float]) -> dict[str, Any]:
        return {
            "shannon": shannon_entropy(x).to_dict(),
            "sample": sample_entropy(x).to_dict(),
            "approximate": approximate_entropy(x).to_dict(),
            "permutation": permutation_entropy(x).to_dict(),
            "disclaimer": "Statistical descriptors only — not guaranteed predictive signals.",
        }

    def hurst(self, x: np.ndarray | list[float]) -> AnalysisResult:
        return hurst_exponent(x)

    def cointegration(
        self,
        x: np.ndarray | list[float],
        y: np.ndarray | list[float],
        *,
        method: Literal["engle_granger", "johansen"] = "engle_granger",
    ) -> AnalysisResult:
        if method == "johansen":
            return johansen_trace(x, y)
        return engle_granger(x, y)

    def dependence(
        self,
        x: np.ndarray | list[float],
        y: np.ndarray | list[float],
    ) -> dict[str, Any]:
        return {
            "granger": granger_causality(x, y).to_dict(),
            "mutual_information": mutual_information(x, y).to_dict(),
            "distance_correlation": distance_correlation(x, y).to_dict(),
            "tail_dependence": empirical_tail_dependence(x, y).to_dict(),
            "cointegration": self.cointegration(x, y).to_dict(),
        }

    def anomalies(self, x: np.ndarray | list[float], *, method: str | None = None) -> AnalysisResult:
        m = method or self.settings.anomaly.method
        if m == "statistical":
            return zscore_anomalies(x, threshold=self.settings.anomaly.z_threshold)
        if m == "isolation_forest":
            return isolation_forest_anomalies(x, contamination=self.settings.anomaly.contamination)
        if m == "matrix_profile":
            return matrix_profile_anomalies(x, window=self.settings.motif.window)
        return robust_zscore_anomalies(x, threshold=self.settings.anomaly.z_threshold)

    def motifs(self, x: np.ndarray | list[float]) -> AnalysisResult:
        return find_motifs(x, window=self.settings.motif.window, top_k=self.settings.motif.top_k)

    def dtw(
        self,
        x: np.ndarray | list[float],
        y: np.ndarray | list[float],
        *,
        soft: bool = False,
    ) -> AnalysisResult:
        if soft:
            return soft_dtw(x, y)
        return dtw_distance(x, y)

    def features(self, x: np.ndarray | list[float]) -> AnalysisResult:
        cfg = self.settings.features
        return extract_features(
            x,
            period=self.settings.decomposition.period,
            window=cfg.window,
            include_entropy=cfg.include_entropy,
            include_hurst=cfg.include_hurst,
            include_spectral=cfg.include_spectral,
        )

    def diagnostics(self, x: np.ndarray | list[float]) -> dict[str, AnalysisResult]:
        return full_diagnostics(x, period=self.settings.decomposition.period, alpha=self.settings.stationarity.alpha)

    def visualize(self, x: np.ndarray | list[float]) -> dict[str, Any]:
        arr = as_float_array(x)
        dec = self.decompose(arr)
        cp = self.change_points(arr)
        ac = self.correlate(arr)
        an = self.anomalies(arr)
        spec = fft_spectrum(arr)
        freqs = amp = None
        if isinstance(spec.value, dict):
            freqs = spec.value.get("frequencies")
            amp = spec.value.get("power")
        anomaly_idx = an.value if isinstance(an.value, list) else (an.metadata or {}).get("indices", [])
        return {
            "decomposition": decomposition_chart(dec),
            "change_points": change_point_chart(arr, cp),
            "acf": acf_chart(ac),
            "anomalies": anomaly_chart(arr, list(anomaly_idx) if anomaly_idx else []),
            "spectrum": spectrum_chart(freqs, amp) if freqs is not None else {},
        }

    def methods(self) -> list[str]:
        return list_methods()

    def save(self, path: str | Path) -> Path:
        return self._serializer.save(self, path)

    @classmethod
    def load(cls, path: str | Path, settings: TimeSeriesSettings | None = None) -> TimeSeriesAnalyticsEngine:
        ser = TimeSeriesSerializer()
        payload = ser.load(path)
        eng = cls(settings=settings or TimeSeriesSettings.default())
        eng.import_state(payload)
        return eng

    def export_state(self) -> dict[str, Any]:
        return {
            "settings": self.settings.model_dump(),
            "fitted": self._fitted,
            "last_keys": list(self._last.keys()),
            "data_version": self.settings.data_version,
        }

    def import_state(self, payload: dict[str, Any]) -> TimeSeriesAnalyticsEngine:
        if "settings" in payload:
            try:
                self.settings = TimeSeriesSettings.from_mapping(payload["settings"])
            except Exception:  # noqa: BLE001
                pass
        self._fitted = bool(payload.get("fitted", False))
        return self
