"""Method registry for Institutional Time-Series Analytics."""

from __future__ import annotations

from typing import Any, Callable


_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> Callable[..., Any]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown timeseries method: {name}")
    return _REGISTRY[name]


def list_methods() -> list[str]:
    return sorted(_REGISTRY)


def ensure_timeseries_loaded() -> list[str]:
    """Import analytical modules so register hooks (if any) and APIs are available."""
    modules = [
        "iqrp.app.timeseries.decomposition",
        "iqrp.app.timeseries.stationarity",
        "iqrp.app.timeseries.autocorrelation",
        "iqrp.app.timeseries.change_points",
        "iqrp.app.timeseries.spectral",
        "iqrp.app.timeseries.wavelets",
        "iqrp.app.timeseries.nonlinear",
        "iqrp.app.timeseries.dependence",
        "iqrp.app.timeseries.anomaly",
        "iqrp.app.timeseries.motifs",
        "iqrp.app.timeseries.alignment",
        "iqrp.app.timeseries.transforms",
        "iqrp.app.timeseries.diagnostics",
        "iqrp.app.timeseries.features",
    ]
    loaded: list[str] = []
    import importlib

    for m in modules:
        try:
            importlib.import_module(m)
            loaded.append(m)
        except Exception:  # noqa: BLE001
            continue
    # populate registry with core callables
    _populate()
    return loaded


def _populate() -> None:
    from iqrp.app.timeseries.alignment.dtw import dtw_distance
    from iqrp.app.timeseries.anomaly.robust import robust_zscore_anomalies
    from iqrp.app.timeseries.autocorrelation.acf import acf
    from iqrp.app.timeseries.change_points.pelt import pelt_detect
    from iqrp.app.timeseries.decomposition.stl import stl_decompose
    from iqrp.app.timeseries.dependence.cointegration import engle_granger
    from iqrp.app.timeseries.motifs.discovery import find_motifs
    from iqrp.app.timeseries.nonlinear.hurst import hurst_exponent
    from iqrp.app.timeseries.spectral.fft import fft_spectrum
    from iqrp.app.timeseries.stationarity.adf import adf
    from iqrp.app.timeseries.wavelets.discrete import dwt_haar

    mapping = {
        "stl": stl_decompose,
        "adf": adf,
        "acf": acf,
        "pelt": pelt_detect,
        "fft": fft_spectrum,
        "dwt": dwt_haar,
        "hurst": hurst_exponent,
        "engle_granger": engle_granger,
        "robust_anomalies": robust_zscore_anomalies,
        "motifs": find_motifs,
        "dtw": dtw_distance,
    }
    for k, v in mapping.items():
        _REGISTRY.setdefault(k, v)
