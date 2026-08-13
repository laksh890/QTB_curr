"""Discrete Haar wavelet transform."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def dwt_haar(
    x: np.ndarray | list[float],
    *,
    level: int | None = None,
) -> AnalysisResult:
    """Multi-level Haar DWT with per-level energy fractions (FULL_SAMPLE)."""
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    if n < 2:
        return AnalysisResult(
            method="dwt_haar",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="energy uniformly distributed across scales",
            alternative_hypothesis="energy concentrated at particular scales",
            parameters={"level": level},
        )
    # pad to even length power-friendly
    n_work = n if n % 2 == 0 else n - 1
    if n_work < 2:
        return AnalysisResult(
            method="dwt_haar",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="energy uniformly distributed across scales",
            alternative_hypothesis="energy concentrated at particular scales",
            parameters={"level": level},
        )
    z = finite[:n_work].copy()
    max_level = int(np.floor(np.log2(n_work)))
    L = int(level) if level is not None else max_level
    L = int(np.clip(L, 1, max_level))

    details: list[np.ndarray] = []
    approx = z
    for _ in range(L):
        if approx.size < 2:
            break
        if approx.size % 2 == 1:
            approx = approx[:-1]
        even = approx[0::2]
        odd = approx[1::2]
        # orthonormal Haar
        cA = (even + odd) / np.sqrt(2.0)
        cD = (even - odd) / np.sqrt(2.0)
        details.append(cD)
        approx = cA

    energies = [float(np.sum(d**2)) for d in details]
    e_approx = float(np.sum(approx**2))
    total = sum(energies) + e_approx
    if total < 1e-18:
        fracs = [0.0] * len(energies)
        approx_frac = 0.0
    else:
        fracs = [e / total for e in energies]
        approx_frac = e_approx / total

    return AnalysisResult(
        method="dwt_haar",
        value={
            "approximation": approx,
            "details": details,
            "energy": energies,
            "energy_fractions": fracs,
            "approx_energy": e_approx,
            "approx_energy_fraction": approx_frac,
        },
        statistic=float(max(fracs)) if fracs else approx_frac,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="energy uniformly distributed across scales",
        alternative_hypothesis="energy concentrated at particular scales",
        parameters={"level": L, "n_used": n_work},
        metadata={"n": n, "total_energy": total, "max_level": max_level},
    )
