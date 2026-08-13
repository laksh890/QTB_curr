"""Bayesian Online Changepoint Detection (simple BOCPD)."""

from __future__ import annotations

import math

import numpy as np

from iqrp.app.timeseries.base import ChangePointResult, TemporalMode, as_float_array


def bayesian_online_changepoint(
    x: np.ndarray | list[float],
    *,
    hazard: float = 1.0 / 200.0,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    threshold: float = 0.5,
) -> ChangePointResult:
    """Adams & MacKay BOCPD with Normal-Inverse-Gamma conjugate prior (CAUSAL).

    Returns indices where the posterior run-length probability mass at r=0
    (changepoint probability) exceeds ``threshold``.
    """
    y = as_float_array(x)
    n = y.size
    if n < 5:
        return ChangePointResult(
            method="bayesian_online_changepoint",
            indices=[],
            scores=None,
            kind="mean",
            parameters={
                "hazard": hazard,
                "mu0": mu0,
                "kappa0": kappa0,
                "alpha0": alpha0,
                "beta0": beta0,
                "threshold": threshold,
            },
            temporal_mode=TemporalMode.CAUSAL,
            metadata={"status": "insufficient_data", "n": n},
        )

    h = float(np.clip(hazard, 1e-8, 1.0 - 1e-8))
    # R[t, r] ≈ P(run_length = r | x_{1:t}); we keep a growing vector pruned
    max_r = n
    R = np.zeros((n + 1, max_r + 1), dtype=np.float64)
    R[0, 0] = 1.0

    mu = np.full(max_r + 1, mu0, dtype=np.float64)
    kappa = np.full(max_r + 1, kappa0, dtype=np.float64)
    alpha = np.full(max_r + 1, alpha0, dtype=np.float64)
    beta = np.full(max_r + 1, beta0, dtype=np.float64)

    cp_prob = np.zeros(n, dtype=np.float64)
    indices: list[int] = []

    for t in range(1, n + 1):
        xt = y[t - 1]
        if not np.isfinite(xt):
            R[t] = R[t - 1]
            continue

        # predictive probabilities under each run length
        pred = np.zeros(t, dtype=np.float64)
        for r in range(t):
            pred[r] = _student_t_pdf(xt, mu[r], kappa[r], alpha[r], beta[r])

        # growth probabilities
        R[t, 1 : t + 1] = R[t - 1, :t] * pred * (1.0 - h)
        # changepoint probability (r=0)
        R[t, 0] = np.sum(R[t - 1, :t] * pred * h)
        # normalize
        evidence = np.sum(R[t, : t + 1])
        if evidence < 1e-300:
            R[t, 0] = 1.0
            evidence = 1.0
        R[t, : t + 1] /= evidence
        cp_prob[t - 1] = float(R[t, 0])
        if R[t, 0] >= threshold and t > 1:
            indices.append(t - 1)

        # update sufficient stats for new run lengths (backwards to avoid overwrite)
        new_mu = np.empty(t + 1, dtype=np.float64)
        new_kappa = np.empty(t + 1, dtype=np.float64)
        new_alpha = np.empty(t + 1, dtype=np.float64)
        new_beta = np.empty(t + 1, dtype=np.float64)
        new_mu[0] = mu0
        new_kappa[0] = kappa0
        new_alpha[0] = alpha0
        new_beta[0] = beta0
        for r in range(t):
            k = kappa[r]
            m = mu[r]
            a = alpha[r]
            b = beta[r]
            new_kappa[r + 1] = k + 1.0
            new_mu[r + 1] = (k * m + xt) / (k + 1.0)
            new_alpha[r + 1] = a + 0.5
            new_beta[r + 1] = b + 0.5 * k * (xt - m) ** 2 / (k + 1.0)
        mu[: t + 1] = new_mu
        kappa[: t + 1] = new_kappa
        alpha[: t + 1] = new_alpha
        beta[: t + 1] = new_beta

    return ChangePointResult(
        method="bayesian_online_changepoint",
        indices=indices,
        scores=cp_prob,
        kind="mean",
        parameters={
            "hazard": h,
            "mu0": mu0,
            "kappa0": kappa0,
            "alpha0": alpha0,
            "beta0": beta0,
            "threshold": threshold,
        },
        temporal_mode=TemporalMode.CAUSAL,
        metadata={"n": n, "max_cp_prob": float(np.nanmax(cp_prob))},
    )


def _student_t_pdf(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    """Predictive Student-t density for N-IG conjugate (up to stable scale)."""
    # var = beta * (kappa+1) / (alpha * kappa)
    # df = 2 alpha
    df = 2.0 * alpha
    var = beta * (kappa + 1.0) / (alpha * kappa + 1e-18)
    var = max(var, 1e-12)
    z = (x - mu) / np.sqrt(var)
    # unnormalized but relative comparisons matter; use log-stable approx
    # Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + z²/ν)^(-(ν+1)/2)
    log_c = (
        _log_gamma(0.5 * (df + 1.0))
        - _log_gamma(0.5 * df)
        - 0.5 * np.log(df * np.pi * var)
    )
    log_p = log_c - 0.5 * (df + 1.0) * np.log1p((z * z) / df)
    return float(np.exp(np.clip(log_p, -700, 700)))


def _log_gamma(z: float) -> float:
    return float(math.lgamma(z))
