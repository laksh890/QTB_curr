"""Conditional variance recursions for the GARCH family."""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # optional acceleration
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # noqa: BLE001
    _HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def deco(fn):  # type: ignore[no-untyped-def]
            return fn

        if args and callable(args[0]):
            return args[0]
        return deco


@njit(cache=True)
def _garch_core(eps2: np.ndarray, omega: float, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    n = eps2.shape[0]
    p = alpha.shape[0]
    q = beta.shape[0]
    var = np.empty(n)
    backcast = float(np.mean(eps2)) if n else 1.0
    var[0] = max(backcast, 1e-12)
    for t in range(1, n):
        s = omega
        for i in range(p):
            if t - 1 - i >= 0:
                s += alpha[i] * eps2[t - 1 - i]
            else:
                s += alpha[i] * backcast
        for j in range(q):
            if t - 1 - j >= 0:
                s += beta[j] * var[t - 1 - j]
            else:
                s += beta[j] * backcast
        var[t] = s if s > 1e-12 else 1e-12
    return var


@njit(cache=True)
def _gjr_core(
    eps: np.ndarray, eps2: np.ndarray, omega: float, alpha: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    n = eps2.shape[0]
    p = alpha.shape[0]
    o = gamma.shape[0]
    q = beta.shape[0]
    var = np.empty(n)
    backcast = float(np.mean(eps2)) if n else 1.0
    var[0] = max(backcast, 1e-12)
    for t in range(1, n):
        s = omega
        for i in range(p):
            idx = t - 1 - i
            s += alpha[i] * (eps2[idx] if idx >= 0 else backcast)
        for i in range(o):
            idx = t - 1 - i
            e = eps[idx] if idx >= 0 else 0.0
            e2 = eps2[idx] if idx >= 0 else backcast
            s += gamma[i] * e2 * (1.0 if e < 0 else 0.0)
        for j in range(q):
            idx = t - 1 - j
            s += beta[j] * (var[idx] if idx >= 0 else backcast)
        var[t] = s if s > 1e-12 else 1e-12
    return var


@njit(cache=True)
def _egarch_core(
    eps: np.ndarray, omega: float, alpha: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    n = eps.shape[0]
    p = alpha.shape[0]
    o = gamma.shape[0]
    q = beta.shape[0]
    logv = np.empty(n)
    backcast = float(np.log(max(np.mean(eps * eps), 1e-12)))
    logv[0] = backcast
    for t in range(1, n):
        s = omega
        for j in range(q):
            idx = t - 1 - j
            s += beta[j] * (logv[idx] if idx >= 0 else backcast)
        sig = np.exp(0.5 * (logv[t - 1] if t else backcast))
        z = eps[t - 1] / max(sig, 1e-12)
        for i in range(p):
            s += alpha[i] * (abs(z) - np.sqrt(2.0 / np.pi))
        for i in range(o):
            s += gamma[i] * z
        logv[t] = s
    return np.exp(logv)


@njit(cache=True)
def _ewma_core(eps2: np.ndarray, lam: float) -> np.ndarray:
    n = eps2.shape[0]
    var = np.empty(n)
    var[0] = max(float(np.mean(eps2)) if n else 1.0, 1e-12)
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1.0 - lam) * eps2[t - 1]
        if var[t] < 1e-12:
            var[t] = 1e-12
    return var


@njit(cache=True)
def _aparch_core(
    eps: np.ndarray,
    omega: float,
    alpha: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    delta: float,
) -> np.ndarray:
    n = eps.shape[0]
    p = alpha.shape[0]
    o = gamma.shape[0]
    q = beta.shape[0]
    power = np.empty(n)
    backcast = float(np.mean(np.abs(eps) ** delta)) if n else 1.0
    power[0] = max(backcast, 1e-12)
    for t in range(1, n):
        s = omega
        for i in range(max(p, o)):
            idx = t - 1 - i
            e = eps[idx] if idx >= 0 else 0.0
            a = alpha[i] if i < p else 0.0
            g = gamma[i] if i < o else 0.0
            s += a * (abs(e) - g * e) ** delta
        for j in range(q):
            idx = t - 1 - j
            s += beta[j] * (power[idx] if idx >= 0 else backcast)
        power[t] = s if s > 1e-12 else 1e-12
    return power ** (2.0 / max(delta, 1e-6))


@njit(cache=True)
def _cgarch_core(
    eps2: np.ndarray, omega: float, rho: float, phi: float, alpha: float, beta: float
) -> tuple:
    n = eps2.shape[0]
    q = np.empty(n)
    h = np.empty(n)
    backcast = float(np.mean(eps2)) if n else 1.0
    q[0] = max(backcast, 1e-12)
    h[0] = q[0]
    for t in range(1, n):
        q[t] = omega + rho * q[t - 1] + phi * (eps2[t - 1] - h[t - 1])
        if q[t] < 1e-12:
            q[t] = 1e-12
        h[t] = q[t] + alpha * (eps2[t - 1] - q[t - 1]) + beta * (h[t - 1] - q[t - 1])
        if h[t] < 1e-12:
            h[t] = 1e-12
    return h, q


def garch_variance(eps: np.ndarray, omega: float, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    e = np.asarray(eps, dtype=np.float64).reshape(-1)
    return _garch_core(e * e, float(omega), np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64))


def arch_variance(eps: np.ndarray, omega: float, alpha: np.ndarray) -> np.ndarray:
    return garch_variance(eps, omega, alpha, np.zeros(0, dtype=np.float64))


def ewma_variance(eps: np.ndarray, lam: float = 0.94) -> np.ndarray:
    e = np.asarray(eps, dtype=np.float64).reshape(-1)
    return _ewma_core(e * e, float(np.clip(lam, 1e-6, 1 - 1e-6)))


def gjr_variance(
    eps: np.ndarray, omega: float, alpha: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    e = np.asarray(eps, dtype=np.float64).reshape(-1)
    return _gjr_core(
        e,
        e * e,
        float(omega),
        np.asarray(alpha, dtype=np.float64),
        np.asarray(gamma, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
    )


def egarch_variance(
    eps: np.ndarray, omega: float, alpha: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    e = np.asarray(eps, dtype=np.float64).reshape(-1)
    return _egarch_core(
        e,
        float(omega),
        np.asarray(alpha, dtype=np.float64),
        np.asarray(gamma, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
    )


def aparch_variance(
    eps: np.ndarray,
    omega: float,
    alpha: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    delta: float = 2.0,
) -> np.ndarray:
    e = np.asarray(eps, dtype=np.float64).reshape(-1)
    return _aparch_core(
        e,
        float(omega),
        np.asarray(alpha, dtype=np.float64),
        np.asarray(gamma, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
        float(delta),
    )


def cgarch_variance(
    eps: np.ndarray, omega: float, rho: float, phi: float, alpha: float, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    e2 = np.asarray(eps, dtype=np.float64).reshape(-1) ** 2
    h, q = _cgarch_core(e2, float(omega), float(rho), float(phi), float(alpha), float(beta))
    return np.asarray(h), np.asarray(q)


def figarch_variance(
    eps: np.ndarray, omega: float, phi: float, d: float, beta: float
) -> np.ndarray:
    """FIGARCH(1,d,1) via truncated ARCH(∞) weights."""
    e2 = np.asarray(eps, dtype=np.float64).reshape(-1)
    n = e2.size
    d = float(np.clip(d, 1e-4, 0.99))
    phi = float(np.clip(phi, 0.0, 0.99))
    beta = float(np.clip(beta, 0.0, 0.99))
    # λ_k weights truncated
    trunc = min(n, 200)
    lam = np.zeros(trunc)
    # (1-L)^d expansion coeffs π
    pi = np.zeros(trunc)
    pi[0] = 1.0
    for k in range(1, trunc):
        pi[k] = (k - 1 - d) / k * pi[k - 1]
    # λ(L) = (1 - φL)(1-L)^d (1-βL)^{-1} roughly → recursive
    # simplified Baillie weights
    lam[0] = phi - beta + d
    for k in range(1, trunc):
        lam[k] = beta * lam[k - 1] + (d / (k + 1) - (d - phi) / k) * (pi[k - 1] if k else 0.0)
        # stabilize
        lam[k] = max(min(lam[k], 1.0), 0.0)
    s_lam = float(np.sum(lam)) or 1.0
    lam = lam / s_lam * max(1.0 - beta, 1e-6)
    var = np.empty(n)
    backcast = float(np.mean(e2)) if n else 1.0
    var[0] = max(backcast, 1e-12)
    for t in range(1, n):
        s = omega
        for k in range(min(trunc, t)):
            s += lam[k] * e2[t - 1 - k]
        s += beta * var[t - 1]
        var[t] = s if s > 1e-12 else 1e-12
    return var


def forecast_garch_path(
    last_eps2: float,
    last_var: float,
    omega: float,
    alpha: float,
    beta: float,
    *,
    horizon: int,
) -> np.ndarray:
    """Multi-step GARCH(1,1) variance forecasts."""
    h = max(int(horizon), 1)
    out = np.empty(h)
    a, b = float(alpha), float(beta)
    persist = a + b
    uncond = omega / max(1.0 - persist, 1e-8) if persist < 1 else last_var
    # h=1
    out[0] = omega + a * last_eps2 + b * last_var
    for i in range(1, h):
        out[i] = uncond + (persist**i) * (out[0] - uncond)
        out[i] = max(out[i], 1e-12)
    out[0] = max(out[0], 1e-12)
    return out
