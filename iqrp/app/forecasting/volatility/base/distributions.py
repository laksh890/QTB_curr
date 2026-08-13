"""Error distribution log-densities for volatility MLE."""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
from scipy import special
from scipy.special import gamma as gamma_fn

DistName = Literal["gaussian", "student_t", "skew_t", "ged", "laplace", "custom"]


def logpdf_gaussian(z: np.ndarray) -> np.ndarray:
    return -0.5 * np.log(2 * np.pi) - 0.5 * z**2


def logpdf_student_t(z: np.ndarray, *, df: float = 8.0) -> np.ndarray:
    nu = max(float(df), 2.01)
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
    return c - ((nu + 1) / 2) * np.log1p(z**2 / nu)


def logpdf_skew_t(z: np.ndarray, *, df: float = 8.0, skew: float = 0.0) -> np.ndarray:
    """Hansen (1994) skewed-t approximation via two-piece scaling."""
    nu = max(float(df), 2.01)
    lam = float(np.clip(skew, -0.99, 0.99))
    a = 4 * lam * (nu - 2) / ((nu - 1) * np.sqrt(np.pi * (nu - 2))) * gamma_fn((nu + 1) / 2) / max(
        gamma_fn(nu / 2), 1e-300
    )
    a = float(a) if np.isfinite(a) else 0.0
    b = np.sqrt(1 + 3 * lam**2 - a**2)
    # standardize
    x = z
    out = np.empty_like(x, dtype=np.float64)
    mask = x < -a / b
    # left / right pieces
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    left = (b * x + a) / (1 - lam)
    right = (b * x + a) / (1 + lam)
    out[mask] = np.log(b) + c - ((nu + 1) / 2) * np.log1p(left[mask] ** 2 / (nu - 2))
    out[~mask] = np.log(b) + c - ((nu + 1) / 2) * np.log1p(right[~mask] ** 2 / (nu - 2))
    return out


def logpdf_ged(z: np.ndarray, *, nu: float = 1.5) -> np.ndarray:
    nu = max(float(nu), 0.5)
    lam = np.sqrt(2 ** (-2 / nu) * gamma_fn(1 / nu) / max(gamma_fn(3 / nu), 1e-300))
    c = np.log(nu) - np.log(lam) - (1 + 1 / nu) * np.log(2) - special.gammaln(1 / nu)
    return c - 0.5 * np.abs(z / lam) ** nu


def logpdf_laplace(z: np.ndarray) -> np.ndarray:
    return -np.log(2.0) - np.abs(z)


_CUSTOM: dict[str, Callable[[np.ndarray], np.ndarray]] = {}


def register_custom_distribution(name: str, fn: Callable[[np.ndarray], np.ndarray]) -> None:
    _CUSTOM[name] = fn


def logpdf(
    z: np.ndarray,
    *,
    name: DistName | str = "gaussian",
    df: float = 8.0,
    skew: float = 0.0,
    ged_nu: float = 1.5,
) -> np.ndarray:
    x = np.asarray(z, dtype=np.float64)
    if name == "gaussian":
        return logpdf_gaussian(x)
    if name == "student_t":
        return logpdf_student_t(x, df=df)
    if name == "skew_t":
        return logpdf_skew_t(x, df=df, skew=skew)
    if name == "ged":
        return logpdf_ged(x, nu=ged_nu)
    if name == "laplace":
        return logpdf_laplace(x)
    if name in _CUSTOM:
        return np.asarray(_CUSTOM[name](x), dtype=np.float64)
    if name == "custom" and "default" in _CUSTOM:
        return np.asarray(_CUSTOM["default"](x), dtype=np.float64)
    return logpdf_gaussian(x)


def distribution_params(name: str, settings: Any) -> dict[str, float]:
    dist = getattr(settings, "distribution", None)
    if dist is None:
        return {"df": 8.0, "skew": 0.0, "ged_nu": 1.5}
    return {"df": float(dist.df), "skew": float(dist.skew), "ged_nu": float(dist.ged_nu)}
