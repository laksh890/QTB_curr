"""Classical numerical optimization routines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]

from iqrp.app.math._array import as_vector


def newton(
    f: Callable[[float], float],
    f_prime: Callable[[float], float],
    x0: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> dict[str, Any]:
    x = float(x0)
    for i in range(max_iter):
        df = f_prime(x)
        if abs(df) < 1e-15:
            break
        step = f(x) / df
        x_new = x - step
        if abs(x_new - x) < tol:
            return {"x": x_new, "iterations": i + 1, "success": True}
        x = x_new
    return {"x": x, "iterations": max_iter, "success": abs(f(x)) < tol}


def bfgs(
    fun: Callable[[np.ndarray], float],
    x0: Any,
    *,
    jac: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    x0_arr = as_vector(x0)
    result = optimize.minimize(fun, x0_arr, method="BFGS", jac=jac)
    return {
        "x": np.asarray(result.x, dtype=np.float64),
        "fun": float(result.fun),
        "success": bool(result.success),
        "nit": int(result.nit),
        "message": str(result.message),
    }


def golden_search(
    fun: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> dict[str, Any]:
    phi = (1 + np.sqrt(5)) / 2
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    for i in range(max_iter):
        if abs(b - a) < tol:
            x = 0.5 * (a + b)
            return {"x": x, "fun": float(fun(x)), "iterations": i, "success": True}
        if fun(c) < fun(d):
            b = d
        else:
            a = c
        c = b - (b - a) / phi
        d = a + (b - a) / phi
    x = 0.5 * (a + b)
    return {"x": x, "fun": float(fun(x)), "iterations": max_iter, "success": True}
