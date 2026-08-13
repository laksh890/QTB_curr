"""Gradient-based optimization primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from iqrp.app.math._array import as_vector


def numerical_gradient(
    fun: Callable[[np.ndarray], float],
    x: Any,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    x0 = as_vector(x).astype(np.float64)
    g = np.empty_like(x0)
    for i in range(len(x0)):
        e = np.zeros_like(x0)
        e[i] = eps
        g[i] = (fun(x0 + e) - fun(x0 - e)) / (2 * eps)
    return g


def gradient_descent(
    fun: Callable[[np.ndarray], float],
    x0: Any,
    *,
    grad: Callable[[np.ndarray], np.ndarray] | None = None,
    lr: float = 0.1,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> dict[str, Any]:
    x = as_vector(x0).astype(np.float64)
    gfun = grad or (lambda z: numerical_gradient(fun, z))
    history = [float(fun(x))]
    for i in range(max_iter):
        g = gfun(x)
        x_new = x - lr * g
        f_new = float(fun(x_new))
        history.append(f_new)
        if np.linalg.norm(x_new - x) < tol:
            return {
                "x": x_new,
                "fun": f_new,
                "iterations": i + 1,
                "success": True,
                "history": np.asarray(history),
            }
        x = x_new
    return {
        "x": x,
        "fun": float(fun(x)),
        "iterations": max_iter,
        "success": False,
        "history": np.asarray(history),
    }


def projected_gradient_descent(
    fun: Callable[[np.ndarray], float],
    x0: Any,
    project: Callable[[np.ndarray], np.ndarray],
    *,
    lr: float = 0.1,
    max_iter: int = 300,
) -> dict[str, Any]:
    x = project(as_vector(x0).astype(np.float64))
    for _ in range(max_iter):
        g = numerical_gradient(fun, x)
        x = project(x - lr * g)
    return {"x": x, "fun": float(fun(x)), "iterations": max_iter, "success": True}
