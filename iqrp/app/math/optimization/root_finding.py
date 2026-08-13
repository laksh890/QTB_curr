"""Root-finding algorithms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scipy import optimize  # type: ignore[import-untyped]


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> dict[str, Any]:
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    lo, hi = a, b
    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or abs(hi - lo) < tol:
            return {"root": mid, "iterations": i + 1, "success": True}
        if fa * fm < 0:
            hi = mid
            fb = fm
        else:
            lo = mid
            fa = fm
    return {"root": 0.5 * (lo + hi), "iterations": max_iter, "success": False}


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> dict[str, Any]:
    a, b = float(x0), float(x1)
    fa, fb = f(a), f(b)
    for i in range(max_iter):
        denom = fb - fa
        if abs(denom) < 1e-15:
            break
        x = b - fb * (b - a) / denom
        if abs(x - b) < tol:
            return {"root": x, "iterations": i + 1, "success": True}
        a, fa = b, fb
        b, fb = x, f(x)
    return {"root": b, "iterations": max_iter, "success": abs(fb) < tol}


def brent(f: Callable[[float], float], a: float, b: float) -> dict[str, Any]:
    root = float(optimize.brentq(f, a, b))
    return {"root": root, "success": True, "iterations": -1}


def find_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    method: str = "brent",
) -> dict[str, Any]:
    if method == "bisection":
        return bisection(f, a, b)
    if method == "secant":
        return secant(f, a, b)
    if method == "brent":
        return brent(f, a, b)
    raise ValueError(f"Unknown root method '{method}'")
