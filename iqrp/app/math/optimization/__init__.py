"""Numerical optimization."""

from iqrp.app.math.optimization.gradient import (
    gradient_descent,
    numerical_gradient,
    projected_gradient_descent,
)
from iqrp.app.math.optimization.numerical import bfgs, golden_search, newton
from iqrp.app.math.optimization.root_finding import (
    bisection,
    brent,
    find_root,
    secant,
)

__all__ = [
    "bfgs",
    "bisection",
    "brent",
    "find_root",
    "golden_search",
    "gradient_descent",
    "newton",
    "numerical_gradient",
    "projected_gradient_descent",
    "secant",
]
