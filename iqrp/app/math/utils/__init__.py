"""Numerical utilities."""

from iqrp.app.math.utils.numerical_stability import (
    clip_finite,
    logsumexp,
    protect_overflow,
    safe_divide,
    softplus,
    stable_softmax,
)
from iqrp.app.math.utils.precision import (
    cast,
    is_close,
    machine_eps,
    nextafter,
    relative_error,
    resolve_dtype,
)

__all__ = [
    "cast",
    "clip_finite",
    "is_close",
    "logsumexp",
    "machine_eps",
    "nextafter",
    "protect_overflow",
    "relative_error",
    "resolve_dtype",
    "safe_divide",
    "softplus",
    "stable_softmax",
]
