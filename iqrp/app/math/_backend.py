"""Optional acceleration backends (Numba / JAX) with NumPy fallback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

_HAS_NUMBA = False
_HAS_JAX = False
_numba: Any = None
_jnp: Any = None

try:  # pragma: no cover - optional dependency
    import numba as _numba_mod  # type: ignore[import-not-found]

    _numba = _numba_mod
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    pass

try:  # pragma: no cover - optional dependency
    import jax.numpy as _jnp_mod  # type: ignore[import-not-found]

    _jnp = _jnp_mod
    _HAS_JAX = True
except ImportError:  # pragma: no cover
    pass


def has_numba() -> bool:
    return _HAS_NUMBA


def has_jax() -> bool:
    return _HAS_JAX


def njit[F: Callable[..., Any]](fn: F | None = None, **kwargs: Any) -> Any:
    """Decorator: Numba njit when available, otherwise identity."""

    def wrap(f: F) -> F:
        if _HAS_NUMBA and _numba is not None:  # pragma: no cover
            return _numba.njit(**kwargs)(f)  # type: ignore[no-any-return]
        return f

    if fn is not None:
        return wrap(fn)
    return wrap


def xp(backend: str = "numpy") -> Any:
    """Return array module for the requested backend."""
    if backend == "jax" and _HAS_JAX and _jnp is not None:  # pragma: no cover
        return _jnp
    return np
