"""Institutional Probability & Statistical Computing Engine.

Mathematical backbone for every future IQRP algorithm. No financial models.
"""

from iqrp.app.math import matrices, optimization, probability, statistics, stochastic, utils
from iqrp.app.math._array import as_array, as_matrix, as_vector
from iqrp.app.math._backend import has_jax, has_numba, njit, xp

__all__ = [
    "as_array",
    "as_matrix",
    "as_vector",
    "has_jax",
    "has_numba",
    "matrices",
    "njit",
    "optimization",
    "probability",
    "statistics",
    "stochastic",
    "utils",
    "xp",
]
