"""Matrix engine."""

from iqrp.app.math.matrices.decomposition import cholesky, lu, qr, svd
from iqrp.app.math.matrices.eigen import (
    condition_number,
    eig,
    eigh,
    principal_components,
    spectral_radius,
)
from iqrp.app.math.matrices.matrix import (
    det,
    frobenius_norm,
    hadamard,
    inverse,
    is_positive_definite,
    is_symmetric,
    kronecker,
    multiply,
    normalize_rows,
    pseudo_inverse,
    trace,
    transpose,
)
from iqrp.app.math.matrices.sparse import (
    dense,
    identity,
    sparse_add,
    sparse_multiply,
    sparsity,
    to_csc,
    to_csr,
)

__all__ = [
    "cholesky",
    "condition_number",
    "dense",
    "det",
    "eig",
    "eigh",
    "frobenius_norm",
    "hadamard",
    "identity",
    "inverse",
    "is_positive_definite",
    "is_symmetric",
    "kronecker",
    "lu",
    "multiply",
    "normalize_rows",
    "principal_components",
    "pseudo_inverse",
    "qr",
    "sparse_add",
    "sparse_multiply",
    "sparsity",
    "spectral_radius",
    "svd",
    "to_csc",
    "to_csr",
    "trace",
    "transpose",
]
