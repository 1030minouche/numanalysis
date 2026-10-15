"""Glass-box numerical analysis in pure Python.

Examples
--------
>>> from numanalysis import lu
>>> P, L, U = lu([[0, 1], [1, 0]])
>>> U
[[1.0, 0.0], [0.0, 1.0]]
"""

from . import exceptions, linalg
from ._steps import Steps
from .exceptions import (
    InexactResultError,
    NotPositiveDefiniteError,
    NotSymmetricError,
    NumAnalysisError,
    ShapeError,
    SingularMatrixError,
    ZeroPivotError,
)
from .linalg import (
    CholeskyResult,
    LDLResult,
    LUResult,
    Traced,
    back_substitution,
    cholesky,
    cond,
    det,
    forward_substitution,
    hilbert,
    inv,
    is_hermitian,
    is_positive_definite,
    is_square,
    is_symmetric,
    ldlt,
    lu,
    norm,
    random_spd,
    solve,
    solve_tridiagonal,
    vandermonde,
)

__version__ = "0.4.0.dev0"

__all__ = [
    "CholeskyResult",
    "InexactResultError",
    "LDLResult",
    "LUResult",
    "NotPositiveDefiniteError",
    "NotSymmetricError",
    "NumAnalysisError",
    "ShapeError",
    "SingularMatrixError",
    "Steps",
    "Traced",
    "ZeroPivotError",
    "__version__",
    "back_substitution",
    "cholesky",
    "cond",
    "det",
    "exceptions",
    "forward_substitution",
    "hilbert",
    "inv",
    "is_hermitian",
    "is_positive_definite",
    "is_square",
    "is_symmetric",
    "ldlt",
    "linalg",
    "lu",
    "norm",
    "random_spd",
    "solve",
    "solve_tridiagonal",
    "vandermonde",
]
