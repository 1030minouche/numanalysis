"""Glass-box numerical analysis in pure Python.

Examples
--------
>>> from numanalysis import lu
>>> P, L, U = lu([[0, 1], [1, 0]])
>>> U
[[1.0, 0.0], [0.0, 1.0]]
"""

from . import exceptions, linalg
from .exceptions import (
    NotPositiveDefiniteError,
    NotSymmetricError,
    NumAnalysisError,
    ShapeError,
    SingularMatrixError,
)
from .linalg import (
    CholeskyResult,
    LUResult,
    cholesky,
    is_positive_definite,
    is_square,
    is_symmetric,
    lu,
)

__version__ = "0.2.0.dev0"

__all__ = [
    "CholeskyResult",
    "LUResult",
    "NotPositiveDefiniteError",
    "NotSymmetricError",
    "NumAnalysisError",
    "ShapeError",
    "SingularMatrixError",
    "__version__",
    "cholesky",
    "exceptions",
    "is_positive_definite",
    "is_square",
    "is_symmetric",
    "linalg",
    "lu",
]
