"""Exceptions raised by numanalysis.

Every exception derives from :class:`NumAnalysisError`, itself a subclass of
:class:`ValueError`, so ``except ValueError`` keeps working.
"""

__all__ = [
    "NotPositiveDefiniteError",
    "NotSymmetricError",
    "NumAnalysisError",
    "ShapeError",
    "SingularMatrixError",
]


class NumAnalysisError(ValueError):
    """Base class for all numanalysis errors."""


class ShapeError(NumAnalysisError):
    """The input does not have the required shape (empty, ragged, not square...)."""


class SingularMatrixError(NumAnalysisError):
    """The matrix is singular, so the requested operation is impossible."""


class NotSymmetricError(NumAnalysisError):
    """The matrix is not symmetric."""


class NotPositiveDefiniteError(NumAnalysisError):
    """The matrix is not positive definite."""
