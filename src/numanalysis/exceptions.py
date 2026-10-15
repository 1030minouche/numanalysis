"""Exceptions raised by numanalysis.

Every exception derives from :class:`NumAnalysisError`, itself a subclass of
:class:`ValueError`, so ``except ValueError`` keeps working.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._steps import Steps

__all__ = [
    "InexactResultError",
    "NotPositiveDefiniteError",
    "NotSymmetricError",
    "NumAnalysisError",
    "ShapeError",
    "SingularMatrixError",
    "ZeroPivotError",
]


class NumAnalysisError(ValueError):
    """Base class for all numanalysis errors.

    Attributes
    ----------
    steps : Steps or None
        When the failing call had ``steps=True``, the steps recorded up to the
        failure (for instance the leading minor that is not positive).
    """

    steps: "Steps | None" = None


class ShapeError(NumAnalysisError):
    """The input does not have the required shape (empty, ragged, not square...)."""


class SingularMatrixError(NumAnalysisError):
    """The matrix is singular, so the requested operation is impossible."""


class ZeroPivotError(NumAnalysisError):
    """Elimination without pivoting met a zero pivot.

    The matrix may still be invertible: a method with pivoting (such as
    :func:`~numanalysis.lu`) can factorize it.
    """


class NotSymmetricError(NumAnalysisError):
    """The matrix is not symmetric."""


class NotPositiveDefiniteError(NumAnalysisError):
    """The matrix is not positive definite."""


class InexactResultError(NumAnalysisError):
    """An exact type cannot hold the result, such as ``sqrt(2)`` as a ``Fraction``."""
