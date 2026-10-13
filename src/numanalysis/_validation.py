"""Input normalisation shared by every algorithm.

This module is private. In v0.3 the scalar handling moves to ``_scalar.py``.
"""

from collections.abc import Iterable
from decimal import Decimal
from fractions import Fraction
from numbers import Complex, Integral, Real
from typing import Any, TypeAlias

from .exceptions import ShapeError

__all__ = ["Matrix", "MatrixLike", "Scalar", "as_matrix", "as_square_matrix", "promote"]

#: A scalar: ``float``, ``complex``, ``Fraction`` or ``Decimal``.
Scalar: TypeAlias = Any
#: The output matrix type: a list of rows.
Matrix: TypeAlias = list[list[Scalar]]
#: Anything accepted as a matrix: lists, tuples, NumPy arrays...
MatrixLike: TypeAlias = Iterable[Iterable[Any]]

_KEPT_TYPES = (float, complex, Fraction, Decimal)


def promote(x: object) -> Scalar:
    """Normalise one scalar.

    ``float``, ``complex``, ``Fraction`` and ``Decimal`` are kept as they are;
    integers (including ``bool`` and NumPy integers) become ``float``; NumPy
    floating and complex scalars become their built-in equivalent.

    Parameters
    ----------
    x : object
        The value to normalise.

    Returns
    -------
    Scalar
        The normalised value.

    Raises
    ------
    TypeError
        If `x` is not a number.

    Examples
    --------
    >>> promote(3)
    3.0
    >>> promote(Fraction(1, 3))
    Fraction(1, 3)
    """
    if type(x) in _KEPT_TYPES:
        return x
    if isinstance(x, Integral):
        return float(int(x))
    if isinstance(x, Fraction | Decimal):
        return x
    if isinstance(x, Real):
        return float(x)
    if isinstance(x, Complex):
        return complex(x)
    raise TypeError(f"matrix entries must be numbers, got {type(x).__name__!r}")


def as_matrix(A: MatrixLike, *, name: str = "A") -> Matrix:
    """Copy `A` into a fresh list of lists of normalised scalars.

    The input is never modified. Complexity: O(mn).

    Parameters
    ----------
    A : sequence of sequences
        A non-empty rectangular matrix (lists, tuples, NumPy array...).
    name : str, optional
        Name used in error messages.

    Returns
    -------
    Matrix
        A new matrix whose entries went through :func:`promote`.

    Raises
    ------
    ShapeError
        If `A` is empty, ragged or not two-dimensional.
    TypeError
        If an entry is not a number.

    Examples
    --------
    >>> as_matrix(((1, 2), (3, 4)))
    [[1.0, 2.0], [3.0, 4.0]]
    """
    if isinstance(A, str | bytes):
        raise ShapeError(f"{name} must be a sequence of sequences, got a string")
    try:
        rows = list(A)
    except TypeError:
        raise ShapeError(f"{name} must be a sequence of sequences") from None
    if not rows:
        raise ShapeError(f"{name} must not be empty")
    M: Matrix = []
    for i, row in enumerate(rows):
        if isinstance(row, str | bytes):
            raise ShapeError(f"row {i} of {name} is a string, not a sequence")
        try:
            M.append([promote(x) for x in row])
        except TypeError as exc:
            if "matrix entries" in str(exc):
                raise
            raise ShapeError(f"row {i} of {name} is not a sequence") from None
    ncols = len(M[0])
    if ncols == 0:
        raise ShapeError(f"{name} must not have empty rows")
    for i, row in enumerate(M):
        if len(row) != ncols:
            raise ShapeError(
                f"{name} is ragged: row 0 has {ncols} entries, row {i} has {len(row)}"
            )
    return M


def as_square_matrix(A: MatrixLike, *, name: str = "A") -> Matrix:
    """Like :func:`as_matrix`, but also require a square matrix.

    Examples
    --------
    >>> as_square_matrix([[1, 2, 3]])
    Traceback (most recent call last):
        ...
    numanalysis.exceptions.ShapeError: A must be square, got shape (1, 3)
    """
    M = as_matrix(A, name=name)
    if len(M) != len(M[0]):
        raise ShapeError(f"{name} must be square, got shape ({len(M)}, {len(M[0])})")
    return M
