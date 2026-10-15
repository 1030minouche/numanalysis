"""Input normalisation shared by every algorithm.

This module is private. Scalar types are handled by ``_scalar.py``.
"""

from collections.abc import Iterable
from typing import Any, TypeAlias

from ._scalar import Field, Scalar, field_of
from .exceptions import ShapeError

__all__ = ["Matrix", "MatrixLike", "as_matrix", "as_square_matrix", "read_matrix"]

#: The output matrix type: a list of rows.
Matrix: TypeAlias = list[list[Scalar]]
#: Anything accepted as a matrix: lists, tuples, NumPy arrays...
MatrixLike: TypeAlias = Iterable[Iterable[Any]]


def read_matrix(A: MatrixLike, *, name: str = "A") -> list[list[object]]:
    """Copy `A` into a fresh list of lists, checking only its shape.

    The entries are kept as they are; see :func:`as_matrix` to convert them.
    The input is never modified. Complexity: O(mn).

    Raises
    ------
    ShapeError
        If `A` is empty, ragged or not two-dimensional.

    Examples
    --------
    >>> read_matrix(((1, 2), (3, 4)))
    [[1, 2], [3, 4]]
    """
    if isinstance(A, str | bytes):
        raise ShapeError(f"{name} must be a sequence of sequences, got a string")
    try:
        rows = list(A)
    except TypeError:
        raise ShapeError(f"{name} must be a sequence of sequences") from None
    if not rows:
        raise ShapeError(f"{name} must not be empty")
    M: list[list[object]] = []
    for i, row in enumerate(rows):
        if isinstance(row, str | bytes):
            raise ShapeError(f"row {i} of {name} is a string, not a sequence")
        try:
            M.append(list(row))
        except TypeError:
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


def as_matrix(
    A: MatrixLike, *, name: str = "A", field: Field | None = None
) -> tuple[Matrix, Field]:
    """Copy `A` into a fresh matrix whose entries all belong to one field.

    The input is never modified. Complexity: O(mn).

    Parameters
    ----------
    A : sequence of sequences
        A non-empty rectangular matrix (lists, tuples, NumPy array...).
    name : str, optional
        Name used in error messages.
    field : Field, optional
        The field to convert to. By default it is chosen from the entries with
        :func:`numanalysis._scalar.field_of`.

    Returns
    -------
    M : Matrix
        The converted copy.
    field : Field
        The field of its entries.

    Raises
    ------
    ShapeError
        If `A` is empty, ragged or not two-dimensional.
    TypeError
        If an entry is not a number, or the entries cannot be mixed.

    Examples
    --------
    >>> from fractions import Fraction
    >>> as_matrix(((1, 2), (3, 4)))
    ([[1.0, 2.0], [3.0, 4.0]], <numanalysis field float>)
    >>> as_matrix([[1, Fraction(1, 2)]])
    ([[Fraction(1, 1), Fraction(1, 2)]], <numanalysis field Fraction>)
    """
    raw = read_matrix(A, name=name)
    if field is None:
        field = field_of(x for row in raw for x in row)
    return [[field.convert(x) for x in row] for row in raw], field


def as_square_matrix(
    A: MatrixLike, *, name: str = "A", field: Field | None = None
) -> tuple[Matrix, Field]:
    """Like :func:`as_matrix`, but also require a square matrix.

    Examples
    --------
    >>> as_square_matrix([[1, 2, 3]])
    Traceback (most recent call last):
        ...
    numanalysis.exceptions.ShapeError: A must be square, got shape (1, 3)
    """
    M, F = as_matrix(A, name=name, field=field)
    if len(M) != len(M[0]):
        raise ShapeError(f"{name} must be square, got shape ({len(M)}, {len(M[0])})")
    return M, F
