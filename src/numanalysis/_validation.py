"""Input normalisation shared by every algorithm.

This module is private. Scalar types are handled by ``_scalar.py``.
"""

from collections.abc import Iterable
from typing import Any, TypeAlias

from ._scalar import Field, Scalar, field_of
from .exceptions import ShapeError

__all__ = [
    "Matrix",
    "MatrixLike",
    "Vector",
    "as_matrix",
    "as_square_matrix",
    "as_system",
    "read_matrix",
    "read_vector",
]

#: The output matrix type: a list of rows.
Matrix: TypeAlias = list[list[Scalar]]
#: The output vector type.
Vector: TypeAlias = list[Scalar]
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
    _check_square(M, name)
    return M, F


def _check_square(M: list[list[Any]], name: str) -> None:
    if len(M) != len(M[0]):
        raise ShapeError(f"{name} must be square, got shape ({len(M)}, {len(M[0])})")


def _is_iterable(x: object) -> bool:
    if isinstance(x, str | bytes):
        return True
    try:
        iter(x)  # type: ignore[call-overload]
    except TypeError:
        return False
    return True


def read_vector(
    b: Iterable[Any], *, name: str = "b", allow_empty: bool = False
) -> list[object]:
    """Copy the one-dimensional sequence `b` into a fresh list, checking its shape.

    Raises
    ------
    ShapeError
        If `b` is empty (unless `allow_empty`), not a sequence, or has
        sequences as entries.

    Examples
    --------
    >>> read_vector((1, 2))
    [1, 2]
    >>> read_vector([[1], [2]])
    Traceback (most recent call last):
        ...
    numanalysis.exceptions.ShapeError: b must be one-dimensional, but entry 0 ...
    """
    if isinstance(b, str | bytes):
        raise ShapeError(f"{name} must be a sequence of numbers, got a string")
    try:
        v = list(b)
    except TypeError:
        raise ShapeError(f"{name} must be a sequence of numbers") from None
    if not v and not allow_empty:
        raise ShapeError(f"{name} must not be empty")
    for i, x in enumerate(v):
        if _is_iterable(x):
            raise ShapeError(
                f"{name} must be one-dimensional, but entry {i} is a sequence"
            )
    return v


def as_system(
    A: MatrixLike, b: Iterable[Any], *, name: str = "A", rhs_name: str = "b"
) -> tuple[Matrix, Matrix, Field, bool]:
    """Read a square system ``A x = b`` whose right-hand side is a vector or a matrix.

    Both operands are converted to one common field.

    Returns
    -------
    M : Matrix
        The converted copy of `A`.
    B : Matrix
        The right-hand side as a matrix with n rows (one column for a vector).
    field : Field
        The common field.
    is_vector : bool
        True if `b` was a vector.

    Raises
    ------
    ShapeError
        If `A` is not square or `b` does not have as many rows as `A`.

    Examples
    --------
    >>> from fractions import Fraction
    >>> as_system([[2, 0], [0, 2]], [Fraction(1, 2), 1])
    ([[Fraction(2, 1), Fraction(0, 1)], [Fraction(0, 1), Fraction(2, 1)]], \
[[Fraction(1, 2)], [Fraction(1, 1)]], <numanalysis field Fraction>, True)
    """
    raw_a = read_matrix(A, name=name)
    _check_square(raw_a, name)
    if isinstance(b, str | bytes):
        raise ShapeError(f"{rhs_name} must be a vector or a matrix, got a string")
    try:
        rows = list(b)
    except TypeError:
        raise ShapeError(f"{rhs_name} must be a vector or a matrix") from None
    is_vector = bool(rows) and not _is_iterable(rows[0])
    if is_vector:
        raw_b = [[x] for x in read_vector(rows, name=rhs_name)]
    else:
        raw_b = read_matrix(rows, name=rhs_name)
    if len(raw_b) != len(raw_a):
        raise ShapeError(
            f"{rhs_name} has {len(raw_b)} rows but {name} has {len(raw_a)}"
        )
    F = field_of(x for M in (raw_a, raw_b) for row in M for x in row)
    M = [[F.convert(x) for x in row] for row in raw_a]
    B = [[F.convert(x) for x in row] for row in raw_b]
    return M, B, F, is_vector
