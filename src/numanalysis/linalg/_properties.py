"""Structural properties of matrices: squareness, symmetry, positive definiteness."""

from decimal import Decimal
from fractions import Fraction

from .._validation import Matrix, MatrixLike, Scalar, as_matrix

__all__ = ["is_positive_definite", "is_square", "is_symmetric"]

_DEFAULT_RTOL = 1e-12


def _is_exact(M: Matrix) -> bool:
    """Return True if every entry has an exact type (``Fraction`` or ``Decimal``)."""
    return all(isinstance(x, Fraction | Decimal) for row in M for x in row)


def _close(a: Scalar, b: Scalar, rtol: float, atol: float) -> bool:
    """Compare two scalars, exactly when both tolerances are zero."""
    if rtol == 0 and atol == 0:
        return bool(a == b)
    return bool(abs(a - b) <= atol + rtol * max(abs(a), abs(b)))


def is_square(A: MatrixLike) -> bool:
    """Tell whether `A` is a square matrix.

    Complexity: O(n²) (the input is validated).

    Parameters
    ----------
    A : sequence of sequences
        A non-empty rectangular matrix.

    Returns
    -------
    bool
        True if `A` has as many rows as columns.

    Raises
    ------
    ShapeError
        If `A` is empty or ragged.

    Examples
    --------
    >>> is_square([[1, 2], [3, 4]])
    True
    >>> is_square([[1, 2, 3], [4, 5, 6]])
    False
    """
    M = as_matrix(A)
    return len(M) == len(M[0])


def is_symmetric(
    A: MatrixLike, *, rtol: float | None = None, atol: float = 0.0
) -> bool:
    """Tell whether `A` equals its transpose.

    Entries are compared with ``|a_ij - a_ji| <= atol + rtol * max(|a_ij|, |a_ji|)``.
    Complex matrices are compared with the plain transpose, not the conjugate one.
    Complexity: O(n²).

    Parameters
    ----------
    A : sequence of sequences
        A non-empty rectangular matrix.
    rtol : float, optional
        Relative tolerance. Defaults to 0 (exact comparison) when every entry is a
        ``Fraction`` or a ``Decimal``, and to 1e-12 otherwise.
    atol : float, optional
        Absolute tolerance, 0 by default.

    Returns
    -------
    bool
        True if `A` is square and symmetric.

    Raises
    ------
    ShapeError
        If `A` is empty or ragged.

    Examples
    --------
    >>> is_symmetric([[2, 1], [1, 3]])
    True
    >>> is_symmetric([[1, 100], [0, 1]])
    False
    >>> is_symmetric([[1, 2, 3]])
    False
    """
    M = as_matrix(A)
    n = len(M)
    if n != len(M[0]):
        return False
    if rtol is None:
        rtol = 0.0 if _is_exact(M) else _DEFAULT_RTOL
    return all(
        _close(M[i][j], M[j][i], rtol, atol) for i in range(n) for j in range(i + 1, n)
    )


def is_positive_definite(A: MatrixLike, *, rtol: float | None = None) -> bool:
    """Tell whether `A` is symmetric positive definite.

    The matrix must be symmetric (checked with :func:`is_symmetric`) and every
    pivot of Gaussian elimination without pivoting must be strictly positive,
    which is equivalent to xᵀAx > 0 for every x ≠ 0. No square root is taken,
    so the test is exact for ``Fraction`` and ``Decimal`` matrices.
    Complexity: about n³/3 flops.

    Parameters
    ----------
    A : sequence of sequences
        A non-empty rectangular matrix with real entries.
    rtol : float, optional
        Relative tolerance of the symmetry test (see :func:`is_symmetric`).

    Returns
    -------
    bool
        True if `A` is square, symmetric and positive definite.

    Raises
    ------
    ShapeError
        If `A` is empty or ragged.
    TypeError
        If `A` has complex entries.

    Examples
    --------
    >>> is_positive_definite([[4, 2], [2, 3]])
    True
    >>> is_positive_definite([[1, 2], [2, 1]])
    False
    >>> is_positive_definite([[1, 100], [0, 1]])  # not symmetric
    False
    """
    M = as_matrix(A)
    if any(isinstance(x, complex) for row in M for x in row):
        raise TypeError("is_positive_definite does not support complex matrices yet")
    if not is_symmetric(M, rtol=rtol):
        return False
    n = len(M)
    for k in range(n):
        pivot = M[k][k]
        if not pivot > 0:
            return False
        # Update the upper triangle of the trailing submatrix (it stays symmetric).
        for i in range(k + 1, n):
            factor = M[k][i] / pivot
            for j in range(i, n):
                M[i][j] -= factor * M[k][j]
    return True
