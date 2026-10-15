"""Vector and matrix norms, condition number."""

import math
from collections.abc import Iterable
from typing import Any

from .._scalar import Field, Scalar, field_of
from .._validation import (
    Matrix,
    MatrixLike,
    _is_iterable,
    as_matrix,
    as_square_matrix,
    read_vector,
)
from ._solve import inv

__all__ = ["cond", "norm"]

NormOrder = int | float | str | None


def _abs2(x: Scalar, F: Field) -> Scalar:
    """``|x|²``, exact for ``Fraction`` (no square root involved)."""
    return F.real(x * F.conj(x))


def _vector_norm(v: list[Scalar], F: Field, ord: NormOrder) -> Scalar:
    zero = F.real(F.zero)
    if ord == 1:
        return sum((abs(x) for x in v), start=zero)
    if ord is None or ord == 2:
        return F.real_sqrt(sum((_abs2(x, F) for x in v), start=zero))
    if ord == math.inf:
        return max(abs(x) for x in v)
    raise ValueError(f"vector norm order must be 1, 2 or inf, got {ord!r}")


def _matrix_norm(M: Matrix, F: Field, ord: NormOrder) -> Scalar:
    zero = F.real(F.zero)
    if ord == 1:  # largest column sum
        return max(
            sum((abs(x) for x in col), start=zero) for col in zip(*M, strict=True)
        )
    if ord == math.inf:  # largest row sum
        return max(sum((abs(x) for x in row), start=zero) for row in M)
    if ord is None or ord == "fro":
        return F.real_sqrt(sum((_abs2(x, F) for row in M for x in row), start=zero))
    if ord == 2:
        raise ValueError("the matrix 2-norm needs the SVD, which arrives in v0.4")
    raise ValueError(f"matrix norm order must be 1, inf or 'fro', got {ord!r}")


def norm(x: Iterable[Any], ord: NormOrder = None) -> Scalar:
    """Norm of a vector or of a matrix.

    ========= ========================== ==============================
    ord       vector                     matrix
    ========= ========================== ==============================
    None      2-norm                     Frobenius norm
    1         ``sum |x_i|``              largest column sum of ``|a_ij|``
    2         ``sqrt(sum |x_i|²)``       not available yet (needs the SVD)
    inf       ``max |x_i|``              largest row sum of ``|a_ij|``
    "fro"     —                          ``sqrt(sum |a_ij|²)``
    ========= ========================== ==============================

    The 1- and ∞-norms are exact for ``Fraction`` input. The norms involving a
    square root are exact only when the root is rational, and raise
    :class:`~numanalysis.exceptions.InexactResultError` otherwise. The result is
    real, even for complex input. Complexity: O(n) for a vector, O(n²) for a
    matrix.

    Parameters
    ----------
    x : sequence or sequence of sequences
        A vector or a matrix. Not modified.
    ord : {None, 1, 2, math.inf, "fro"}, optional
        Order of the norm (see the table).

    Returns
    -------
    scalar
        The norm, as a non-negative real.

    Raises
    ------
    ValueError
        If `ord` is not supported for this kind of input.
    InexactResultError
        If the result is irrational and the input is made of ``Fraction``.

    Examples
    --------
    >>> import math
    >>> norm([3, 4])
    5.0
    >>> norm([[1, -2], [3, 4]], 1), norm([[1, -2], [3, 4]], math.inf)
    (6.0, 7.0)
    >>> from fractions import Fraction as F
    >>> norm([[F(1, 2), F(-1, 3)], [0, 1]], math.inf)
    Fraction(1, 1)
    """
    if isinstance(x, str | bytes):
        raise TypeError("norm expects a vector or a matrix, got a string")
    rows = list(x)
    if rows and _is_iterable(rows[0]):
        M, F = as_matrix(rows, name="x")
        return _matrix_norm(M, F, ord)
    v = read_vector(rows, name="x")
    F = field_of(v)
    return _vector_norm([F.convert(e) for e in v], F, ord)


def cond(A: MatrixLike, ord: NormOrder = 1) -> Scalar:
    """Condition number ``‖A‖ ‖A⁻¹‖`` of a square matrix.

    It bounds the amplification of relative errors when solving ``A x = b``: with
    ``float`` arithmetic, expect to lose about ``log10(cond(A))`` significant
    digits. Computed with an explicit inverse, so it is exact for ``Fraction``
    matrices in the 1- and ∞-norms. Complexity: about 2n³ flops.

    Parameters
    ----------
    A : sequence of sequences
        Square invertible matrix. Not modified.
    ord : {1, math.inf, "fro"}, optional
        Matrix norm used (see :func:`norm`); the 1-norm by default.

    Returns
    -------
    scalar
        The condition number (≥ 1 for the 1- and ∞-norms).

    Raises
    ------
    ShapeError
        If `A` is not square.
    SingularMatrixError
        If `A` is singular (its condition number is infinite).

    Examples
    --------
    >>> from fractions import Fraction as F
    >>> cond([[F(1), F(1, 2)], [F(1, 2), F(1, 3)]])
    Fraction(27, 1)
    """
    M, F = as_square_matrix(A)
    if ord is None:
        ord = "fro"
    return _matrix_norm(M, F, ord) * _matrix_norm(inv(M), F, ord)
