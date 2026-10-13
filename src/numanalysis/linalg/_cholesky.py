"""Cholesky factorization of symmetric positive definite matrices."""

import math
from decimal import Decimal

from .._validation import Matrix, MatrixLike, Scalar, as_square_matrix
from ..exceptions import NotPositiveDefiniteError, NotSymmetricError
from ._properties import is_symmetric
from ._results import CholeskyResult

__all__ = ["cholesky"]


def _sqrt(x: Scalar) -> Scalar:
    """Square root of a positive scalar (``Decimal`` stays ``Decimal``)."""
    if isinstance(x, Decimal):
        return x.sqrt()
    return math.sqrt(x)


def cholesky(A: MatrixLike, *, rtol: float | None = None) -> CholeskyResult:
    """Factorize a symmetric positive definite matrix as ``A = L Lᵀ``.

    The Cholesky-Banachiewicz algorithm computes `L` row by row in a single pass;
    a non-positive diagonal term means that `A` is not positive definite.

    Integer and ``Fraction`` entries are computed as ``float`` (exact square
    roots for fractions arrive in v0.3); ``Decimal`` entries stay ``Decimal``.

    Complexity: about n³/3 flops and n square roots.

    Parameters
    ----------
    A : sequence of sequences
        Real symmetric positive definite matrix. Not modified.
    rtol : float, optional
        Relative tolerance of the symmetry test (see :func:`is_symmetric`).

    Returns
    -------
    CholeskyResult
        Exposes ``L``, lower triangular with a positive diagonal; unpacks as
        ``(L,)``.

    Raises
    ------
    ShapeError
        If `A` is empty, ragged or not square.
    NotSymmetricError
        If `A` is not symmetric.
    NotPositiveDefiniteError
        If `A` is not positive definite.
    TypeError
        If `A` has complex or non-numeric entries.

    See Also
    --------
    lu : Factorization of any square matrix.
    is_positive_definite : Test without computing square roots.

    Examples
    --------
    >>> cholesky([[4, 2], [2, 5]]).L
    [[2.0, 0.0], [1.0, 2.0]]
    >>> cholesky([[1, 2], [2, 1]])
    Traceback (most recent call last):
        ...
    numanalysis.exceptions.NotPositiveDefiniteError: A is not positive definite: ...
    """
    M = as_square_matrix(A)
    if any(isinstance(x, complex) for row in M for x in row):
        raise TypeError("cholesky does not support complex matrices yet")
    if not is_symmetric(M, rtol=rtol):
        raise NotSymmetricError("A must be symmetric")
    n = len(M)
    L: Matrix = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            s = M[i][j] - sum((L[i][k] * L[j][k] for k in range(j)), start=0 * M[i][j])
            L[i].append(s / L[j][j])
        d = M[i][i] - sum((x * x for x in L[i]), start=0 * M[i][i])
        if not d > 0:
            raise NotPositiveDefiniteError(
                "A is not positive definite: "
                f"the leading minor of order {i + 1} is not positive"
            )
        L[i].append(_sqrt(d))
    zero = L[0][0] * 0
    for i in range(n):
        L[i].extend([zero] * (n - i - 1))
    return CholeskyResult(L=L)
