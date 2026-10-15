"""Cholesky factorization of Hermitian positive definite matrices."""

from .._validation import Matrix, MatrixLike, as_square_matrix
from ..exceptions import NotPositiveDefiniteError, NotSymmetricError
from ._properties import _equals_transpose
from ._results import CholeskyResult

__all__ = ["cholesky"]


def cholesky(A: MatrixLike, *, rtol: float | None = None) -> CholeskyResult:
    """Factorize a symmetric positive definite matrix as ``A = L Lᵀ``.

    The Cholesky-Banachiewicz algorithm computes `L` row by row in a single pass;
    a non-positive diagonal term means that `A` is not positive definite. A
    complex matrix must be Hermitian, and is factorized as ``A = L Lᴴ``.

    The computation runs in the scalar type of the entries (integers alone are
    computed as ``float``). With ``Fraction`` entries the result is exact, which
    is only possible when every square root is rational; otherwise
    :class:`~numanalysis.exceptions.InexactResultError` is raised.

    Complexity: about n³/3 flops and n square roots.

    Parameters
    ----------
    A : sequence of sequences
        Symmetric (or Hermitian) positive definite matrix. Not modified.
    rtol : float, optional
        Relative tolerance of the symmetry test (see :func:`is_symmetric`).

    Returns
    -------
    CholeskyResult
        Exposes ``L``, lower triangular with a real positive diagonal; unpacks
        as ``(L,)``.

    Raises
    ------
    ShapeError
        If `A` is empty, ragged or not square.
    NotSymmetricError
        If `A` is not symmetric (not Hermitian, for a complex matrix).
    NotPositiveDefiniteError
        If `A` is not positive definite.
    InexactResultError
        If `A` has ``Fraction`` entries and a square root is irrational.
    TypeError
        If an entry is not a number, or the entries cannot be mixed.

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

    Exact when the square roots are rational:

    >>> from fractions import Fraction as F
    >>> cholesky([[F(4), F(2)], [F(2), F(5)]]).L
    [[Fraction(2, 1), Fraction(0, 1)], [Fraction(1, 1), Fraction(2, 1)]]
    """
    M, F = as_square_matrix(A)
    if not _equals_transpose(M, F, conjugate=True, rtol=rtol, atol=0):
        kind = "Hermitian" if F.is_complex else "symmetric"
        raise NotSymmetricError(f"A must be {kind}")
    n = len(M)
    zero = F.zero
    L: Matrix = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            s = M[i][j] - sum((L[i][k] * F.conj(L[j][k]) for k in range(j)), start=zero)
            L[i].append(s / L[j][j])
        d = F.real(M[i][i] - sum((x * F.conj(x) for x in L[i]), start=zero))
        if not d > 0:
            raise NotPositiveDefiniteError(
                "A is not positive definite: "
                f"the leading minor of order {i + 1} is not positive"
            )
        L[i].append(F.convert(F.sqrt(d)))
    for i in range(n):
        L[i].extend([zero] * (n - i - 1))
    return CholeskyResult(L=L)
