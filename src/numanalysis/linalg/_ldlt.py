"""Square-root-free Cholesky factorization ``A = L D Lᵀ``."""

from .._validation import Matrix, MatrixLike, Vector, as_square_matrix
from ..exceptions import NotSymmetricError, ZeroPivotError
from ._properties import _equals_transpose
from ._results import LDLResult

__all__ = ["ldlt"]


def ldlt(A: MatrixLike, *, rtol: float | None = None) -> LDLResult:
    """Factorize a symmetric matrix as ``A = L D Lᵀ``, without square roots.

    `L` is unit lower triangular and `D` diagonal. Since no square root is taken,
    the factorization of a ``Fraction`` matrix is always exact, and `A` does not
    have to be positive definite: it only needs non-zero leading principal
    minors. `A` is positive definite exactly when every entry of `D` is positive.
    A complex matrix must be Hermitian, and is factorized as ``A = L D Lᴴ`` (with
    a real `D`). No pivoting is done.

    Complexity: about n³/3 flops.

    Parameters
    ----------
    A : sequence of sequences
        Symmetric (or Hermitian) matrix. Not modified.
    rtol : float, optional
        Relative tolerance of the symmetry test (see :func:`is_symmetric`).

    Returns
    -------
    LDLResult
        Unpacks as ``L, D``; also exposes ``d``, the diagonal of `D`.

    Raises
    ------
    ShapeError
        If `A` is empty, ragged or not square.
    NotSymmetricError
        If `A` is not symmetric (not Hermitian, for a complex matrix).
    ZeroPivotError
        If a leading principal minor is zero.

    See Also
    --------
    cholesky : ``A = L Lᵀ``, for positive definite matrices.

    Examples
    --------
    >>> from fractions import Fraction as F
    >>> L, D = ldlt([[F(2), F(1)], [F(1), F(2)]])
    >>> L
    [[Fraction(1, 1), Fraction(0, 1)], [Fraction(1, 2), Fraction(1, 1)]]
    >>> ldlt([[F(2), F(1)], [F(1), F(2)]]).d
    [Fraction(2, 1), Fraction(3, 2)]

    An indefinite matrix is factorized too:

    >>> ldlt([[1, 2], [2, 1]]).d
    [1.0, -3.0]
    """
    M, F = as_square_matrix(A)
    if not _equals_transpose(M, F, conjugate=True, rtol=rtol, atol=0):
        kind = "Hermitian" if F.is_complex else "symmetric"
        raise NotSymmetricError(f"A must be {kind}")
    n = len(M)
    zero, one = F.zero, F.one
    L: Matrix = [[one if i == j else zero for j in range(n)] for i in range(n)]
    d: Vector = []
    for j in range(n):
        s = M[j][j] - sum(
            (L[j][k] * F.conj(L[j][k]) * d[k] for k in range(j)), start=zero
        )
        dj = F.convert(F.real(s))
        if dj == 0:
            raise ZeroPivotError(
                f"the leading principal minor of order {j + 1} is zero"
            )
        d.append(dj)
        for i in range(j + 1, n):
            s = M[i][j] - sum(
                (L[i][k] * F.conj(L[j][k]) * d[k] for k in range(j)), start=zero
            )
            L[i][j] = s / dj
    D: Matrix = [[d[i] if i == j else zero for j in range(n)] for i in range(n)]
    return LDLResult(L=L, D=D, d=d)
