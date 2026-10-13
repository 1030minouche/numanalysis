"""LU factorization with partial pivoting."""

from .._validation import Matrix, MatrixLike, as_square_matrix
from ._results import LUResult

__all__ = ["lu"]


def lu(A: MatrixLike) -> LUResult:
    """Factorize a square matrix as ``P A = L U`` with partial pivoting.

    Gaussian elimination where, at step k, the row with the largest ``|a_ik|``
    (i ≥ k) becomes the pivot row; ties keep the smallest index. The
    factorization is computed once and returned as a whole.

    A singular matrix is factorized too: `U` then has a zero on its diagonal.
    Integer entries are computed as ``float``; ``Fraction`` and ``Decimal``
    entries are kept, so the factorization of a ``Fraction`` matrix is exact.

    Complexity: about 2n³/3 flops.

    Parameters
    ----------
    A : sequence of sequences
        Square matrix (list of lists, tuples, NumPy array...). Not modified.

    Returns
    -------
    LUResult
        Unpacks as ``P, L, U``; also exposes ``perm``, the row permutation.

    Raises
    ------
    ShapeError
        If `A` is empty, ragged or not square.
    TypeError
        If an entry is not a number.

    See Also
    --------
    cholesky : Faster factorization for symmetric positive definite matrices.

    Notes
    -----
    ``scipy.linalg.lu`` uses the other convention, ``A = P L U``: its permutation
    matrix is the transpose of the one returned here.

    Examples
    --------
    >>> P, L, U = lu([[1, 2], [3, 4]])
    >>> P
    [[0.0, 1.0], [1.0, 0.0]]
    >>> L
    [[1.0, 0.0], [0.3333333333333333, 1.0]]
    >>> U
    [[3.0, 4.0], [0.0, 0.6666666666666667]]

    With fractions the result is exact:

    >>> from fractions import Fraction as F
    >>> P, L, U = lu([[F(1), F(2)], [F(3), F(4)]])
    >>> L[1][0], U[1][1]
    (Fraction(1, 3), Fraction(2, 3))
    """
    U = as_square_matrix(A)
    n = len(U)
    zero = U[0][0] * 0
    one = zero + 1
    L: Matrix = [[one if i == j else zero for j in range(n)] for i in range(n)]
    perm = list(range(n))

    for k in range(n - 1):
        p = max(range(k, n), key=lambda i: abs(U[i][k]))
        if p != k:
            U[k], U[p] = U[p], U[k]
            perm[k], perm[p] = perm[p], perm[k]
            # Swap the multipliers already computed (columns < k).
            L[k][:k], L[p][:k] = L[p][:k], L[k][:k]
        pivot = U[k][k]
        if pivot == 0:
            continue  # the whole column is zero: nothing to eliminate
        for i in range(k + 1, n):
            factor = U[i][k] / pivot
            L[i][k] = factor
            U[i][k] = zero
            for j in range(k + 1, n):
                U[i][j] -= factor * U[k][j]

    P: Matrix = [[one if j == perm[i] else zero for j in range(n)] for i in range(n)]
    return LUResult(P=P, L=L, U=U, perm=perm)
