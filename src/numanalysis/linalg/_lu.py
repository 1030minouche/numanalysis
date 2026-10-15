"""LU factorization with partial or complete pivoting."""

from typing import Literal

from .._scalar import Field
from .._validation import Matrix, MatrixLike, as_square_matrix
from ._results import LUResult

__all__ = ["lu"]

Pivoting = Literal["partial", "complete"]


def _check_pivot(pivot: str) -> None:
    if pivot not in ("partial", "complete"):
        raise ValueError(f"pivot must be 'partial' or 'complete', got {pivot!r}")


def _factor(
    U: Matrix, F: Field, pivot: str
) -> tuple[Matrix, Matrix, list[int], list[int]]:
    """Gaussian elimination in place on `U`; return ``L, U, perm, col_perm``.

    At step k, partial pivoting brings the row with the largest ``|u_ik|``
    (i ≥ k) to position k; complete pivoting brings the largest ``|u_ij|``
    (i, j ≥ k) to position (k, k) by swapping rows and columns. Ties keep the
    smallest index (row first). A zero pivot means the remaining column (or
    submatrix) is zero: the step is skipped and `U` gets a zero on its diagonal.
    """
    n = len(U)
    zero, one = F.zero, F.one
    L: Matrix = [[one if i == j else zero for j in range(n)] for i in range(n)]
    perm = list(range(n))
    col_perm = list(range(n))

    for k in range(n - 1):
        if pivot == "complete":
            p, q = max(
                ((i, j) for i in range(k, n) for j in range(k, n)),
                key=lambda ij: abs(U[ij[0]][ij[1]]),
            )
            if q != k:
                for row in U:
                    row[k], row[q] = row[q], row[k]
                col_perm[k], col_perm[q] = col_perm[q], col_perm[k]
        else:
            p = max(range(k, n), key=lambda i: abs(U[i][k]))
        if p != k:
            U[k], U[p] = U[p], U[k]
            perm[k], perm[p] = perm[p], perm[k]
            # Swap the multipliers already computed (columns < k).
            L[k][:k], L[p][:k] = L[p][:k], L[k][:k]
        pivot_value = U[k][k]
        if pivot_value == 0:
            continue  # nothing left to eliminate in this column
        for i in range(k + 1, n):
            factor = U[i][k] / pivot_value
            L[i][k] = factor
            U[i][k] = zero
            for j in range(k + 1, n):
                U[i][j] -= factor * U[k][j]
    return L, U, perm, col_perm


def _permutation_matrix(perm: list[int], F: Field, *, columns: bool) -> Matrix:
    n = len(perm)
    if columns:  # Q[i][j] = 1 if i == perm[j]
        return [[F.one if i == perm[j] else F.zero for j in range(n)] for i in range(n)]
    return [[F.one if j == perm[i] else F.zero for j in range(n)] for i in range(n)]


def lu(A: MatrixLike, *, pivot: Pivoting = "partial") -> LUResult:
    """Factorize a square matrix as ``P A = L U`` (or ``P A Q = L U``).

    Gaussian elimination where, at step k, the row with the largest ``|a_ik|``
    (i ≥ k) becomes the pivot row; ties keep the smallest index. With
    ``pivot="complete"`` the largest entry of the whole remaining submatrix is
    brought to the pivot position by swapping rows and columns, which gives
    ``P A Q = L U``. The factorization is computed once and returned as a whole.

    A singular matrix is factorized too: `U` then has a zero on its diagonal.
    The computation runs in the scalar type of the entries (``float``,
    ``complex``, ``Fraction``, ``Decimal`` or mpmath); integers alone are computed
    as ``float``, and the factorization of a ``Fraction`` matrix is exact.

    Complexity: about 2n³/3 flops, plus n³/3 comparisons for complete pivoting.

    Parameters
    ----------
    A : sequence of sequences
        Square matrix (list of lists, tuples, NumPy array...). Not modified.
    pivot : {"partial", "complete"}, optional
        Pivoting strategy, partial by default.

    Returns
    -------
    LUResult
        Unpacks as ``P, L, U`` (``P, L, U, Q`` with complete pivoting); also
        exposes ``perm`` and ``col_perm``, the row and column permutations.

    Raises
    ------
    ShapeError
        If `A` is empty, ragged or not square.
    TypeError
        If an entry is not a number, or the entries cannot be mixed.
    ValueError
        If `pivot` is not a known strategy.

    See Also
    --------
    cholesky : Faster factorization for symmetric positive definite matrices.
    solve, det, inv : Built on this factorization.

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

    Complete pivoting also permutes the columns:

    >>> P, L, U, Q = lu([[1, 2], [3, 4]], pivot="complete")
    >>> U
    [[4.0, 3.0], [0.0, -0.5]]
    """
    _check_pivot(pivot)
    M, F = as_square_matrix(A)
    L, U, perm, col_perm = _factor(M, F, pivot)
    return LUResult(
        P=_permutation_matrix(perm, F, columns=False),
        L=L,
        U=U,
        perm=perm,
        Q=_permutation_matrix(col_perm, F, columns=True),
        col_perm=col_perm,
        pivot=pivot,
    )
