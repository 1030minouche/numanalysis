"""Direct solvers: triangular substitutions, solve, det, inv, Thomas algorithm."""

from collections.abc import Iterable
from typing import Any

from .._scalar import Field, Scalar, field_of
from .._validation import (
    Matrix,
    MatrixLike,
    Vector,
    as_square_matrix,
    as_system,
    read_vector,
)
from ..exceptions import ShapeError, SingularMatrixError, ZeroPivotError
from ._lu import Pivoting, _check_pivot, _factor

__all__ = [
    "back_substitution",
    "det",
    "forward_substitution",
    "inv",
    "solve",
    "solve_tridiagonal",
]


def _forward(L: Matrix, B: Matrix, *, unit_diagonal: bool) -> Matrix:
    """Solve ``L X = B`` row by row, reading only the lower triangle of `L`."""
    X: Matrix = []
    for i in range(len(L)):
        row = list(B[i])
        for k in range(i):
            row = [r - L[i][k] * x for r, x in zip(row, X[k], strict=True)]
        if not unit_diagonal:
            d = L[i][i]
            if d == 0:
                raise SingularMatrixError(f"zero on the diagonal at row {i}")
            row = [r / d for r in row]
        X.append(row)
    return X


def _backward(U: Matrix, B: Matrix, *, unit_diagonal: bool) -> Matrix:
    """Solve ``U X = B`` from the last row up, reading only the upper triangle."""
    n = len(U)
    X: Matrix = [[] for _ in range(n)]
    for i in reversed(range(n)):
        row = list(B[i])
        for k in range(i + 1, n):
            row = [r - U[i][k] * x for r, x in zip(row, X[k], strict=True)]
        if not unit_diagonal:
            d = U[i][i]
            if d == 0:
                raise SingularMatrixError(f"zero on the diagonal at row {i}")
            row = [r / d for r in row]
        X[i] = row
    return X


def _unwrap(X: Matrix, is_vector: bool) -> Any:
    return [row[0] for row in X] if is_vector else X


def forward_substitution(
    L: MatrixLike, b: Iterable[Any], *, unit_diagonal: bool = False
) -> Any:
    """Solve the lower triangular system ``L x = b``.

    Only the lower triangle of `L` is read (with ``unit_diagonal=True``, only the
    part strictly below the diagonal), so the `L` of a packed factorization can
    be passed as is. Complexity: n² flops per right-hand side.

    Parameters
    ----------
    L : sequence of sequences
        Square matrix, lower triangular.
    b : sequence or sequence of sequences
        Right-hand side: a vector, or a matrix whose columns are solved together.
    unit_diagonal : bool, optional
        Assume that the diagonal of `L` is made of ones.

    Returns
    -------
    list
        The solution, with the same shape as `b`.

    Raises
    ------
    ShapeError
        If `L` is not square or `b` does not have as many rows as `L`.
    SingularMatrixError
        If the diagonal of `L` has a zero.

    See Also
    --------
    back_substitution : The upper triangular counterpart.

    Examples
    --------
    >>> forward_substitution([[2, 0], [1, 1]], [4, 3])
    [2.0, 1.0]
    >>> from fractions import Fraction as F
    >>> forward_substitution([[3, 0], [1, 3]], [F(1), F(0)])
    [Fraction(1, 3), Fraction(-1, 9)]
    """
    M, B, _, is_vector = as_system(L, b, name="L")
    return _unwrap(_forward(M, B, unit_diagonal=unit_diagonal), is_vector)


def back_substitution(
    U: MatrixLike, b: Iterable[Any], *, unit_diagonal: bool = False
) -> Any:
    """Solve the upper triangular system ``U x = b``.

    Only the upper triangle of `U` is read (with ``unit_diagonal=True``, only the
    part strictly above the diagonal). Complexity: n² flops per right-hand side.

    Parameters
    ----------
    U : sequence of sequences
        Square matrix, upper triangular.
    b : sequence or sequence of sequences
        Right-hand side: a vector, or a matrix whose columns are solved together.
    unit_diagonal : bool, optional
        Assume that the diagonal of `U` is made of ones.

    Returns
    -------
    list
        The solution, with the same shape as `b`.

    Raises
    ------
    ShapeError
        If `U` is not square or `b` does not have as many rows as `U`.
    SingularMatrixError
        If the diagonal of `U` has a zero.

    See Also
    --------
    forward_substitution : The lower triangular counterpart.

    Examples
    --------
    >>> back_substitution([[2, 1], [0, 4]], [3, 4])
    [1.0, 1.0]
    """
    M, B, _, is_vector = as_system(U, b, name="U")
    return _unwrap(_backward(M, B, unit_diagonal=unit_diagonal), is_vector)


def _lu_solve(M: Matrix, B: Matrix, F: Field, pivot: str) -> Matrix:
    """Solve ``M X = B`` by LU factorization; `M` is overwritten."""
    L, U, perm, col_perm = _factor(M, F, pivot)
    for k in range(len(U)):
        if U[k][k] == 0:
            raise SingularMatrixError(f"A is singular: zero pivot at step {k + 1}")
    Y = _forward(L, [B[p] for p in perm], unit_diagonal=True)
    Z = _backward(U, Y, unit_diagonal=False)
    X: Matrix = [[] for _ in Z]
    for j, row in zip(col_perm, Z, strict=True):  # undo the column permutation
        X[j] = row
    return X


def solve(A: MatrixLike, b: Iterable[Any], *, pivot: Pivoting = "partial") -> Any:
    """Solve the linear system ``A x = b``.

    `A` is factorized as ``P A = L U`` (``P A Q = L U`` with complete pivoting),
    then ``L y = P b`` is solved by forward substitution and ``U x = y`` by back
    substitution. `A` and `b` are computed in one common scalar type, so a
    ``Fraction`` system is solved exactly. Complexity: about 2n³/3 flops, plus
    2n² per right-hand side.

    Parameters
    ----------
    A : sequence of sequences
        Square invertible matrix. Not modified.
    b : sequence or sequence of sequences
        Right-hand side: a vector, or a matrix whose columns are solved together.
    pivot : {"partial", "complete"}, optional
        Pivoting strategy of the LU factorization.

    Returns
    -------
    list
        The solution, with the same shape as `b`.

    Raises
    ------
    ShapeError
        If `A` is not square or `b` does not have as many rows as `A`.
    SingularMatrixError
        If `A` is singular (a zero pivot is met).

    See Also
    --------
    lu : The factorization used here.
    solve_tridiagonal : Linear-time solver for tridiagonal systems.

    Examples
    --------
    >>> solve([[2, 1], [1, 3]], [3, 5])
    [0.8, 1.4]
    >>> from fractions import Fraction as F
    >>> solve([[F(2), 1], [1, 3]], [3, 5])
    [Fraction(4, 5), Fraction(7, 5)]
    """
    _check_pivot(pivot)
    M, B, F, is_vector = as_system(A, b)
    return _unwrap(_lu_solve(M, B, F, pivot), is_vector)


def _permutation_sign(perm: list[int]) -> int:
    """Return +1 for an even permutation, -1 for an odd one (cycle counting)."""
    seen = [False] * len(perm)
    sign = 1
    for start in range(len(perm)):
        if seen[start]:
            continue
        length = 0
        i = start
        while not seen[i]:
            seen[i] = True
            i = perm[i]
            length += 1
        if length % 2 == 0:  # a cycle of even length is an odd permutation
            sign = -sign
    return sign


def det(A: MatrixLike, *, pivot: Pivoting = "partial") -> Scalar:
    """Compute the determinant of a square matrix.

    Computed from ``P A = L U`` (or ``P A Q = L U``) as the product of the
    pivots ``u_kk`` times the signs of the permutations. Exact for ``Fraction``
    matrices. Complexity: about 2n³/3 flops.

    Parameters
    ----------
    A : sequence of sequences
        Square matrix. Not modified.
    pivot : {"partial", "complete"}, optional
        Pivoting strategy of the LU factorization.

    Returns
    -------
    scalar
        ``det(A)``, in the scalar type of `A` (zero if `A` is singular).

    Raises
    ------
    ShapeError
        If `A` is empty, ragged or not square.

    Examples
    --------
    >>> det([[1, 2], [3, 4]])
    -2.0
    >>> from fractions import Fraction as F
    >>> det([[F(1, 2), F(1, 3)], [F(1, 4), F(1, 5)]])
    Fraction(1, 60)
    """
    _check_pivot(pivot)
    M, F = as_square_matrix(A)
    _, U, perm, col_perm = _factor(M, F, pivot)
    d = F.one
    for k in range(len(U)):
        d *= U[k][k]
    if _permutation_sign(perm) * _permutation_sign(col_perm) < 0:
        d = -d
    return d


def inv(A: MatrixLike, *, pivot: Pivoting = "partial") -> Matrix:
    """Compute the inverse of a square matrix.

    Solves ``A X = I`` column by column with one LU factorization. Exact for
    ``Fraction`` matrices. Prefer :func:`solve` to compute ``A⁻¹ b``: it is
    cheaper and more accurate. Complexity: about 2n³ flops.

    Parameters
    ----------
    A : sequence of sequences
        Square invertible matrix. Not modified.
    pivot : {"partial", "complete"}, optional
        Pivoting strategy of the LU factorization.

    Returns
    -------
    Matrix
        ``A⁻¹``.

    Raises
    ------
    ShapeError
        If `A` is empty, ragged or not square.
    SingularMatrixError
        If `A` is singular.

    Examples
    --------
    >>> from fractions import Fraction as F
    >>> inv([[F(2), F(1)], [F(1), F(1)]])
    [[Fraction(1, 1), Fraction(-1, 1)], [Fraction(-1, 1), Fraction(2, 1)]]
    """
    _check_pivot(pivot)
    M, F = as_square_matrix(A)
    n = len(M)
    identity = [[F.one if i == j else F.zero for j in range(n)] for i in range(n)]
    return _lu_solve(M, identity, F, pivot)


def solve_tridiagonal(
    lower: Iterable[Any],
    diag: Iterable[Any],
    upper: Iterable[Any],
    b: Iterable[Any],
) -> Vector:
    """Solve a tridiagonal system with the Thomas algorithm.

    The matrix has `diag` on its diagonal, `lower` just below and `upper` just
    above. Gaussian elimination without pivoting reduces to one forward sweep and
    one backward sweep. It is stable for diagonally dominant or symmetric positive
    definite matrices. Complexity: 8n flops.

    Parameters
    ----------
    lower : sequence
        Sub-diagonal, n - 1 entries.
    diag : sequence
        Diagonal, n entries.
    upper : sequence
        Super-diagonal, n - 1 entries.
    b : sequence
        Right-hand side, n entries.

    Returns
    -------
    Vector
        The solution.

    Raises
    ------
    ShapeError
        If the lengths are inconsistent.
    ZeroPivotError
        If a zero pivot is met (the matrix may still be invertible: use
        :func:`solve`).

    Examples
    --------
    The 1D Poisson matrix ``tridiag(-1, 2, -1)``, in floating point and exactly:

    >>> solve_tridiagonal([-1, -1], [2, 2, 2], [-1, -1], [1, 0, 1])
    [0.9999999999999999, 0.9999999999999998, 0.9999999999999999]
    >>> from fractions import Fraction as F
    >>> solve_tridiagonal([-1, -1], [F(2), 2, 2], [-1, -1], [1, 0, 1])
    [Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)]
    """
    a = read_vector(lower, name="lower", allow_empty=True)
    d = read_vector(diag, name="diag")
    c = read_vector(upper, name="upper", allow_empty=True)
    r = read_vector(b, name="b")
    n = len(d)
    for name, v, size in (("lower", a, n - 1), ("upper", c, n - 1), ("b", r, n)):
        if len(v) != size:
            raise ShapeError(f"{name} must have {size} entries, got {len(v)}")
    F = field_of([*a, *d, *c, *r])
    sub: Vector = [F.convert(x) for x in a]
    dia: Vector = [F.convert(x) for x in d]
    sup: Vector = [F.convert(x) for x in c]
    rhs: Vector = [F.convert(x) for x in r]

    cp: Vector = []  # modified super-diagonal
    rp: Vector = []  # modified right-hand side
    for i in range(n):
        m = dia[i] if i == 0 else dia[i] - sub[i - 1] * cp[i - 1]
        if m == 0:
            raise ZeroPivotError(f"zero pivot at row {i}")
        if i < n - 1:
            cp.append(sup[i] / m)
        rp.append(rhs[i] / m if i == 0 else (rhs[i] - sub[i - 1] * rp[i - 1]) / m)
    x: Vector = [F.zero] * n
    x[n - 1] = rp[n - 1]
    for i in reversed(range(n - 1)):
        x[i] = rp[i] - cp[i] * x[i + 1]
    return x
