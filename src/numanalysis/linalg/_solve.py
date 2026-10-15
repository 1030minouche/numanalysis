"""Direct solvers: triangular substitutions, solve, det, inv, Thomas algorithm."""

from collections.abc import Iterable
from typing import Any, Literal, overload

from .._scalar import Field, field_of
from .._steps import Determinant, Formula, Header, Note, Steps, VectorSnapshot
from .._validation import (
    Matrix,
    MatrixLike,
    Vector,
    as_square_matrix,
    as_system,
    read_vector,
)
from ..exceptions import ShapeError, SingularMatrixError, ZeroPivotError
from ._lu import Pivoting, _check_pivot, _factor, _record_factors
from ._results import Traced

__all__ = [
    "back_substitution",
    "det",
    "forward_substitution",
    "inv",
    "solve",
    "solve_tridiagonal",
]


def _substitute(
    T: Matrix,
    B: Matrix,
    F: Field,
    *,
    lower: bool,
    unit_diagonal: bool,
    rec: Steps | None = None,
    name: str = "x",
) -> Matrix:
    """Solve ``T X = B`` for a triangular `T`, reading only its triangle.

    Row i is ``x_i = (b_i - sum_k t_ik x_k) / t_ii``, from the first row down when
    `lower`, from the last row up otherwise. Each row is recorded in `rec`.
    """
    n = len(T)
    X: Matrix = [[] for _ in range(n)]
    order = range(n) if lower else reversed(range(n))
    for i in order:
        known = range(i) if lower else range(i + 1, n)
        s = [
            sum((T[i][k] * X[k][c] for k in known), start=F.zero)
            for c in range(len(B[i]))
        ]
        row = [b - v for b, v in zip(B[i], s, strict=True)]
        d = None if unit_diagonal else T[i][i]
        if d is not None:
            if d == 0:
                exc = SingularMatrixError(f"zero on the diagonal at row {i}")
                if rec is not None:
                    rec.add(Note("singular"))
                    exc.steps = rec
                raise exc
            row = [r / d for r in row]
        X[i] = row
        if rec is not None:
            one = len(row) == 1  # show scalars rather than 1-tuples
            rec.add(
                Formula(
                    name,
                    (i,),
                    B[i][0] if one else B[i],
                    (s[0] if one else s) if len(known) else None,
                    d,
                    row[0] if one else row,
                )
            )
    return X


def _unwrap(X: Matrix, is_vector: bool) -> Any:
    return [row[0] for row in X] if is_vector else X


def _record_value(rec: Steps, key: str, name: str, X: Matrix, is_vector: bool) -> None:
    """Append a section `key` showing the vector or matrix `X`."""
    rec.add(Header(key))
    if is_vector:
        rec.add(VectorSnapshot(name, tuple(row[0] for row in X)))
    else:
        rec.snapshot(X, name=name)


def _triangular(
    T: MatrixLike,
    b: Iterable[Any],
    *,
    lower: bool,
    unit_diagonal: bool,
    steps: bool,
    lang: str,
) -> Any:
    M, B, F, is_vector = as_system(T, b, name="L" if lower else "U")
    rec = Steps(lang=lang) if steps else None
    if rec is not None:
        rec.add(Header("forward_plain" if lower else "back_plain"))
    X = _substitute(M, B, F, lower=lower, unit_diagonal=unit_diagonal, rec=rec)
    value = _unwrap(X, is_vector)
    if rec is None:
        return value
    _record_value(rec, "solution", "x" if is_vector else "X", X, is_vector)
    return Traced(value, rec)


def forward_substitution(
    L: MatrixLike,
    b: Iterable[Any],
    *,
    unit_diagonal: bool = False,
    steps: bool = False,
    lang: str = "en",
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
    steps : bool, optional
        Also return the computation of every unknown, as a
        :class:`~numanalysis.linalg.Traced` result.
    lang : {"en", "fr"}, optional
        Default language of the recorded steps.

    Returns
    -------
    list or Traced
        The solution, with the same shape as `b`; with ``steps=True``, a result
        that unpacks as ``x, steps``.

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
    >>> x, steps = forward_substitution([[3, 0], [1, 3]], [F(1), F(0)], steps=True)
    >>> x
    [Fraction(1, 3), Fraction(-1, 9)]
    >>> print(steps)
    Forward substitution: L x = b
      x₁ = 1 / 3 = 1/3
      x₂ = (0 − (1/3)) / 3 = -1/9
    Solution
      x = (1/3, -1/9)
    """
    return _triangular(
        L, b, lower=True, unit_diagonal=unit_diagonal, steps=steps, lang=lang
    )


def back_substitution(
    U: MatrixLike,
    b: Iterable[Any],
    *,
    unit_diagonal: bool = False,
    steps: bool = False,
    lang: str = "en",
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
    steps : bool, optional
        Also return the computation of every unknown, as a
        :class:`~numanalysis.linalg.Traced` result.
    lang : {"en", "fr"}, optional
        Default language of the recorded steps.

    Returns
    -------
    list or Traced
        The solution, with the same shape as `b`; with ``steps=True``, a result
        that unpacks as ``x, steps``.

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
    return _triangular(
        U, b, lower=False, unit_diagonal=unit_diagonal, steps=steps, lang=lang
    )


def _lu_solve(
    M: Matrix,
    B: Matrix,
    F: Field,
    pivot: str,
    rec: Steps | None = None,
    *,
    is_vector: bool = True,
) -> Matrix:
    """Solve ``M X = B`` by LU factorization; `M` is overwritten."""
    L, U, perm, col_perm = _factor(M, F, pivot, rec)
    _record_factors(rec, L=L, U=U)
    for k in range(len(U)):
        if U[k][k] == 0:
            exc = SingularMatrixError(f"A is singular: zero pivot at step {k + 1}")
            if rec is not None:
                rec.add(Note("singular"))
                exc.steps = rec
            raise exc
    PB = [B[p] for p in perm]
    complete = pivot == "complete"
    if rec is not None:
        rec.add(Header("forward"))
        if is_vector:
            rec.add(VectorSnapshot("P b", tuple(row[0] for row in PB), "permuted_rhs"))
        else:
            rec.snapshot(PB, name="P B")
    Y = _substitute(L, PB, F, lower=True, unit_diagonal=True, rec=rec, name="y")
    if rec is not None:
        rec.add(Header("back" if complete else "back_x"))
    Z = _substitute(
        U,
        Y,
        F,
        lower=False,
        unit_diagonal=False,
        rec=rec,
        name="z" if complete else "x",
    )
    X: Matrix = [[] for _ in Z]
    for j, row in zip(col_perm, Z, strict=True):  # undo the column permutation
        X[j] = row
    if rec is not None and complete:
        rec.add(Note("unpermute"))
    return X


def solve(
    A: MatrixLike,
    b: Iterable[Any],
    *,
    pivot: Pivoting = "partial",
    steps: bool = False,
    lang: str = "en",
) -> Any:
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
    steps : bool, optional
        Also return every step (elimination, then both substitutions), as a
        :class:`~numanalysis.linalg.Traced` result.
    lang : {"en", "fr"}, optional
        Default language of the recorded steps.

    Returns
    -------
    list or Traced
        The solution, with the same shape as `b`; with ``steps=True``, a result
        that unpacks as ``x, steps``.

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
    >>> x, steps = solve([[F(2), 1], [1, 3]], [3, 5], steps=True, lang="fr")
    >>> x
    [Fraction(4, 5), Fraction(7, 5)]
    >>> print(steps)
    Élimination de Gauss : L_i ← L_i − (a_ik / a_kk)·L_k sous chaque pivot
    Étape 1
      pivot : le plus grand |a_i1| de la colonne 1 vaut 2, ligne 1
      L₂ ← L₂ − (1/2)·L₁
      Matrice après l'étape 1 :
        [2    1]
        [0  5/2]
    Résultat
      L =
        [  1  0]
        [1/2  1]
      U =
        [2    1]
        [0  5/2]
    Descente : L y = P b
      Second membre permuté : P b = (3, 5)
      y₁ = 3
      y₂ = 5 − (3/2) = 7/2
    Remontée : U x = y
      x₂ = (7/2) / (5/2) = 7/5
      x₁ = (3 − (7/5)) / 2 = 4/5
    Solution
      x = (4/5, 7/5)
    """
    _check_pivot(pivot)
    M, B, F, is_vector = as_system(A, b)
    rec = Steps(lang=lang) if steps else None
    X = _lu_solve(M, B, F, pivot, rec, is_vector=is_vector)
    value = _unwrap(X, is_vector)
    if rec is None:
        return value
    _record_value(rec, "solution", "x" if is_vector else "X", X, is_vector)
    return Traced(value, rec)


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


def det(
    A: MatrixLike,
    *,
    pivot: Pivoting = "partial",
    steps: bool = False,
    lang: str = "en",
) -> Any:
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
    steps : bool, optional
        Also return the elimination steps and the final product, as a
        :class:`~numanalysis.linalg.Traced` result.
    lang : {"en", "fr"}, optional
        Default language of the recorded steps.

    Returns
    -------
    scalar or Traced
        ``det(A)``, in the scalar type of `A` (zero if `A` is singular); with
        ``steps=True``, a result that unpacks as ``det, steps``.

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
    >>> d, steps = det([[F(1), 2], [3, 4]], steps=True)
    >>> print(steps.to_text().splitlines()[-1])
      det A = (−1) · 3 · (2/3) = -2
    """
    _check_pivot(pivot)
    M, F = as_square_matrix(A)
    rec = Steps(lang=lang) if steps else None
    _, U, perm, col_perm = _factor(M, F, pivot, rec)
    pivots = [U[k][k] for k in range(len(U))]
    d = F.one
    for u in pivots:
        d *= u
    sign = _permutation_sign(perm) * _permutation_sign(col_perm)
    if sign < 0:
        d = -d
    if rec is None:
        return d
    rec.add(Header("determinant"))
    rec.add(Determinant(sign, tuple(pivots), d))
    return Traced(d, rec)


@overload
def inv(
    A: MatrixLike,
    *,
    pivot: Pivoting = ...,
    steps: Literal[False] = ...,
    lang: str = ...,
) -> Matrix: ...
@overload
def inv(
    A: MatrixLike, *, pivot: Pivoting = ..., steps: Literal[True], lang: str = ...
) -> Traced: ...
def inv(
    A: MatrixLike,
    *,
    pivot: Pivoting = "partial",
    steps: bool = False,
    lang: str = "en",
) -> Matrix | Traced:
    """Compute the inverse of a square matrix.

    Solves ``A X = I`` with one LU factorization, all columns at once. Exact for
    ``Fraction`` matrices. Prefer :func:`solve` to compute ``A⁻¹ b``: it is
    cheaper and more accurate. Complexity: about 2n³ flops.

    Parameters
    ----------
    A : sequence of sequences
        Square invertible matrix. Not modified.
    pivot : {"partial", "complete"}, optional
        Pivoting strategy of the LU factorization.
    steps : bool, optional
        Also return every step, as a :class:`~numanalysis.linalg.Traced` result.
    lang : {"en", "fr"}, optional
        Default language of the recorded steps.

    Returns
    -------
    Matrix or Traced
        ``A⁻¹``; with ``steps=True``, a result that unpacks as ``inverse, steps``.

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
    rec = Steps(lang=lang) if steps else None
    if rec is not None:
        rec.add(Note("inverse", level=0))
    X = _lu_solve(M, identity, F, pivot, rec, is_vector=False)
    if rec is None:
        return X
    _record_value(rec, "solution", "A⁻¹", X, False)
    return Traced(X, rec)


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
