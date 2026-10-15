from fractions import Fraction
from typing import Any

import pytest
from hypothesis import given

from linalg_helpers import matmul, max_abs_diff, square_int_matrices, transpose
from numanalysis.exceptions import NotSymmetricError, ShapeError, ZeroPivotError
from numanalysis.linalg._ldlt import ldlt
from numanalysis.linalg._properties import is_positive_definite


def reconstruct(L: Any, D: Any, *, conjugate: bool = False) -> Any:
    LT = transpose(L)
    if conjugate:
        LT = [[x.conjugate() for x in row] for row in LT]
    return matmul(matmul(L, D), LT)


class TestLDLT:
    def test_known_values(self) -> None:
        F = Fraction
        res = ldlt([[F(2), F(1)], [F(1), F(2)]])
        L, D = res
        assert L == [[1, 0], [F(1, 2), 1]]
        assert D == [[2, 0], [0, F(3, 2)]]
        assert res.d == [2, F(3, 2)]

    def test_exact_where_cholesky_is_not(self) -> None:
        A = [[Fraction(x) for x in row] for row in [[2, 1, 0], [1, 2, 1], [0, 1, 2]]]
        L, D = ldlt(A)
        assert reconstruct(L, D) == A
        assert all(type(x) is Fraction for M in (L, D) for row in M for x in row)

    def test_indefinite(self) -> None:
        res = ldlt([[1, 2], [2, 1]])
        assert res.d == [1.0, -3.0]

    def test_zero_pivot(self) -> None:
        with pytest.raises(ZeroPivotError, match="order 1"):
            ldlt([[0, 1], [1, 0]])
        with pytest.raises(ZeroPivotError, match="order 2"):
            ldlt([[1, 1], [1, 1]])

    def test_not_symmetric(self) -> None:
        with pytest.raises(NotSymmetricError, match="symmetric"):
            ldlt([[1, 2], [3, 4]])
        with pytest.raises(NotSymmetricError, match="Hermitian"):
            ldlt([[1, 1j], [1j, 1]])

    def test_invalid_shape(self) -> None:
        with pytest.raises(ShapeError):
            ldlt([[1, 2]])

    def test_complex_hermitian(self) -> None:
        A = [[4, 2 + 2j], [2 - 2j, 6]]
        L, D = ldlt(A)
        assert all(type(x) is complex for row in L for x in row)
        assert D[1][1].imag == 0
        assert max_abs_diff(reconstruct(L, D, conjugate=True), A) < 1e-14

    def test_input_not_mutated(self) -> None:
        A = [[4, 2], [2, 5]]
        ldlt(A)
        assert A == [[4, 2], [2, 5]]

    @given(square_int_matrices(5, 9))
    def test_property_exact_reconstruction(self, B: list[list[int]]) -> None:
        n = len(B)
        # B + Bᵀ + c I is symmetric; skip the rare zero-minor cases.
        A = [
            [Fraction(B[i][j] + B[j][i] + (3 * n if i == j else 0)) for j in range(n)]
            for i in range(n)
        ]
        try:
            res = ldlt(A)
        except ZeroPivotError:
            return
        assert reconstruct(res.L, res.D) == A
        assert is_positive_definite(A) == all(x > 0 for x in res.d)

    def test_scipy_oracle_on_spd(self) -> None:
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(10)
        for n in (1, 4, 8):
            B = rng.standard_normal((n, n))
            A = B @ B.T + n * np.eye(n)
            L, D = ldlt(A)
            C = np.linalg.cholesky(A)  # C = L sqrt(D)
            np.testing.assert_allclose(np.array(L) * np.sqrt(np.diag(D)), C)
