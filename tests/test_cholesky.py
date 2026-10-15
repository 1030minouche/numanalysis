from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from linalg_helpers import matmul, max_abs, max_abs_diff, transpose
from numanalysis.exceptions import (
    InexactResultError,
    NotPositiveDefiniteError,
    NotSymmetricError,
    ShapeError,
)
from numanalysis.linalg._cholesky import cholesky


def check_cholesky(A: list[list[float]], tol: float = 1e-12) -> list[list[float]]:
    L = cholesky(A).L
    n = len(L)
    assert all(L[i][j] == 0 for i in range(n) for j in range(i + 1, n))
    assert all(L[i][i] > 0 for i in range(n))
    assert max_abs_diff(matmul(L, transpose(L)), A) <= tol * max_abs(A)
    return L


class TestCholesky:
    def test_known_values(self) -> None:
        assert cholesky([[4, 2], [2, 5]]).L == [[2.0, 0.0], [1.0, 2.0]]
        A = [[1, 2, 6, 8], [2, 5, 15, 23], [6, 15, 46, 73], [8, 23, 73, 130]]
        expected = [[1, 0, 0, 0], [2, 1, 0, 0], [6, 3, 1, 0], [8, 7, 4, 1]]
        assert cholesky(A).L == expected
        assert all(type(x) is float for row in cholesky(A).L for x in row)

    def test_unpacking(self) -> None:
        res = cholesky([[9]])
        (L,) = res
        assert L == res.L == [[3.0]]

    def test_input_not_mutated(self) -> None:
        A = [[4, 2], [2, 5]]
        cholesky(A)
        assert A == [[4, 2], [2, 5]]

    @pytest.mark.parametrize(
        ("A", "order"),
        [([[0]], 1), ([[-4]], 1), ([[1, 2], [2, 1]], 2), ([[1, 1], [1, 1]], 2)],
    )
    def test_not_positive_definite(self, A: list[list[int]], order: int) -> None:
        with pytest.raises(NotPositiveDefiniteError, match=f"order {order}"):
            cholesky(A)

    def test_not_symmetric(self) -> None:
        with pytest.raises(NotSymmetricError):
            cholesky([[1, 100], [0, 1]])

    @pytest.mark.parametrize("bad", [[], [[1, 2]], [[1, 2], [3]]])
    def test_invalid_shapes(self, bad: list[list[int]]) -> None:
        with pytest.raises(ShapeError):
            cholesky(bad)

    def test_complex_hermitian(self) -> None:
        A = [[4, 2 + 2j], [2 - 2j, 6]]
        L = cholesky(A).L
        assert all(type(x) is complex for row in L for x in row)
        assert L[0][0] == 2
        assert L[1][1].imag == 0
        assert L[1][1].real > 0
        LH = [[x.conjugate() for x in row] for row in transpose(L)]
        assert max_abs_diff(matmul(L, LH), A) <= 1e-14

    def test_complex_symmetric_is_not_hermitian(self) -> None:
        with pytest.raises(NotSymmetricError, match="Hermitian"):
            cholesky([[2, 1j], [1j, 2]])
        with pytest.raises(NotSymmetricError, match="Hermitian"):
            cholesky([[1j]])  # the diagonal of a Hermitian matrix is real

    def test_complex_not_positive_definite(self) -> None:
        with pytest.raises(NotPositiveDefiniteError, match="order 2"):
            cholesky([[1, 2j], [-2j, 1]])

    def test_numpy_oracle_complex(self) -> None:
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(2)
        for n in (1, 3, 6):
            B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            A = B @ B.conj().T + n * np.eye(n)
            np.testing.assert_allclose(cholesky(A).L, np.linalg.cholesky(A))

    def test_errors_are_value_errors(self) -> None:
        with pytest.raises(ValueError, match="positive definite"):
            cholesky([[-1]])

    def test_fraction_is_exact(self) -> None:
        F = Fraction
        A = [[F(1), F(2), F(6)], [F(2), F(5), F(15)], [F(6), F(15), F(46)]]
        L = cholesky(A).L
        assert L == [[1, 0, 0], [2, 1, 0], [6, 3, 1]]
        assert all(type(x) is Fraction for row in L for x in row)
        assert matmul(L, transpose(L)) == A
        assert cholesky([[F(1, 4)]]).L == [[F(1, 2)]]

    def test_fraction_irrational_root(self) -> None:
        with pytest.raises(InexactResultError, match="irrational"):
            cholesky([[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]])

    def test_mixed_fraction_and_int(self) -> None:
        L = cholesky([[Fraction(4), 2], [2, 5]]).L
        assert all(type(x) is Fraction for row in L for x in row)

    def test_mpmath(self) -> None:
        mpmath = pytest.importorskip("mpmath")
        with mpmath.workdps(40):
            A = [[mpmath.mpf(2), 1], [1, 2]]
            L = cholesky(A).L
            assert all(isinstance(x, mpmath.mpf) for row in L for x in row)
            assert abs(L[0][0] ** 2 - 2) < mpmath.mpf("1e-38")
            assert max_abs_diff(matmul(L, transpose(L)), A) < 1e-38

    def test_decimal(self) -> None:
        with localcontext() as ctx:
            ctx.prec = 50
            A = [[Decimal(2), Decimal(1)], [Decimal(1), Decimal(2)]]
            L = cholesky(A).L
            assert all(type(x) is Decimal for row in L for x in row)
            assert abs(L[0][0] ** 2 - 2) < Decimal("1e-48")
            assert L[0][1] == 0

    @given(
        st.integers(1, 6).flatmap(
            lambda n: st.lists(
                st.lists(st.floats(-10, 10), min_size=n, max_size=n),
                min_size=n,
                max_size=n,
            )
        )
    )
    def test_property_llt_equals_a(self, B: list[list[float]]) -> None:
        n = len(B)
        A = matmul(B, transpose(B))
        for i in range(n):
            A[i][i] += n  # B Bᵀ + n I is symmetric positive definite
        check_cholesky(A, tol=1e-10)

    def test_numpy_oracle(self) -> None:
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(1)
        for n in (1, 3, 8):
            B = rng.standard_normal((n, n))
            A = B @ B.T + n * np.eye(n)
            np.testing.assert_allclose(cholesky(A).L, np.linalg.cholesky(A))
