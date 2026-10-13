from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from linalg_helpers import is_lower_unit, is_upper, matmul, max_abs, max_abs_diff
from numanalysis._validation import promote
from numanalysis.exceptions import ShapeError
from numanalysis.linalg._lu import lu
from numanalysis.linalg._results import LUResult


def check_factorization(A: list[list[float]], tol: float = 1e-12) -> LUResult:
    res = lu(A)
    P, L, U = res
    assert is_lower_unit(L)
    assert is_upper(U)
    assert all(abs(L[i][j]) <= 1 for i in range(len(L)) for j in range(i))
    PA = matmul(P, A)
    assert max_abs_diff(PA, matmul(L, U)) <= tol * max(1.0, max_abs(A))
    assert [[promote(x) for x in A[p]] for p in res.perm] == PA
    return res


class TestLU:
    def test_result_unpacks_and_has_attributes(self) -> None:
        res = lu([[4, 3], [6, 3]])
        P, L, U = res
        assert (P, L, U) == (res.P, res.L, res.U)
        assert res.perm == [1, 0]

    def test_known_values(self) -> None:
        P, L, U = lu([[4, 3], [6, 3]])
        assert P == [[0.0, 1.0], [1.0, 0.0]]
        assert L == [[1.0, 0.0], [pytest.approx(2 / 3), 1.0]]
        assert U == [[6.0, 3.0], [0.0, 1.0]]

    def test_one_by_one(self) -> None:
        assert tuple(lu([[5]])) == ([[1.0]], [[1.0]], [[5.0]])

    def test_three_by_three(self) -> None:
        check_factorization([[2, -1, -2], [-4, 6, 3], [-4, -2, 8]])

    def test_singular_matrix_is_factorized(self) -> None:
        _, _, U = check_factorization([[1, 2, 3], [2, 4, 6], [1, 0, 1]])
        assert any(U[i][i] == 0 for i in range(3))
        _, _, U = check_factorization([[0, 0], [0, 0]])
        assert U == [[0.0, 0.0], [0.0, 0.0]]

    def test_zero_column_is_skipped(self) -> None:
        check_factorization([[0, 1, 2], [0, 3, 4], [0, 5, 7]])

    def test_input_not_mutated_and_tuples_accepted(self) -> None:
        A = [[1, 2], [3, 4]]
        lu(A)
        assert A == [[1, 2], [3, 4]]
        assert lu(((1, 2), (3, 4))).perm == [1, 0]

    @pytest.mark.parametrize("bad", [[], [[1, 2, 3], [4, 5, 6]], [[1, 2], [3]]])
    def test_invalid_shapes(self, bad: list[list[int]]) -> None:
        with pytest.raises(ShapeError):
            lu(bad)

    def test_fraction_is_exact(self) -> None:
        n = 6
        H = [[Fraction(1, i + j + 1) for j in range(n)] for i in range(n)]
        P, L, U = lu(H)
        assert matmul(P, H) == matmul(L, U)
        assert all(type(x) is Fraction for M in (P, L, U) for row in M for x in row)

    def test_decimal(self) -> None:
        A = [[Decimal(2), Decimal(1)], [Decimal(4), Decimal(3)]]
        _, L, U = lu(A)
        assert L[1][0] == Decimal("0.5")
        assert U == [[Decimal(4), Decimal(3)], [Decimal(0), Decimal("-0.5")]]

    def test_complex(self) -> None:
        check_factorization([[1j, 2], [3, 4 - 1j]])  # type: ignore[list-item]

    @given(
        st.integers(1, 6).flatmap(
            lambda n: st.lists(
                st.lists(
                    st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                    min_size=n,
                    max_size=n,
                ),
                min_size=n,
                max_size=n,
            )
        )
    )
    def test_property_pa_equals_lu(self, A: list[list[float]]) -> None:
        check_factorization(A, tol=1e-9)

    @given(
        st.integers(1, 5).flatmap(
            lambda n: st.lists(
                st.lists(st.integers(-20, 20), min_size=n, max_size=n),
                min_size=n,
                max_size=n,
            )
        )
    )
    def test_property_exact_in_fractions(self, A: list[list[int]]) -> None:
        F = [[Fraction(x) for x in row] for row in A]
        P, L, U = lu(F)
        assert matmul(P, F) == matmul(L, U)

    def test_scipy_oracle(self) -> None:
        np = pytest.importorskip("numpy")
        scipy_linalg = pytest.importorskip("scipy.linalg")
        rng = np.random.default_rng(42)
        for n in (1, 2, 5, 10):
            A = rng.standard_normal((n, n))
            P, L, U = lu(A)
            Ps, Ls, Us = scipy_linalg.lu(A)
            # SciPy returns A = Ps L U, so our P is the transpose of Ps.
            np.testing.assert_allclose(np.array(P), Ps.T)
            np.testing.assert_allclose(np.array(L), Ls, atol=1e-12)
            np.testing.assert_allclose(np.array(U), Us, atol=1e-12)
