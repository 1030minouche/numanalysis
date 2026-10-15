from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from linalg_helpers import identity, matmul, max_abs_diff, square_int_matrices
from numanalysis._validation import as_system, read_vector
from numanalysis.exceptions import ShapeError, SingularMatrixError, ZeroPivotError
from numanalysis.linalg._solve import (
    _permutation_sign,
    back_substitution,
    det,
    forward_substitution,
    inv,
    solve,
    solve_tridiagonal,
)


def fractions(A: list[list[int]]) -> list[list[Fraction]]:
    return [[Fraction(x) for x in row] for row in A]


def matvec(A: Any, x: Any) -> list[Any]:
    return [row[0] for row in matmul(A, [[v] for v in x])]


class TestReaders:
    def test_read_vector(self) -> None:
        assert read_vector((1, 2.5)) == [1, 2.5]
        assert read_vector([], allow_empty=True) == []

    @pytest.mark.parametrize(
        ("bad", "match"),
        [([], "empty"), ("ab", "string"), (3, "sequence"), ([[1]], "one-dim")],
    )
    def test_read_vector_errors(self, bad: Any, match: str) -> None:
        with pytest.raises(ShapeError, match=match):
            read_vector(bad)

    def test_read_vector_rejects_string_entries(self) -> None:
        with pytest.raises(ShapeError, match="one-dimensional"):
            read_vector(["ab"])

    def test_as_system_vector_and_matrix(self) -> None:
        M, B, F, is_vector = as_system([[1, 0], [0, 1]], [1, Fraction(1, 2)])
        assert is_vector
        assert F.exact
        assert B == [[1], [Fraction(1, 2)]]
        assert all(type(x) is Fraction for row in M for x in row)
        _, B, _, is_vector = as_system([[1, 0], [0, 1]], [[1, 2], [3, 4]])
        assert not is_vector
        assert B == [[1.0, 2.0], [3.0, 4.0]]

    @pytest.mark.parametrize(
        ("A", "b", "match"),
        [
            ([[1, 2]], [1], "square"),
            ([[1]], [1, 2], "rows"),
            ([[1]], "a", "string"),
            ([[1]], 5, "vector or a matrix"),
            ([[1]], [], "empty"),
        ],
    )
    def test_as_system_errors(self, A: Any, b: Any, match: str) -> None:
        with pytest.raises(ShapeError, match=match):
            as_system(A, b)


class TestSubstitutions:
    def test_forward(self) -> None:
        assert forward_substitution([[2, 0], [1, 1]], [4, 3]) == [2.0, 1.0]

    def test_back(self) -> None:
        assert back_substitution([[2, 1], [0, 4]], [3, 4]) == [1.0, 1.0]

    def test_only_the_triangle_is_read(self) -> None:
        packed = [[2, 99], [1, 1]]
        assert forward_substitution(packed, [4, 3]) == [2.0, 1.0]
        assert back_substitution([[2, 1], [99, 4]], [3, 4]) == [1.0, 1.0]

    def test_unit_diagonal(self) -> None:
        assert forward_substitution([[5, 0], [2, 7]], [1, 3], unit_diagonal=True) == [
            1.0,
            1.0,
        ]
        assert back_substitution([[5, 2], [0, 7]], [3, 1], unit_diagonal=True) == [
            1.0,
            1.0,
        ]

    def test_matrix_right_hand_side(self) -> None:
        X = forward_substitution([[1, 0], [1, 1]], [[1, 2], [3, 4]])
        assert X == [[1.0, 2.0], [2.0, 2.0]]

    def test_zero_diagonal(self) -> None:
        with pytest.raises(SingularMatrixError, match="row 1"):
            forward_substitution([[1, 0], [1, 0]], [1, 1])
        with pytest.raises(SingularMatrixError, match="row 0"):
            back_substitution([[0, 1], [0, 1]], [1, 1])

    def test_exact(self) -> None:
        L: list[list[Any]] = [[Fraction(3), 0], [1, 3]]
        assert forward_substitution(L, [1, 0]) == [Fraction(1, 3), Fraction(-1, 9)]

    def test_scipy_oracle(self) -> None:
        np = pytest.importorskip("numpy")
        scipy_linalg = pytest.importorskip("scipy.linalg")
        rng = np.random.default_rng(5)
        for n in (1, 4, 9):
            T = rng.standard_normal((n, n)) + n * np.eye(n)
            b = rng.standard_normal(n)
            np.testing.assert_allclose(
                forward_substitution(T, b),
                scipy_linalg.solve_triangular(T, b, lower=True),
            )
            np.testing.assert_allclose(
                back_substitution(T, b),
                scipy_linalg.solve_triangular(T, b, lower=False),
            )


class TestSolve:
    def test_known_values(self) -> None:
        assert solve([[2, 1], [1, 3]], [3, 5]) == pytest.approx([0.8, 1.4])
        assert solve([[0, 1], [1, 0]], [2, 3]) == [3.0, 2.0]

    def test_exact(self) -> None:
        x = solve([[Fraction(2), 1], [1, 3]], [3, 5])
        assert x == [Fraction(4, 5), Fraction(7, 5)]
        assert all(type(v) is Fraction for v in x)

    def test_fraction_right_hand_side_makes_it_exact(self) -> None:
        x = solve([[2, 1], [1, 3]], [Fraction(3), 5])
        assert x == [Fraction(4, 5), Fraction(7, 5)]

    def test_matrix_right_hand_side(self) -> None:
        A = fractions([[2, 1], [1, 3]])
        X = solve(A, [[3, 1], [5, 0]])
        assert matmul(A, X) == [[3, 1], [5, 0]]

    @pytest.mark.parametrize("pivot", ["partial", "complete"])
    def test_singular(self, pivot: Any) -> None:
        with pytest.raises(SingularMatrixError, match="singular"):
            solve([[1, 2], [2, 4]], [1, 2], pivot=pivot)

    def test_complete_pivoting(self) -> None:
        A = fractions([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        b = [1, 2, 3]
        assert solve(A, b, pivot="complete") == solve(A, b)

    def test_complex_and_decimal(self) -> None:
        x = solve([[1j, 1], [1, 1j]], [1, 1])
        assert all(type(v) is complex for v in x)
        assert max_abs_diff([matvec([[1j, 1], [1, 1j]], x)], [[1, 1]]) < 1e-15
        y = solve([[Decimal(4), Decimal(1)], [Decimal(1), Decimal(3)]], [1, 2])
        assert all(type(v) is Decimal for v in y)

    def test_mpmath(self) -> None:
        mpmath = pytest.importorskip("mpmath")
        with mpmath.workdps(50):
            n = 10
            H = [[mpmath.mpf(1) / (i + j + 1) for j in range(n)] for i in range(n)]
            b = [sum(row) for row in H]  # exact solution: all ones
            x = solve(H, b)
            assert max(abs(v - 1) for v in x) < mpmath.mpf("1e-30")

    def test_does_not_mutate(self) -> None:
        A = [[2.0, 1.0], [1.0, 3.0]]
        b = [3.0, 5.0]
        solve(A, b)
        assert A == [[2.0, 1.0], [1.0, 3.0]]
        assert b == [3.0, 5.0]

    @given(square_int_matrices(), st.lists(st.integers(-9, 9), min_size=5, max_size=5))
    def test_property_exact_residual(
        self, A: list[list[int]], values: list[int]
    ) -> None:
        Fr = fractions(A)
        assume(det(Fr) != 0)
        b = [Fraction(v) for v in values[: len(A)]]
        for pivot in ("partial", "complete"):
            x = solve(Fr, b, pivot=pivot)
            assert matvec(Fr, x) == b

    def test_scipy_oracle(self) -> None:
        np = pytest.importorskip("numpy")
        scipy_linalg = pytest.importorskip("scipy.linalg")
        rng = np.random.default_rng(6)
        for n in (1, 3, 10):
            A = rng.standard_normal((n, n))
            B = rng.standard_normal((n, 2))
            np.testing.assert_allclose(solve(A, B), scipy_linalg.solve(A, B))
            np.testing.assert_allclose(
                solve(A, B[:, 0], pivot="complete"), scipy_linalg.solve(A, B[:, 0])
            )


class TestDet:
    @pytest.mark.parametrize(
        ("perm", "sign"),
        [([0], 1), ([1, 0], -1), ([1, 2, 0], 1), ([2, 1, 0], -1), ([1, 0, 3, 2], 1)],
    )
    def test_permutation_sign(self, perm: list[int], sign: int) -> None:
        assert _permutation_sign(perm) == sign

    def test_known_values(self) -> None:
        assert det([[1, 2], [3, 4]]) == -2.0
        assert det([[1, 2], [3, 4]], pivot="complete") == -2.0
        assert det([[5]]) == 5.0
        assert det([[1, 2], [2, 4]]) == 0
        assert det([[0, 1], [1, 0]]) == -1.0

    def test_exact_hilbert(self) -> None:
        # det(H_4) = 1 / 6048000
        H = [[Fraction(1, i + j + 1) for j in range(4)] for i in range(4)]
        assert det(H) == Fraction(1, 6048000)
        assert det(H, pivot="complete") == Fraction(1, 6048000)

    @given(square_int_matrices(4, 9), square_int_matrices(4, 9))
    def test_property_multiplicative(
        self, A: list[list[int]], B: list[list[int]]
    ) -> None:
        n = min(len(A), len(B))
        A = [row[:n] for row in A[:n]]
        B = [row[:n] for row in B[:n]]
        Fa, Fb = fractions(A), fractions(B)
        assert det(matmul(Fa, Fb)) == det(Fa) * det(Fb)
        assert det(Fa) == det(Fa, pivot="complete")

    @given(square_int_matrices(5, 9))
    def test_property_integer_matrix_has_integer_det(self, A: list[list[int]]) -> None:
        d = det(fractions(A))
        assert d.denominator == 1

    def test_numpy_oracle(self) -> None:
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(7)
        for n in (1, 2, 6):
            A = rng.standard_normal((n, n))
            assert det(A) == pytest.approx(np.linalg.det(A))
            assert det(A, pivot="complete") == pytest.approx(np.linalg.det(A))


class TestInv:
    def test_exact(self) -> None:
        A = fractions([[2, 1], [1, 1]])
        assert inv(A) == [[1, -1], [-1, 2]]

    def test_singular(self) -> None:
        with pytest.raises(SingularMatrixError):
            inv([[1, 2], [2, 4]])

    @given(square_int_matrices(4, 9))
    def test_property_exact_inverse(self, A: list[list[int]]) -> None:
        Fr = fractions(A)
        assume(det(Fr) != 0)
        assert matmul(inv(Fr), Fr) == identity(len(A))
        assert inv(Fr, pivot="complete") == inv(Fr)

    def test_numpy_oracle(self) -> None:
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(8)
        A = rng.standard_normal((6, 6))
        np.testing.assert_allclose(inv(A), np.linalg.inv(A))


class TestTridiagonal:
    def test_poisson_exact(self) -> None:
        n = 6
        x = solve_tridiagonal(
            [-1] * (n - 1), [Fraction(2)] * n, [-1] * (n - 1), [1] + [0] * (n - 2) + [1]
        )
        assert x == [1] * n
        assert all(type(v) is Fraction for v in x)

    def test_size_one(self) -> None:
        assert solve_tridiagonal([], [4], [], [2]) == [0.5]

    def test_agrees_with_solve(self) -> None:
        a, d, c, b = [1, 2], [Fraction(5), 6, 7], [3, 4], [1, 2, 3]
        A = [[5, 3, 0], [1, 6, 4], [0, 2, 7]]
        assert solve_tridiagonal(a, d, c, b) == solve(fractions(A), b)

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            (([1], [1, 2, 3], [1, 1], [1, 1, 1]), "lower must have 2"),
            (([1, 1], [1, 2, 3], [1], [1, 1, 1]), "upper must have 2"),
            (([1, 1], [1, 2, 3], [1, 1], [1, 1]), "b must have 3"),
        ],
    )
    def test_bad_lengths(self, args: Any, match: str) -> None:
        with pytest.raises(ShapeError, match=match):
            solve_tridiagonal(*args)

    def test_zero_pivot(self) -> None:
        with pytest.raises(ZeroPivotError, match="row 0"):
            solve_tridiagonal([1], [0, 1], [1], [1, 1])
        with pytest.raises(ZeroPivotError, match="row 1"):
            solve_tridiagonal([1], [1, 1], [1], [1, 1])

    def test_scipy_oracle(self) -> None:
        np = pytest.importorskip("numpy")
        scipy_linalg = pytest.importorskip("scipy.linalg")
        rng = np.random.default_rng(9)
        n = 50
        a, c = rng.standard_normal(n - 1), rng.standard_normal(n - 1)
        d = 4 + rng.random(n)  # diagonally dominant
        b = rng.standard_normal(n)
        banded = np.zeros((3, n))
        banded[0, 1:], banded[1], banded[2, :-1] = c, d, a
        np.testing.assert_allclose(
            solve_tridiagonal(a, d, c, b), scipy_linalg.solve_banded((1, 1), banded, b)
        )
