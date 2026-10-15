import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from linalg_helpers import matmul, square_int_matrices
from numanalysis.exceptions import InexactResultError, SingularMatrixError
from numanalysis.linalg._generators import hilbert, random_spd, vandermonde
from numanalysis.linalg._norms import cond, norm
from numanalysis.linalg._properties import is_positive_definite
from numanalysis.linalg._solve import det, inv


class TestVectorNorms:
    def test_values(self) -> None:
        assert norm([3, 4]) == 5.0
        assert norm([3, -4], 1) == 7.0
        assert norm([3, -4], math.inf) == 4.0
        assert norm([3, 4], 2) == 5.0

    def test_exact(self) -> None:
        v = [Fraction(3, 5), Fraction(-4, 5)]
        assert norm(v) == 1
        assert type(norm(v)) is Fraction
        assert norm(v, 1) == Fraction(7, 5)
        with pytest.raises(InexactResultError):
            norm([Fraction(1), Fraction(1)])

    def test_complex_gives_real(self) -> None:
        assert norm([3j, 4]) == 5.0
        assert type(norm([3j, 4])) is float
        assert norm([1 + 1j], 1) == pytest.approx(math.sqrt(2))

    def test_decimal(self) -> None:
        assert norm([Decimal(3), Decimal(4)]) == Decimal(5)

    def test_bad_order(self) -> None:
        with pytest.raises(ValueError, match="vector norm order"):
            norm([1, 2], "fro")

    def test_rejects_strings(self) -> None:
        with pytest.raises(TypeError, match="string"):
            norm("ab")

    def test_numpy_oracle(self) -> None:
        np = pytest.importorskip("numpy")
        v = np.random.default_rng(11).standard_normal(20)
        for ord in (1, 2, math.inf):
            assert norm(v, ord) == pytest.approx(np.linalg.norm(v, ord))


class TestMatrixNorms:
    def test_values(self) -> None:
        A = [[1, -2], [3, 4]]
        assert norm(A, 1) == 6.0
        assert norm(A, math.inf) == 7.0
        assert norm(A) == norm(A, "fro") == math.sqrt(30)

    def test_exact(self) -> None:
        A = [[Fraction(1, 2), Fraction(-1, 3)], [0, 1]]
        assert norm(A, math.inf) == 1
        assert norm(A, 1) == Fraction(4, 3)
        assert norm([[Fraction(3), 0], [0, Fraction(4)]], "fro") == 5

    def test_complex(self) -> None:
        assert norm([[3j, 0], [0, 4]], "fro") == 5.0

    def test_two_norm_not_available(self) -> None:
        with pytest.raises(ValueError, match="SVD"):
            norm([[1, 0], [0, 1]], 2)

    def test_bad_order(self) -> None:
        with pytest.raises(ValueError, match="matrix norm order"):
            norm([[1]], 3)

    @given(square_int_matrices(5, 9), square_int_matrices(5, 9))
    def test_property_submultiplicative(
        self, A: list[list[int]], B: list[list[int]]
    ) -> None:
        n = min(len(A), len(B))
        Fa = [[Fraction(x) for x in row[:n]] for row in A[:n]]
        Fb = [[Fraction(x) for x in row[:n]] for row in B[:n]]
        AB = matmul(Fa, Fb)
        for ord in (1, math.inf):
            assert norm(AB, ord) <= norm(Fa, ord) * norm(Fb, ord)

    @given(st.lists(st.integers(-50, 50), min_size=1, max_size=10))
    def test_property_vector_norm_inequalities(self, v: list[int]) -> None:
        f = [Fraction(x) for x in v]
        n = len(v)
        assert norm(f, math.inf) <= norm(f, 1) <= n * norm(f, math.inf)

    def test_numpy_oracle(self) -> None:
        np = pytest.importorskip("numpy")
        A = np.random.default_rng(12).standard_normal((5, 4))
        for ord in (1, math.inf, "fro"):
            assert norm(A, ord) == pytest.approx(np.linalg.norm(A, ord))


class TestCond:
    @pytest.mark.parametrize(("n", "expected"), [(3, 748), (4, 28375), (5, 943656)])
    def test_hilbert_exact(self, n: int, expected: int) -> None:
        H = hilbert(n, exact=True)
        assert cond(H) == expected
        assert cond(H, math.inf) == expected  # H is symmetric

    def test_identity(self) -> None:
        assert cond([[1, 0], [0, 1]]) == 1.0
        assert cond([[1, 0], [0, 1]], None) == pytest.approx(2.0)

    def test_singular(self) -> None:
        with pytest.raises(SingularMatrixError):
            cond([[1, 2], [2, 4]])

    def test_numpy_oracle(self) -> None:
        np = pytest.importorskip("numpy")
        A = np.random.default_rng(13).standard_normal((6, 6))
        for ord in (1, math.inf, "fro"):
            assert cond(A, ord) == pytest.approx(np.linalg.cond(A, ord))


class TestGenerators:
    def test_hilbert(self) -> None:
        H = hilbert(3, exact=True)
        assert H[2][2] == Fraction(1, 5)
        assert all(type(x) is Fraction for row in H for x in row)
        assert all(type(x) is float for row in hilbert(3) for x in row)

    def test_hilbert_inverse_has_integer_entries(self) -> None:
        Hi = inv(hilbert(6, exact=True))
        assert all(x.denominator == 1 for row in Hi for x in row)
        assert Hi[0][0] == 36

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_bad_size(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            hilbert(bad)
        with pytest.raises(ValueError, match="positive integer"):
            random_spd(bad)

    def test_vandermonde(self) -> None:
        assert vandermonde([1, 2, 3]) == [[1, 1, 1], [4, 2, 1], [9, 3, 1]]
        V = vandermonde([0.5, 2], exact=True)
        assert V == [[Fraction(1, 2), 1], [2, 1]]
        assert all(type(x) is Fraction for row in V for x in row)
        assert vandermonde([0, 1], increasing=True) == [[1, 0], [1, 1]]

    @given(st.lists(st.integers(-6, 6), min_size=1, max_size=6, unique=True))
    def test_property_vandermonde_determinant(self, x: list[int]) -> None:
        V = vandermonde(x, increasing=True, exact=True)
        expected = math.prod(
            x[j] - x[i] for i in range(len(x)) for j in range(i + 1, len(x))
        )
        assert det(V) == expected

    def test_vandermonde_numpy_oracle(self) -> None:
        np = pytest.importorskip("numpy")
        x = [0.5, -1.0, 2.0, 3.0]
        np.testing.assert_allclose(vandermonde(x), np.vander(x))
        np.testing.assert_allclose(
            vandermonde(x, increasing=True), np.vander(x, increasing=True)
        )

    def test_random_spd(self) -> None:
        A = random_spd(5, seed=3, exact=True)
        assert is_positive_definite(A)
        assert all(type(x) is Fraction and x.denominator == 1 for row in A for x in row)
        assert random_spd(5, seed=3) == A
        assert all(type(x) is float for row in random_spd(2, seed=0) for x in row)

    @given(st.integers(1, 7), st.integers(0, 10**6))
    def test_property_random_spd_is_spd(self, n: int, seed: int) -> None:
        assert is_positive_definite(random_spd(n, seed=seed, exact=True))
