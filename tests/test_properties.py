from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from numanalysis.exceptions import ShapeError
from numanalysis.linalg._properties import (
    is_hermitian,
    is_positive_definite,
    is_square,
    is_symmetric,
)


class TestIsSquare:
    def test_square_and_rectangular(self) -> None:
        assert is_square([[1]])
        assert is_square([[1, 2], [3, 4]])
        assert not is_square([[1, 2, 3], [4, 5, 6]])
        assert not is_square([[1], [2]])

    def test_invalid_input(self) -> None:
        with pytest.raises(ShapeError):
            is_square([])
        with pytest.raises(ShapeError):
            is_square([[1, 2], [3]])


class TestIsSymmetric:
    def test_basic(self) -> None:
        assert is_symmetric([[1]])
        assert is_symmetric([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        assert not is_symmetric([[1, 2], [3, 4]])
        assert not is_symmetric([[1, 2, 3], [4, 5, 6]])

    def test_float_tolerance(self) -> None:
        assert is_symmetric([[1.0, 0.1 + 0.2], [0.3, 1.0]])
        assert not is_symmetric([[1.0, 0.1 + 0.2], [0.3, 1.0]], rtol=0)
        assert is_symmetric([[1.0, 1e-9], [0.0, 1.0]], atol=1e-8)

    def test_exact_types_compare_exactly(self) -> None:
        third = Fraction(1, 3)
        assert is_symmetric([[third, third], [third, third]])
        assert not is_symmetric(
            [[Fraction(1), third], [third + Fraction(1, 10**30), Fraction(1)]]
        )
        d = Decimal("0.1")
        assert is_symmetric([[Decimal(1), d], [d, Decimal(2)]])

    def test_complex_uses_plain_transpose(self) -> None:
        assert is_symmetric([[1, 1j], [1j, 1]])
        assert not is_symmetric([[1, 1j], [-1j, 1]])

    def test_decimal_uses_relative_tolerance(self) -> None:
        third = Decimal(1) / 3
        assert is_symmetric([[Decimal(1), third * 3], [Decimal(1), Decimal(1)]])
        assert not is_symmetric(
            [[Decimal(1), third * 3], [Decimal(1), Decimal(1)]], rtol=0
        )

    def test_mpmath(self) -> None:
        mpmath = pytest.importorskip("mpmath")
        x = mpmath.mpf(1) / 3
        assert is_symmetric([[1, x], [x, 1]])
        assert not is_symmetric([[1, x], [x + mpmath.mpf("1e-5"), 1]])

    def test_cannot_mix_decimal_and_float(self) -> None:
        with pytest.raises(TypeError, match="cannot mix"):
            is_symmetric([[Decimal(1), 0.5], [0.5, 1]])


class TestIsHermitian:
    def test_complex(self) -> None:
        assert is_hermitian([[2, 1 + 1j], [1 - 1j, 3]])
        assert not is_hermitian([[2, 1 + 1j], [1 + 1j, 3]])
        assert not is_hermitian([[1j, 0], [0, 1]])  # non-real diagonal
        assert not is_hermitian([[1, 2, 3]])

    def test_tolerance(self) -> None:
        assert is_hermitian([[1 + 1e-17j, 0], [0, 1]])
        assert not is_hermitian([[1 + 1e-17j, 0], [0, 1]], rtol=0)
        assert is_hermitian([[1, 1e-9j], [0, 1]], atol=1e-8)

    def test_real_means_symmetric(self) -> None:
        assert is_hermitian([[2, -1], [-1, 2]])
        assert not is_hermitian([[1, 100], [0, 1]])
        third = Fraction(1, 3)
        assert is_hermitian([[third, 1], [1, third]])


class TestIsPositiveDefinite:
    @pytest.mark.parametrize(
        "A",
        [
            [[1]],
            [[4, 2], [2, 3]],
            [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
            [[1, 2, 6, 8], [2, 5, 15, 23], [6, 15, 46, 73], [8, 23, 73, 130]],
        ],
    )
    def test_positive_definite(self, A: list[list[int]]) -> None:
        assert is_positive_definite(A)

    @pytest.mark.parametrize(
        "A",
        [
            [[0]],
            [[-1]],
            [[1, 2], [2, 1]],  # indefinite
            [[1, 1], [1, 1]],  # semi-definite
            [[0, 1], [1, 0]],
            [[1, 2, 3]],  # not square
        ],
    )
    def test_not_positive_definite(self, A: list[list[int]]) -> None:
        assert not is_positive_definite(A)

    def test_exact_in_fractions(self) -> None:
        # Hilbert matrix: SPD but very ill-conditioned.
        n = 8
        H = [[Fraction(1, i + j + 1) for j in range(n)] for i in range(n)]
        assert is_positive_definite(H)
        H[n - 1][n - 1] -= Fraction(1, 10**8)
        assert not is_positive_definite(H)

    def test_does_not_mutate_input(self) -> None:
        A = [[4.0, 2.0], [2.0, 3.0]]
        is_positive_definite(A)
        assert A == [[4.0, 2.0], [2.0, 3.0]]

    def test_complex_hermitian(self) -> None:
        assert is_positive_definite([[2, 1j], [-1j, 2]])
        assert is_positive_definite([[4, 2 + 2j], [2 - 2j, 6]])
        assert not is_positive_definite([[1, 2j], [-2j, 1]])  # indefinite
        assert not is_positive_definite([[2, 1j], [1j, 2]])  # not Hermitian
        assert not is_positive_definite([[1, 0], [0, 1j]])

    def test_agrees_with_numpy_eigenvalues_complex(self) -> None:
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(3)
        for _ in range(50):
            B = rng.integers(-3, 4, size=(3, 3)) + 1j * rng.integers(-3, 4, size=(3, 3))
            A = B + B.conj().T + rng.integers(0, 6) * np.eye(3)
            expected = bool(np.all(np.linalg.eigvalsh(A) > 1e-9))
            assert is_positive_definite(A) == expected

    def test_mpmath(self) -> None:
        mpmath = pytest.importorskip("mpmath")
        with mpmath.workdps(50):
            n = 12
            H = [[mpmath.mpf(1) / (i + j + 1) for j in range(n)] for i in range(n)]
            assert is_positive_definite(H)

    @given(st.lists(st.integers(-10, 10), min_size=9, max_size=9))
    def test_gram_matrices_plus_identity_are_spd(self, values: list[int]) -> None:
        B = [values[0:3], values[3:6], values[6:9]]
        A = [
            [
                Fraction(sum(B[i][k] * B[j][k] for k in range(3)) + (i == j))
                for j in range(3)
            ]
            for i in range(3)
        ]
        assert is_positive_definite(A)

    def test_agrees_with_numpy_eigenvalues(self) -> None:
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(0)
        for _ in range(50):
            B = rng.integers(-3, 4, size=(4, 4))
            A = B + B.T
            expected = bool(np.all(np.linalg.eigvalsh(A) > 1e-9))
            assert is_positive_definite(A) == expected
