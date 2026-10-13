from decimal import Decimal
from fractions import Fraction

import pytest

from numanalysis._validation import as_matrix, as_square_matrix, promote
from numanalysis.exceptions import (
    NotPositiveDefiniteError,
    NotSymmetricError,
    NumAnalysisError,
    ShapeError,
    SingularMatrixError,
)


@pytest.mark.parametrize(
    "exc",
    [ShapeError, SingularMatrixError, NotSymmetricError, NotPositiveDefiniteError],
)
def test_exceptions_derive_from_value_error(exc: type[Exception]) -> None:
    assert issubclass(exc, NumAnalysisError)
    assert issubclass(exc, ValueError)


@pytest.mark.parametrize(
    ("value", "expected", "kind"),
    [
        (3, 3.0, float),
        (True, 1.0, float),
        (2.5, 2.5, float),
        (1 + 2j, 1 + 2j, complex),
        (Fraction(1, 3), Fraction(1, 3), Fraction),
        (Decimal("0.1"), Decimal("0.1"), Decimal),
    ],
)
def test_promote(value: object, expected: object, kind: type) -> None:
    result = promote(value)
    assert result == expected
    assert type(result) is kind


def test_promote_rejects_non_numbers() -> None:
    with pytest.raises(TypeError, match="must be numbers"):
        promote("1")


def test_promote_numpy_scalars() -> None:
    np = pytest.importorskip("numpy")
    assert type(promote(np.int64(2))) is float
    assert type(promote(np.float64(2.5))) is float
    assert type(promote(np.float32(0.5))) is float
    assert type(promote(np.complex128(1j))) is complex


def test_as_matrix_copies_and_does_not_mutate() -> None:
    A = [[1, 2], [3, 4]]
    M = as_matrix(A)
    M[0][0] = 99.0
    assert A == [[1, 2], [3, 4]]
    assert as_matrix(A) == [[1.0, 2.0], [3.0, 4.0]]


def test_as_matrix_accepts_tuples_and_generators() -> None:
    assert as_matrix(((1, 2),)) == [[1.0, 2.0]]
    assert as_matrix(iter([iter([1]), iter([2])])) == [[1.0], [2.0]]


def test_as_matrix_accepts_numpy() -> None:
    np = pytest.importorskip("numpy")
    M = as_matrix(np.array([[1, 2], [3, 4]]))
    assert M == [[1.0, 2.0], [3.0, 4.0]]
    assert all(type(x) is float for row in M for x in row)


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ([], "empty"),
        ([[]], "empty rows"),
        ([[1, 2], [3]], "ragged"),
        ([1, 2], "not a sequence"),
        (5, "sequence of sequences"),
        ("ab", "string"),
        (["ab"], "string"),
    ],
)
def test_as_matrix_rejects_bad_shapes(bad: object, match: str) -> None:
    with pytest.raises(ShapeError, match=match):
        as_matrix(bad)  # type: ignore[arg-type]


def test_as_matrix_rejects_non_numeric_entries() -> None:
    with pytest.raises(TypeError, match="must be numbers"):
        as_matrix([[1, None]])


def test_as_square_matrix() -> None:
    assert as_square_matrix([[1]]) == [[1.0]]
    with pytest.raises(ShapeError, match=r"square, got shape \(2, 3\)"):
        as_square_matrix([[1, 2, 3], [4, 5, 6]], name="A")


def test_promote_keeps_fraction_subclasses() -> None:
    class MyFraction(Fraction):
        pass

    x = MyFraction(1, 2)
    assert promote(x) is x
