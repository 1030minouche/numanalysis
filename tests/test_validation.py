from fractions import Fraction

import pytest

from numanalysis._scalar import FLOAT, FRACTION
from numanalysis._validation import as_matrix, as_square_matrix, read_matrix
from numanalysis.exceptions import (
    InexactResultError,
    NotPositiveDefiniteError,
    NotSymmetricError,
    NumAnalysisError,
    ShapeError,
    SingularMatrixError,
)


@pytest.mark.parametrize(
    "exc",
    [
        ShapeError,
        SingularMatrixError,
        NotSymmetricError,
        NotPositiveDefiniteError,
        InexactResultError,
    ],
)
def test_exceptions_derive_from_value_error(exc: type[Exception]) -> None:
    assert issubclass(exc, NumAnalysisError)
    assert issubclass(exc, ValueError)


def test_read_matrix_keeps_entries() -> None:
    third = Fraction(1, 3)
    M = read_matrix([(1, third), (2.5, 1j)])
    assert M == [[1, third], [2.5, 1j]]
    assert type(M[0][0]) is int
    assert M[0][1] is third


def test_as_matrix_copies_and_does_not_mutate() -> None:
    A = [[1, 2], [3, 4]]
    M, F = as_matrix(A)
    M[0][0] = 99.0
    assert A == [[1, 2], [3, 4]]
    assert as_matrix(A) == ([[1.0, 2.0], [3.0, 4.0]], FLOAT)
    assert F is FLOAT


def test_as_matrix_accepts_tuples_and_generators() -> None:
    assert as_matrix(((1, 2),))[0] == [[1.0, 2.0]]
    assert as_matrix(iter([iter([1]), iter([2])]))[0] == [[1.0], [2.0]]


def test_as_matrix_accepts_numpy() -> None:
    np = pytest.importorskip("numpy")
    M, F = as_matrix(np.array([[1, 2], [3, 4]]))
    assert M == [[1.0, 2.0], [3.0, 4.0]]
    assert F is FLOAT
    assert all(type(x) is float for row in M for x in row)


def test_as_matrix_unifies_the_field() -> None:
    M, F = as_matrix([[1, Fraction(1, 2)], [0, 1]])
    assert F is FRACTION
    assert all(type(x) is Fraction for row in M for x in row)


def test_as_matrix_explicit_field() -> None:
    M, F = as_matrix([[1, 2]], field=FRACTION)
    assert F is FRACTION
    assert M == [[Fraction(1), Fraction(2)]]
    assert all(type(x) is Fraction for x in M[0])


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
    assert as_square_matrix([[1]]) == ([[1.0]], FLOAT)
    with pytest.raises(ShapeError, match=r"square, got shape \(2, 3\)"):
        as_square_matrix([[1, 2, 3], [4, 5, 6]], name="A")


def test_fraction_subclasses_are_kept() -> None:
    class MyFraction(Fraction):
        pass

    x = MyFraction(1, 2)
    M, _ = as_matrix([[x, 1]])
    assert M[0][0] is x
