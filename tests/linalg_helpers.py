"""Small dense-matrix helpers used by the tests (independent of the library)."""

from collections.abc import Sequence
from typing import Any

Mat = Sequence[Sequence[Any]]


def matmul(A: Mat, B: Mat) -> list[list[Any]]:
    return [
        [
            sum((A[i][k] * B[k][j] for k in range(len(B))), start=0 * A[0][0])
            for j in range(len(B[0]))
        ]
        for i in range(len(A))
    ]


def transpose(A: Mat) -> list[list[Any]]:
    return [list(col) for col in zip(*A, strict=True)]


def max_abs_diff(A: Mat, B: Mat) -> float:
    return max(
        float(abs(a - b))
        for ra, rb in zip(A, B, strict=True)
        for a, b in zip(ra, rb, strict=True)
    )


def max_abs(A: Mat) -> float:
    return max(float(abs(x)) for row in A for x in row)


def is_lower_unit(L: Mat) -> bool:
    n = len(L)
    return all(L[i][i] == 1 for i in range(n)) and all(
        L[i][j] == 0 for i in range(n) for j in range(i + 1, n)
    )


def is_upper(U: Mat) -> bool:
    n = len(U)
    return all(U[i][j] == 0 for i in range(n) for j in range(i))
