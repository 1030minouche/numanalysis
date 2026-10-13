"""Result objects returned by the factorizations.

Each result unpacks like a tuple (``P, L, U = lu(A)``) and keeps named attributes.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from .._validation import Matrix

__all__ = ["CholeskyResult", "LUResult"]


@dataclass(frozen=True)
class LUResult:
    """LU factorization with partial pivoting, ``P A = L U``.

    Attributes
    ----------
    P : Matrix
        Permutation matrix.
    L : Matrix
        Unit lower triangular matrix.
    U : Matrix
        Upper triangular matrix.
    perm : list[int]
        Row permutation: row ``i`` of ``P A`` is row ``perm[i]`` of ``A``.
    """

    P: Matrix
    L: Matrix
    U: Matrix
    perm: list[int]

    def __iter__(self) -> Iterator[Matrix]:
        """Unpack as ``P, L, U``."""
        return iter((self.P, self.L, self.U))


@dataclass(frozen=True)
class CholeskyResult:
    """Cholesky factorization, ``A = L Lᵀ``.

    Attributes
    ----------
    L : Matrix
        Lower triangular matrix with a positive diagonal.
    """

    L: Matrix

    def __iter__(self) -> Iterator[Matrix]:
        """Unpack as ``(L,)``."""
        return iter((self.L,))
