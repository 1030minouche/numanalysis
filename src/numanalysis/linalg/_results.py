"""Result objects returned by the factorizations.

Each result unpacks like a tuple (``P, L, U = lu(A)``) and keeps named attributes.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from .._validation import Matrix, Vector

__all__ = ["CholeskyResult", "LDLResult", "LUResult"]


@dataclass(frozen=True)
class LUResult:
    """LU factorization with pivoting, ``P A = L U`` or ``P A Q = L U``.

    Attributes
    ----------
    P : Matrix
        Row permutation matrix.
    L : Matrix
        Unit lower triangular matrix.
    U : Matrix
        Upper triangular matrix.
    perm : list[int]
        Row permutation: row ``i`` of ``P A`` is row ``perm[i]`` of ``A``.
    Q : Matrix
        Column permutation matrix (the identity for partial pivoting).
    col_perm : list[int]
        Column permutation: column ``j`` of ``A Q`` is column ``col_perm[j]``
        of ``A``.
    pivot : str
        ``"partial"`` or ``"complete"``. With complete pivoting the result
        unpacks as ``P, L, U, Q``; otherwise as ``P, L, U``.
    """

    P: Matrix
    L: Matrix
    U: Matrix
    perm: list[int]
    Q: Matrix = field(default_factory=list)
    col_perm: list[int] = field(default_factory=list)
    pivot: str = "partial"

    def __iter__(self) -> Iterator[Matrix]:
        """Unpack as ``P, L, U`` (``P, L, U, Q`` with complete pivoting)."""
        if self.pivot == "complete":
            return iter((self.P, self.L, self.U, self.Q))
        return iter((self.P, self.L, self.U))


@dataclass(frozen=True)
class CholeskyResult:
    """Cholesky factorization, ``A = L Lᵀ`` (``A = L Lᴴ`` if complex).

    Attributes
    ----------
    L : Matrix
        Lower triangular matrix with a real positive diagonal.
    """

    L: Matrix

    def __iter__(self) -> Iterator[Matrix]:
        """Unpack as ``(L,)``."""
        return iter((self.L,))


@dataclass(frozen=True)
class LDLResult:
    """Square-root-free Cholesky factorization, ``A = L D Lᵀ`` (``L D Lᴴ``).

    Attributes
    ----------
    L : Matrix
        Unit lower triangular matrix.
    D : Matrix
        Diagonal matrix.
    d : Vector
        The diagonal of `D`.
    """

    L: Matrix
    D: Matrix
    d: Vector

    def __iter__(self) -> Iterator[Matrix]:
        """Unpack as ``L, D``."""
        return iter((self.L, self.D))
