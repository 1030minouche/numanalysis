"""Result objects returned by the factorizations.

Each result unpacks like a tuple (``P, L, U = lu(A)``) and keeps named attributes.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .._steps import Steps
from .._validation import Matrix, Vector

__all__ = ["CholeskyResult", "LDLResult", "LUResult", "Traced"]


class _ShowsSteps:
    """Jupyter shows the recorded steps, if any, in LaTeX."""

    steps: Steps | None

    def _repr_latex_(self) -> str | None:
        return None if self.steps is None else self.steps._repr_latex_()


@dataclass(frozen=True)
class LUResult(_ShowsSteps):
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
    steps : Steps or None
        The recorded steps, with ``steps=True``.
    """

    P: Matrix
    L: Matrix
    U: Matrix
    perm: list[int]
    Q: Matrix = field(default_factory=list)
    col_perm: list[int] = field(default_factory=list)
    pivot: str = "partial"
    steps: Steps | None = field(default=None, repr=False, compare=False)

    def __iter__(self) -> Iterator[Matrix]:
        """Unpack as ``P, L, U`` (``P, L, U, Q`` with complete pivoting)."""
        if self.pivot == "complete":
            return iter((self.P, self.L, self.U, self.Q))
        return iter((self.P, self.L, self.U))


@dataclass(frozen=True)
class CholeskyResult(_ShowsSteps):
    """Cholesky factorization, ``A = L Lᵀ`` (``A = L Lᴴ`` if complex).

    Attributes
    ----------
    L : Matrix
        Lower triangular matrix with a real positive diagonal.
    steps : Steps or None
        The recorded steps, with ``steps=True``.
    """

    L: Matrix
    steps: Steps | None = field(default=None, repr=False, compare=False)

    def __iter__(self) -> Iterator[Matrix]:
        """Unpack as ``(L,)``."""
        return iter((self.L,))


@dataclass(frozen=True)
class LDLResult(_ShowsSteps):
    """Square-root-free Cholesky factorization, ``A = L D Lᵀ`` (``L D Lᴴ``).

    Attributes
    ----------
    L : Matrix
        Unit lower triangular matrix.
    D : Matrix
        Diagonal matrix.
    d : Vector
        The diagonal of `D`.
    steps : Steps or None
        The recorded steps, with ``steps=True``.
    """

    L: Matrix
    D: Matrix
    d: Vector
    steps: Steps | None = field(default=None, repr=False, compare=False)

    def __iter__(self) -> Iterator[Matrix]:
        """Unpack as ``L, D``."""
        return iter((self.L, self.D))


@dataclass(frozen=True)
class Traced(_ShowsSteps):
    """A value together with the steps that computed it.

    Returned by :func:`solve`, :func:`det`, :func:`inv` and the substitutions when
    they are called with ``steps=True``. Unpacks as ``value, steps``.

    Attributes
    ----------
    value : object
        The result (solution, determinant, inverse...).
    steps : Steps
        The recorded steps.
    """

    value: Any
    steps: Steps

    def __iter__(self) -> Iterator[Any]:
        """Unpack as ``value, steps``."""
        return iter((self.value, self.steps))
