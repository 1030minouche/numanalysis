"""Classic test matrices: Hilbert, Vandermonde, random symmetric positive definite."""

import random
from collections.abc import Iterable
from fractions import Fraction
from typing import Any

from .._scalar import FRACTION, field_of
from .._validation import Matrix, read_vector

__all__ = ["hilbert", "random_spd", "vandermonde"]


def _check_size(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")


def hilbert(n: int, *, exact: bool = False) -> Matrix:
    """Hilbert matrix ``h_ij = 1 / (i + j + 1)`` (indices from 0).

    Symmetric positive definite, and the classic example of an ill-conditioned
    matrix: ``cond(H_n)`` grows like ``e^(3.5 n)``, so in ``float`` arithmetic
    solving ``H x = b`` loses all accuracy around n = 13.

    Parameters
    ----------
    n : int
        Size of the matrix.
    exact : bool, optional
        Return ``Fraction`` entries instead of ``float``.

    Returns
    -------
    Matrix
        The n-by-n Hilbert matrix.

    Examples
    --------
    >>> hilbert(2, exact=True)
    [[Fraction(1, 1), Fraction(1, 2)], [Fraction(1, 2), Fraction(1, 3)]]
    >>> hilbert(2)
    [[1.0, 0.5], [0.5, 0.3333333333333333]]
    """
    _check_size(n)
    return [
        [Fraction(1, i + j + 1) if exact else 1 / (i + j + 1) for j in range(n)]
        for i in range(n)
    ]


def vandermonde(
    x: Iterable[Any], *, increasing: bool = False, exact: bool = False
) -> Matrix:
    """Vandermonde matrix of the nodes `x`: ``v_ij = x_i^(n-1-j)``.

    It is the matrix of polynomial interpolation at the nodes; its determinant
    is ``prod_{i<j} (x_j - x_i)`` (up to sign), and it is badly conditioned for
    many equispaced nodes.

    Parameters
    ----------
    x : sequence
        The n nodes.
    increasing : bool, optional
        Use increasing powers, ``v_ij = x_i^j``, instead of decreasing ones
        (the NumPy default).
    exact : bool, optional
        Convert the nodes to ``Fraction`` (a ``float`` node is converted exactly).
        Otherwise the scalar type of the nodes is kept (integers become
        ``float``).

    Returns
    -------
    Matrix
        The n-by-n Vandermonde matrix.

    Examples
    --------
    >>> vandermonde([1, 2, 3], increasing=True)
    [[1.0, 1.0, 1.0], [1.0, 2.0, 4.0], [1.0, 3.0, 9.0]]
    >>> vandermonde([2, 3], exact=True)
    [[Fraction(2, 1), Fraction(1, 1)], [Fraction(3, 1), Fraction(1, 1)]]
    """
    nodes = read_vector(x, name="x")
    F = FRACTION if exact else field_of(nodes)
    values = [F.convert(v) for v in nodes]
    n = len(values)
    powers = range(n) if increasing else range(n - 1, -1, -1)
    return [[v**k if k else F.one for k in powers] for v in values]


def random_spd(
    n: int, *, seed: int | None = None, exact: bool = False, bound: int = 3
) -> Matrix:
    """Random symmetric positive definite matrix with integer entries.

    Built as ``B Bᵀ + I`` where `B` has random integers in ``[-bound, bound]``,
    so the entries are integers and exact computations stay small.

    Parameters
    ----------
    n : int
        Size of the matrix.
    seed : int, optional
        Seed of the random generator, for reproducible matrices.
    exact : bool, optional
        Return ``Fraction`` entries instead of ``float``.
    bound : int, optional
        Bound of the entries of `B`.

    Returns
    -------
    Matrix
        An n-by-n symmetric positive definite matrix.

    Examples
    --------
    >>> A = random_spd(3, seed=1, exact=True)
    >>> from numanalysis import is_positive_definite
    >>> is_positive_definite(A), A == [list(row) for row in zip(*A)]
    (True, True)
    """
    _check_size(n)
    rng = random.Random(seed)
    B = [[rng.randint(-bound, bound) for _ in range(n)] for _ in range(n)]
    convert = Fraction if exact else float
    return [
        [convert(sum(B[i][k] * B[j][k] for k in range(n)) + (i == j)) for j in range(n)]
        for i in range(n)
    ]
