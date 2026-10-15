"""Structural properties of matrices: squareness, symmetry, positive definiteness."""

from .._scalar import Field
from .._validation import Matrix, MatrixLike, as_matrix

__all__ = ["is_hermitian", "is_positive_definite", "is_square", "is_symmetric"]


def is_square(A: MatrixLike) -> bool:
    """Tell whether `A` is a square matrix.

    Complexity: O(n²) (the input is validated).

    Parameters
    ----------
    A : sequence of sequences
        A non-empty rectangular matrix.

    Returns
    -------
    bool
        True if `A` has as many rows as columns.

    Raises
    ------
    ShapeError
        If `A` is empty or ragged.

    Examples
    --------
    >>> is_square([[1, 2], [3, 4]])
    True
    >>> is_square([[1, 2, 3], [4, 5, 6]])
    False
    """
    M, _ = as_matrix(A)
    return len(M) == len(M[0])


def _equals_transpose(
    M: Matrix, F: Field, *, conjugate: bool, rtol: object, atol: object
) -> bool:
    n = len(M)
    if n != len(M[0]):
        return False
    first = 0 if conjugate else 1  # the diagonal of a Hermitian matrix is real
    return all(
        F.isclose(M[i][j], F.conj(M[j][i]) if conjugate else M[j][i], rtol, atol)
        for i in range(n)
        for j in range(i + first, n)
    )


def is_symmetric(
    A: MatrixLike, *, rtol: float | None = None, atol: float = 0.0
) -> bool:
    """Tell whether `A` equals its transpose.

    Entries are compared with ``|a_ij - a_ji| <= atol + rtol * max(|a_ij|, |a_ji|)``.
    Complex matrices are compared with the plain transpose; see
    :func:`is_hermitian` for the conjugate transpose. Complexity: O(n²).

    Parameters
    ----------
    A : sequence of sequences
        A non-empty rectangular matrix.
    rtol : float, optional
        Relative tolerance. Defaults to 0 (exact comparison) for ``Fraction``
        matrices, and to 4096 machine epsilons otherwise (about 9.1e-13 for
        ``float``).
    atol : float, optional
        Absolute tolerance, 0 by default.

    Returns
    -------
    bool
        True if `A` is square and symmetric.

    Raises
    ------
    ShapeError
        If `A` is empty or ragged.

    Examples
    --------
    >>> is_symmetric([[2, 1], [1, 3]])
    True
    >>> is_symmetric([[1, 100], [0, 1]])
    False
    >>> is_symmetric([[1, 2, 3]])
    False
    """
    M, F = as_matrix(A)
    return _equals_transpose(M, F, conjugate=False, rtol=rtol, atol=atol)


def is_hermitian(
    A: MatrixLike, *, rtol: float | None = None, atol: float = 0.0
) -> bool:
    """Tell whether `A` equals its conjugate transpose.

    Same tolerances as :func:`is_symmetric`; the diagonal must also be real.
    For a real matrix this is the same test as :func:`is_symmetric`.
    Complexity: O(n²).

    Parameters
    ----------
    A : sequence of sequences
        A non-empty rectangular matrix.
    rtol : float, optional
        Relative tolerance (see :func:`is_symmetric`).
    atol : float, optional
        Absolute tolerance, 0 by default.

    Returns
    -------
    bool
        True if `A` is square and Hermitian.

    Raises
    ------
    ShapeError
        If `A` is empty or ragged.

    Examples
    --------
    >>> is_hermitian([[2, 1j], [-1j, 3]])
    True
    >>> is_hermitian([[2, 1j], [1j, 3]])
    False
    """
    M, F = as_matrix(A)
    return _equals_transpose(M, F, conjugate=True, rtol=rtol, atol=atol)


def is_positive_definite(A: MatrixLike, *, rtol: float | None = None) -> bool:
    """Tell whether `A` is symmetric (or Hermitian) positive definite.

    The matrix must be Hermitian (checked with :func:`is_hermitian`, which means
    symmetric for a real matrix) and every pivot of Gaussian elimination without
    pivoting must be strictly positive, which is equivalent to x*Ax > 0 for every
    x ≠ 0. No square root is taken, so the test is exact for ``Fraction``
    matrices. Complexity: about n³/3 flops.

    Parameters
    ----------
    A : sequence of sequences
        A non-empty rectangular matrix.
    rtol : float, optional
        Relative tolerance of the symmetry test (see :func:`is_symmetric`).

    Returns
    -------
    bool
        True if `A` is square, Hermitian and positive definite.

    Raises
    ------
    ShapeError
        If `A` is empty or ragged.

    Examples
    --------
    >>> is_positive_definite([[4, 2], [2, 3]])
    True
    >>> is_positive_definite([[1, 2], [2, 1]])
    False
    >>> is_positive_definite([[1, 100], [0, 1]])  # not symmetric
    False
    >>> is_positive_definite([[2, 1j], [-1j, 2]])
    True
    """
    M, F = as_matrix(A)
    if not _equals_transpose(M, F, conjugate=True, rtol=rtol, atol=0):
        return False
    n = len(M)
    for k in range(n):
        pivot = F.real(M[k][k])
        if not pivot > 0:
            return False
        # Update the upper triangle of the trailing submatrix (it stays Hermitian):
        # a_ij -= a_ik a_kj / a_kk with a_ik = conj(a_ki).
        for i in range(k + 1, n):
            factor = F.conj(M[k][i]) / pivot
            for j in range(i, n):
                M[i][j] -= factor * M[k][j]
    return True
