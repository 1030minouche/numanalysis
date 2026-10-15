# numanalysis

[![CI](https://github.com/1030minouche/numanalysis/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/1030minouche/numanalysis/actions/workflows/ci.yml)
[![Docs](https://github.com/1030minouche/numanalysis/actions/workflows/docs.yml/badge.svg?branch=main)](https://1030minouche.github.io/numanalysis/)
[![PyPI](https://img.shields.io/pypi/v/numanalysis.svg)](https://pypi.org/project/numanalysis/)
[![Python versions](https://img.shields.io/pypi/pyversions/numanalysis.svg)](https://pypi.org/project/numanalysis/)
[![Coverage](https://codecov.io/gh/1030minouche/numanalysis/branch/main/graph/badge.svg)](https://codecov.io/gh/1030minouche/numanalysis)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/1030minouche/numanalysis/blob/main/LICENSE)

*Glass-box numerical analysis in pure Python.* [Version française](https://github.com/1030minouche/numanalysis/blob/main/README.fr.md)
· [Documentation](https://1030minouche.github.io/numanalysis/)

`numanalysis` is a teaching-oriented numerical analysis library. Every algorithm is
written to be read, runs in exact or multiprecision arithmetic as well as in floats,
shows its steps on demand, and is checked against SciPy in the test suite. It does not
try to compete with NumPy or SciPy on speed.

- **Zero runtime dependency**: the core only uses the Python standard library.
- **Any number type**: `float`, `complex`, `Fraction` (exact), `Decimal` and mpmath
  (any precision); nothing is silently converted to `float`.
- **Step by step**: `steps=True` shows the pivots, the row operations and the
  intermediate matrices, as text, Markdown or LaTeX, in English or French.
- **Exercises**: generate matrices whose factorizations come out as small integers,
  with a LaTeX statement and a detailed solution.
- **Typed and tested**: type hints, `py.typed`, property-based tests, 100 % coverage.

## Installation

```bash
pip install numanalysis
```

Python 3.11 or newer is required. Optional extras: `mp` (mpmath) and `plot`
(matplotlib).

## Quick start

LU factorization with partial pivoting, `P A = L U`:

```pycon
>>> from numanalysis import lu
>>> P, L, U = lu([[0, 1], [1, 0]])
>>> P
[[0.0, 1.0], [1.0, 0.0]]
>>> U
[[1.0, 0.0], [0.0, 1.0]]

```

The same factorization in exact arithmetic:

```pycon
>>> from fractions import Fraction
>>> A = [[Fraction(2), Fraction(1)], [Fraction(4), Fraction(3)]]
>>> res = lu(A)
>>> res.L[1][0], res.U[1][1]
(Fraction(1, 2), Fraction(-1, 2))
>>> res.perm
[1, 0]

```

Solve a linear system and see how it is done:

```pycon
>>> from numanalysis import solve
>>> x, steps = solve([[Fraction(2), 1], [1, 3]], [3, 5], steps=True)
>>> x
[Fraction(4, 5), Fraction(7, 5)]
>>> print(steps.to_text().splitlines()[3])
  R₂ ← R₂ − (1/2)·R₁

```

Cholesky factorization of a symmetric positive definite matrix, `A = L Lᵀ`:

```pycon
>>> from numanalysis import cholesky, is_positive_definite
>>> cholesky([[4, 2], [2, 5]]).L
[[2.0, 0.0], [1.0, 2.0]]
>>> is_positive_definite([[1, 100], [0, 1]])
False

```

Errors are explicit and derive from `ValueError`:

```pycon
>>> cholesky([[1, 2], [2, 1]])
Traceback (most recent call last):
    ...
numanalysis.exceptions.NotPositiveDefiniteError: A is not positive definite: the leading minor of order 2 is not positive

```

> **Note.** `scipy.linalg.lu` uses the convention `A = P L U`, so its permutation
> matrix is the transpose of the one returned by `numanalysis.lu`.

## Available functions

| Function | Description |
| --- | --- |
| `lu(A, pivot=...)` | LU with partial or complete pivoting (`P A = L U`, `P A Q = L U`) |
| `cholesky(A)` | Cholesky factorization `A = L Lᵀ` (`L Lᴴ` if complex) |
| `ldlt(A)` | `A = L D Lᵀ` without square roots, exact in `Fraction` |
| `solve(A, b)` | Solve `A x = b` (vector or matrix right-hand side) |
| `solve_tridiagonal(...)` | Thomas algorithm, linear time |
| `forward_substitution`, `back_substitution` | Triangular systems |
| `det(A)`, `inv(A)` | Determinant, inverse |
| `norm(x, ord)`, `cond(A, ord)` | Norms 1, 2 (vectors), ∞, Frobenius; condition number |
| `is_square`, `is_symmetric`, `is_hermitian`, `is_positive_definite` | Properties |
| `hilbert(n)`, `vandermonde(x)`, `random_spd(n)` | Test matrices (`exact=True` for fractions) |
| `numanalysis.exercises` | Exercise generator with LaTeX statements and solutions |

Upgrading from 0.1.x? See the
[migration guide](https://github.com/1030minouche/numanalysis/blob/main/CHANGELOG.md).

## Roadmap

QR and least squares, eigenvalues, iterative methods and sparse matrices, nonlinear
equations, interpolation and quadrature, ODEs, PDEs and a mechanics/aero gallery are
planned.

## Contributing

Contributions are welcome: see
[CONTRIBUTING.md](https://github.com/1030minouche/numanalysis/blob/main/CONTRIBUTING.md).

## License

[MIT](https://github.com/1030minouche/numanalysis/blob/main/LICENSE)
