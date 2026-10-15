# numanalysis

[![CI](https://github.com/1030minouche/numanalysis/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/1030minouche/numanalysis/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/numanalysis.svg)](https://pypi.org/project/numanalysis/)
[![Python versions](https://img.shields.io/pypi/pyversions/numanalysis.svg)](https://pypi.org/project/numanalysis/)
[![Coverage](https://codecov.io/gh/1030minouche/numanalysis/branch/main/graph/badge.svg)](https://codecov.io/gh/1030minouche/numanalysis)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/1030minouche/numanalysis/blob/main/LICENSE)

*Glass-box numerical analysis in pure Python.* [Version française](https://github.com/1030minouche/numanalysis/blob/main/README.fr.md)

`numanalysis` is a teaching-oriented numerical analysis library. Every algorithm is
written to be read, works with exact arithmetic (`fractions.Fraction`) as well as
floats, and is checked against SciPy in the test suite. It does not try to compete
with NumPy or SciPy on speed.

- **Zero runtime dependency**: the core only uses the Python standard library.
- **Any matrix-like input**: lists, tuples or NumPy arrays; the input is never modified.
- **Exact when you want it**: pass `Fraction` entries and get exact results.
- **Typed and tested**: type hints, `py.typed`, property-based tests, 100 % coverage.

## Installation

```bash
pip install numanalysis
```

Python 3.11 or newer is required.

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

| Function                  | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| `lu(A)`                   | LU with partial pivoting, returns `P, L, U` (`P A = L U`) |
| `cholesky(A)`             | Cholesky factorization, returns `.L` (`A = L Lᵀ`)        |
| `is_square(A)`            | Is the matrix square?                                    |
| `is_symmetric(A)`         | Is the matrix symmetric (with a tolerance for floats)?   |
| `is_positive_definite(A)` | Is the matrix symmetric positive definite?               |

Upgrading from 0.1.x? See the
[migration guide](https://github.com/1030minouche/numanalysis/blob/main/CHANGELOG.md).

## Roadmap

Triangular solvers, determinants, conditioning, step-by-step traces with LaTeX
output, QR, eigenvalues, iterative methods, ODEs, PDEs and a mechanics/aero gallery
are planned.

## Contributing

Contributions are welcome: see
[CONTRIBUTING.md](https://github.com/1030minouche/numanalysis/blob/main/CONTRIBUTING.md).

## License

[MIT](https://github.com/1030minouche/numanalysis/blob/main/LICENSE)
