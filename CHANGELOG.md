# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Every algorithm now runs in the scalar type of its input: `float`, `complex`,
  `Fraction`, `Decimal`, and `mpmath.mpf`/`mpc` (optional `mp` extra). A matrix mixing
  kinds is computed in one common type (e.g. `int` + `Fraction` → `Fraction`,
  `Fraction` + `float` → `float`); `Decimal` cannot be mixed with `float`, `complex`,
  `Fraction` or mpmath numbers and raises `TypeError`.
- `is_hermitian(A)`. `cholesky` and `is_positive_definite` accept complex Hermitian
  matrices (`A = L Lᴴ`).
- `InexactResultError`, raised when an exact type cannot represent a result.
- Direct solvers: `forward_substitution`, `back_substitution`, `solve` (vector or
  matrix right-hand side), `det` (product of the pivots times the sign of the
  permutations), `inv`, and `solve_tridiagonal` (Thomas algorithm, linear time).
- Complete pivoting: `lu(A, pivot="complete")` gives `P A Q = L U` and unpacks as
  `P, L, U, Q`; `solve`, `det` and `inv` accept the same `pivot` option. `LUResult`
  gains `Q`, `col_perm` and `pivot`.
- `ldlt(A)`: square-root-free factorization `A = L D Lᵀ` (`L D Lᴴ` if complex),
  exact for `Fraction` matrices and valid for indefinite ones. Returns an `LDLResult`.
- `norm(x, ord)` for vectors (1, 2, ∞) and matrices (1, ∞, Frobenius), real-valued
  and exact for `Fraction` in the 1- and ∞-norms; `cond(A, ord=1)`.
- Test matrices: `hilbert(n)`, `vandermonde(x)` and `random_spd(n, seed=...)`
  (integer entries), all with an `exact=True` option that returns `Fraction` entries.
- Step-by-step traces: `lu`, `cholesky`, `ldlt`, `solve`, `det`, `inv` and the
  substitutions accept `steps=True` (and `lang="fr"`). The trace records the pivot
  choices, the row and column swaps, the row operations (`R₂ ← R₂ − 2·R₁`), every
  computed entry with its numbers, and the intermediate matrices. It renders as text,
  Markdown or LaTeX (`Steps.to_text`, `to_markdown`, `to_latex`), in English or French,
  and is displayed in LaTeX in Jupyter. Factorizations store it in `.steps`; `solve`,
  `det` and `inv` then return a `Traced` result that unpacks as `value, steps`. When a
  traced call fails, the exception carries the steps up to the failure.
- `numanalysis.exercises`: generators of exercises whose answers are small integers
  (`lu_exercise`, `cholesky_exercise`, `solve_exercise`, `det_exercise`), with a LaTeX
  statement and a detailed solution in French or English, and `exercise_sheet` to
  build a complete document (compiles with `pdflatex`). `Steps.to_latex` gains an
  `environment="align"` option whose lines can break across pages.
- `ZeroPivotError`, raised by the methods without pivoting (`ldlt`,
  `solve_tridiagonal`) when they meet a zero pivot.

### Changed

- `cholesky` on `Fraction` input is now exact (it returned `float` before), and raises
  `InexactResultError` when a square root is irrational.
- Default relative tolerance of the symmetry tests: 4096 machine epsilons of the
  current precision (about 9.1e-13 for `float`, was 1e-12). `Decimal` and mpmath
  matrices are now compared with this relative tolerance; only `Fraction` matrices
  are compared exactly.
- Mixed-type matrices give one consistent output type.

## [0.2.0] - 2026-10-13

### Added

- `numanalysis.linalg` subpackage, re-exported at the top level.
- `lu(A)`: LU factorization with partial pivoting, `P A = L U`, computed in a single
  pass. Returns an `LUResult` that unpacks as `P, L, U` and exposes `perm`.
- `cholesky(A)`: single-pass Cholesky factorization returning a `CholeskyResult`
  (`.L`, unpacks as `(L,)`).
- `is_square`, `is_symmetric` (relative tolerance for floats, exact comparison for
  `Fraction`/`Decimal`) and `is_positive_definite` (symmetry + positive pivots, no
  square root, exact for fractions).
- `numanalysis.exceptions`: `NumAnalysisError`, `ShapeError`, `SingularMatrixError`,
  `NotSymmetricError`, `NotPositiveDefiniteError`, all subclasses of `ValueError`.
- Input validation: any sequence of sequences (lists, tuples, NumPy arrays) is
  accepted and never modified; empty, ragged or non-numeric input raises a clear error.
- `Fraction` and `Decimal` matrices are kept exact by `lu` and `is_positive_definite`.
- Type hints and a `py.typed` marker.
- Test suite: regression tests, `hypothesis` properties, SciPy/NumPy as oracles,
  doctests, 100 % coverage.
- Tooling: ruff, mypy (strict), pre-commit.

### Changed

- Sources moved to a `src/numanalysis/` layout; packaging now uses `pyproject.toml`
  with the hatchling backend. The version is defined once in `numanalysis.__version__`.
- Every public name is in English (see the migration guide below).
- Integer input gives `float` output everywhere.

### Fixed

- `from numanalysis import cholesky` raised `ModuleNotFoundError: No module named 'lu'`
  once installed (absolute import replaced by a relative one).
- Importing the package printed messages and ran a test computation.
- LU had no pivoting: `[[0, 1], [1, 0]]` raised `ZeroDivisionError` and
  `[[1e-20, 1], [1, 1]]` gave a wrong factorization.
- `est_definie_positive([[1, 100], [0, 1]])` returned `True`: only the lower triangle
  was read. `is_positive_definite` now checks symmetry first.
- `llu_unit` and `ulu_unit` each recomputed the factorization, and `cholesky`
  factorized twice.
- Inconsistent output types (`[[4, 3], [0.0, -1.5]]`) and failing doctests.
- Wrong metadata: `Requires-Python >=2.7`, Python 2 classifier and outdated
  description. Python ≥ 3.11 is now required.

### Migration from 0.1.x

| 0.1.x                         | 0.2.0                                    |
| ----------------------------- | ---------------------------------------- |
| `lu.verifCarre(A)`            | `is_square(A)`                           |
| `cholesky.est_symetrique(A)`  | `is_symmetric(A)`                        |
| `cholesky.est_definie_positive(A)` | `is_positive_definite(A)`           |
| `lu.llu_unit(A)`              | `lu(A).L` (now with pivoting: `P A = L U`) |
| `lu.ulu_unit(A)`              | `lu(A).U`                                |
| `cholesky.cholesky(A)`        | `cholesky(A).L`                          |
| `ValueError`                  | `ShapeError`, `NotSymmetricError`, `NotPositiveDefiniteError` (still `ValueError` subclasses) |

Because `lu` now pivots, `L` and `U` can differ from the 0.1.x results: compare
`L U` with `P A`, not with `A`.

## [0.1.1] - 2025-10-19

### Added

- First published version: LU (without pivoting) and Cholesky factorizations.

[Unreleased]: https://github.com/1030minouche/numanalysis/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/1030minouche/numanalysis/compare/05ce63b...v0.2.0
[0.1.1]: https://pypi.org/project/numanalysis/0.1.1/
