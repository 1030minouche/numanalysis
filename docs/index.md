# numanalysis

**Glass-box numerical analysis in pure Python.**

numanalysis implements the classic algorithms of numerical analysis so that you can
*look inside them*: every algorithm runs in exact or multiprecision arithmetic,
shows its steps, is verified against SciPy, and has no runtime dependency.

- **Any number type.** The same code runs in `float`, `complex`,
  {class}`~fractions.Fraction` (exact), {class}`~decimal.Decimal` and
  [mpmath](https://mpmath.org) (any precision). See {doc}`guide/scalars`.
- **Step by step.** Pass `steps=True` to see the pivots, the row operations and the
  intermediate matrices, as text, Markdown or LaTeX, in English or French. See
  {doc}`guide/steps`.
- **Ready-made exercises.** Generate matrices whose factorizations come out as small
  integers, with a LaTeX statement and a detailed solution. See {doc}`guide/exercises`.
- **Verified.** Property-based tests in exact arithmetic, SciPy as the oracle, 100 %
  coverage.

```python
>>> from fractions import Fraction
>>> from numanalysis import solve
>>> solve([[Fraction(2), 1], [1, 3]], [3, 5])
[Fraction(4, 5), Fraction(7, 5)]
```

```{toctree}
:maxdepth: 2
:caption: User guide

installation
guide/scalars
guide/steps
guide/exercises
```

```{toctree}
:maxdepth: 1
:caption: Examples

notebooks/hilbert
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/linalg
api/steps
api/exercises
api/exceptions
changelog
```
