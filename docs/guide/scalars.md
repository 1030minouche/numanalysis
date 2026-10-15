---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Number types

Every algorithm is written once and runs in the number type of its input. Nothing is
ever silently converted to `float`.

| Input entries | Computed in | Arithmetic |
|---|---|---|
| `int` only | `float` | IEEE 754 double precision |
| any `float` | `float` | double precision |
| any `complex` | `complex` | double precision |
| `Fraction` (and `int`) | `Fraction` | **exact** |
| `Decimal` (and `int`) | `Decimal` | current {mod}`decimal` context |
| `mpmath.mpf` (and reals) | `mpf` | current `mpmath.mp` precision |
| `mpmath.mpc`, or `mpf` + `complex` | `mpc` | current `mpmath.mp` precision |

A matrix mixing kinds is computed in one common type, so the output always has
consistent entries:

```{code-cell} python
from fractions import Fraction
from numanalysis import lu

P, L, U = lu([[Fraction(1, 2), 1], [3, 4]])
L
```

`Decimal` refuses to mix with `float`, `complex`, `Fraction` and mpmath numbers,
because Python itself refuses such arithmetic.

## Exact arithmetic with fractions

With `Fraction` entries, LU, LDLᵀ, `solve`, `det`, `inv` and the 1- and ∞-norms are
exact. A Cholesky factorization is exact only when its square roots are rational;
otherwise {class}`~numanalysis.InexactResultError` is raised and
{func}`~numanalysis.ldlt`, which takes no square root, is the exact alternative:

```{code-cell} python
from numanalysis import cholesky, ldlt, InexactResultError

A = [[Fraction(2), 1], [1, 2]]
try:
    cholesky(A)
except InexactResultError as exc:
    print(exc)
ldlt(A).d
```

## Multiprecision with mpmath

```{code-cell} python
import mpmath
from numanalysis import hilbert, solve

with mpmath.workdps(50):
    H = [[mpmath.mpf(x) for x in row] for row in hilbert(8, exact=True)]
    x = solve(H, [sum(row) for row in H])  # the exact solution is all ones
    print(max(abs(v - 1) for v in x))
```

## Tolerances

Structural tests such as {func}`~numanalysis.is_symmetric` compare entries exactly for
`Fraction` matrices, and with a relative tolerance of 4096 machine epsilons of the
current precision otherwise (about 9.1e-13 for `float`). Every test accepts explicit
`rtol` and `atol` arguments.
