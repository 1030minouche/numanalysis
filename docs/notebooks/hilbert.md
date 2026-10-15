---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Hilbert matrices: floats against fractions

The Hilbert matrix $H_n = \left(\frac{1}{i + j + 1}\right)_{0 \le i, j < n}$ is
symmetric positive definite, and it is the textbook example of an ill-conditioned
matrix. We solve $H_n x = b$, where $b$ is chosen so that the exact solution is
$x = (1, \dots, 1)$, twice: in `float` arithmetic and with exact fractions.

```{code-cell} python
import math
from numanalysis import cond, hilbert, solve

rows = []
for n in range(2, 16):
    H_float, H_exact = hilbert(n), hilbert(n, exact=True)
    x_float = solve(H_float, [sum(row) for row in H_float])
    x_exact = solve(H_exact, [sum(row) for row in H_exact])
    rows.append(
        (
            n,
            float(cond(H_exact, math.inf)),
            max(abs(v - 1) for v in x_float),
            x_exact == [1] * n,
        )
    )

print(f"{'n':>2}  {'cond(H_n)':>9}  {'float error':>11}  exact")
for n, c, err, exact in rows:
    print(f"{n:2d}  {c:9.1e}  {err:11.1e}  {exact}")
```

The condition number is computed exactly (from the exact inverse, whose entries are
integers) and grows like $e^{3.5n}$. In `float`, the error follows it: a few units in
the last place for $n \le 3$, about $10^{-10}$ around $n = 6$, of order one from
$n \approx 12$, and then no correct digit at all. The fractions give the exact
solution every time.

```{code-cell} python
:tags: [hide-input]
import matplotlib.pyplot as plt

ns = [r[0] for r in rows]
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.semilogy(ns, [r[2] for r in rows], "o-", label="error in float")
ax.semilogy(ns, [r[1] * 2.2e-16 for r in rows], "--", label="cond(H) × machine epsilon")
ax.set_xlabel("n")
ax.set_ylabel("max |x_i − 1|")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
```

The dashed line is the classic estimate: the relative error is bounded by about
$\operatorname{cond}(H_n)$ times the machine epsilon. The rule of thumb is that solving
a linear system loses about $\log_{10} \operatorname{cond}(A)$ significant digits.

## Where do the digits go?

The trace of the exact elimination shows the pivots: they shrink very fast, which is
where floating point loses its accuracy.

```{code-cell} python
from numanalysis import lu

res = lu(hilbert(4, exact=True), steps=True)
[res.U[k][k] for k in range(4)]
```

```{code-cell} python
print(res.steps.to_text())
```
