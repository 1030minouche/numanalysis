---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Exercises

{mod}`numanalysis.exercises` generates exercises whose answers are small integers, so
that they can be solved by hand: the LU factorization with partial pivoting, the
Cholesky factorization, the solution of a linear system and the determinant. Every
candidate matrix is checked by running the library in exact arithmetic.

```{code-cell} python
from numanalysis.exercises import lu_exercise

ex = lu_exercise(3, seed=4)
ex.A
```

The statement and the detailed solution are in LaTeX, in French or English:

```{code-cell} python
print(ex.statement(lang="en"))
```

```{code-cell} python
print(ex.solution(lang="fr"))
```

Since partial pivoting gives multipliers with `|l_ij| ≤ 1`, an integer `L` has
entries in {-1, 0, 1}; the rows are shuffled so that pivoting is needed.

## A complete sheet

{func}`~numanalysis.exercises.exercise_sheet` assembles exercises into a LaTeX document
(it only needs `amsmath` and compiles with `pdflatex`), with the solutions at the end,
after each exercise, or omitted:

```python
from pathlib import Path
from numanalysis.exercises import (
    cholesky_exercise,
    det_exercise,
    exercise_sheet,
    lu_exercise,
    solve_exercise,
)

exercises = [
    lu_exercise(3, seed=1),
    cholesky_exercise(3, seed=2),
    solve_exercise(3, seed=3),
    det_exercise(4, seed=4),
]
Path("sheet.tex").write_text(
    exercise_sheet(exercises, title="Direct methods", solutions="end"),
    encoding="utf-8",
)
```
