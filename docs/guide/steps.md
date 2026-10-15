---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Step-by-step traces

Pass `steps=True` to {func}`~numanalysis.lu`, {func}`~numanalysis.cholesky`,
{func}`~numanalysis.ldlt`, {func}`~numanalysis.solve`, {func}`~numanalysis.det`,
{func}`~numanalysis.inv` or the substitutions to record what the algorithm does: the
pivot it chooses, the swaps, every row operation, every computed entry with its
numbers, and the intermediate matrices.

```{code-cell} python
from fractions import Fraction
from numanalysis import lu

res = lu([[Fraction(2), 1, 1], [4, 3, 3], [8, 7, 9]], steps=True)
print(res.steps)
```

The factorizations keep the trace in their `steps` attribute. {func}`~numanalysis.solve`,
{func}`~numanalysis.det` and {func}`~numanalysis.inv` return a
{class}`~numanalysis.Traced` result instead, which unpacks as `value, steps`:

```{code-cell} python
from numanalysis import solve

x, steps = solve([[Fraction(2), 1], [1, 3]], [3, 5], steps=True, lang="fr")
print(steps)
```

## Formats and languages

A {class}`~numanalysis.Steps` object renders as text, Markdown or LaTeX, in English
(`"en"`) or French (`"fr"`):

```{code-cell} python
from numanalysis import cholesky

steps = cholesky([[4, 2], [2, 5]], steps=True).steps
print(steps.to_markdown(lang="fr"))
```

```{code-cell} python
print(steps.to_latex())
```

In Jupyter, a trace (or a result that carries one) is displayed in LaTeX:

```{code-cell} python
steps
```

For a LaTeX document, `to_latex(environment="align")` gives an `align*` environment
whose lines can break across pages (with `\allowdisplaybreaks`).

## When something fails

The exception raised by a traced call carries the steps up to the failure, in its
`steps` attribute:

```{code-cell} python
from numanalysis import NotPositiveDefiniteError

try:
    cholesky([[1, 2], [2, 1]], steps=True)
except NotPositiveDefiniteError as exc:
    print(exc.steps)
```

## Structured steps

A trace is a sequence of step objects ({class}`~numanalysis._steps.Pivot`,
{class}`~numanalysis._steps.Swap`, {class}`~numanalysis._steps.Eliminate`,
{class}`~numanalysis._steps.Formula`...), so a program can inspect it too:

```{code-cell} python
from numanalysis._steps import Eliminate

[(s.target, s.source, s.factor) for s in res.steps if isinstance(s, Eliminate)]
```
