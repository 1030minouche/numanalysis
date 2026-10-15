# ruff: noqa: E501 -- the translation table has long sentences
"""Step-by-step traces of the algorithms, rendered as text, Markdown or LaTeX.

An algorithm called with ``steps=True`` records a :class:`Steps` object: a list
of structured, language-independent steps (pivot choice, row swap, row
operation, formula with its numbers, intermediate matrix...). The trace is
rendered on demand, in English or French. In Jupyter, a trace (and a result
that carries one) is displayed in LaTeX.

Examples
--------
>>> from numanalysis import lu
>>> res = lu([[2, 1], [4, 3]], steps=True)
>>> print(res.steps.to_text(lang="fr"))
Élimination de Gauss : L_i ← L_i − (a_ik / a_kk)·L_k sous chaque pivot
Étape 1
  pivot : le plus grand |a_i1| de la colonne 1 vaut 4, ligne 2
  L₁ ↔ L₂
  L₂ ← L₂ − 0.5·L₁
  Matrice après l'étape 1 :
    [4     3]
    [0  -0.5]
Résultat
  P =
    [0  1]
    [1  0]
  L =
    [  1  0]
    [0.5  1]
  U =
    [4     3]
    [0  -0.5]
"""

import re
import sys
import textwrap
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from fractions import Fraction
from string import Template
from typing import Any, overload

from ._scalar import Scalar

__all__ = [
    "Determinant",
    "Eliminate",
    "Formula",
    "Header",
    "MatrixSnapshot",
    "Note",
    "Pivot",
    "Step",
    "Steps",
    "Swap",
    "VectorSnapshot",
]

LANGUAGES = ("en", "fr")
FORMATS = ("text", "markdown", "latex")

# Sentences of the traces; ``{}`` fields receive already rendered pieces.
_WORDS: dict[str, dict[str, str]] = {
    "en": {
        "row": "R",
        "col": "C",
        "step": "Step {}",
        "result": "Result",
        "after_step": "Matrix after step {}",
        "pivot_partial": "pivot: the largest |a_i{col}| of column {col} is {value}, in row {row}",
        "pivot_complete": "pivot: the largest |a_ij| of the remaining submatrix is {value}, in row {row}, column {col}",
        "zero_column": "the column is zero below the diagonal: nothing to eliminate",
        "zero_submatrix": "the remaining submatrix is zero: nothing to eliminate",
        "lu_rule": "Gaussian elimination: R_i ← R_i − (a_ik / a_kk)·R_k below each pivot",
        "forward": "Forward substitution: L y = P b",
        "forward_plain": "Forward substitution: L x = b",
        "back_plain": "Back substitution: U x = b",
        "back": "Back substitution: U z = y",
        "back_x": "Back substitution: U x = y",
        "permuted_rhs": "Permuted right-hand side",
        "unpermute": "Undo the column permutation: x = Q z",
        "solution": "Solution",
        "determinant": "Determinant: product of the pivots, times −1 for each swap",
        "inverse": "Inverse: solve A X = I, all columns at once",
        "cholesky_rule": "l_ij = (a_ij − Σ_k l_ik l_jk) / l_jj for j < i, and l_ii = √(a_ii − Σ_k l_ik²)",
        "ldlt_rule": "d_j = a_jj − Σ_k l_jk² d_k, and l_ij = (a_ij − Σ_k l_ik l_jk d_k) / d_j for i > j",
        "row_n": "Row {}",
        "column_n": "Column {}",
        "singular": "zero pivot: the matrix is singular",
        "not_pd": "the term under the square root is not positive: the matrix is not positive definite",
        "zero_pivot": "zero pivot: the factorization without pivoting stops",
    },
    "fr": {
        "row": "L",
        "col": "C",
        "step": "Étape {}",
        "result": "Résultat",
        "after_step": "Matrice après l'étape {}",
        "pivot_partial": "pivot : le plus grand |a_i{col}| de la colonne {col} vaut {value}, ligne {row}",
        "pivot_complete": "pivot : le plus grand |a_ij| de la sous-matrice restante vaut {value}, ligne {row}, colonne {col}",
        "zero_column": "la colonne est nulle sous la diagonale : rien à éliminer",
        "zero_submatrix": "la sous-matrice restante est nulle : rien à éliminer",
        "lu_rule": "Élimination de Gauss : L_i ← L_i − (a_ik / a_kk)·L_k sous chaque pivot",
        "forward": "Descente : L y = P b",
        "forward_plain": "Descente : L x = b",
        "back_plain": "Remontée : U x = b",
        "back": "Remontée : U z = y",
        "back_x": "Remontée : U x = y",
        "permuted_rhs": "Second membre permuté",
        "unpermute": "On défait la permutation des colonnes : x = Q z",
        "solution": "Solution",
        "determinant": "Déterminant : produit des pivots, multiplié par −1 à chaque échange",
        "inverse": "Inverse : on résout A X = I, toutes les colonnes à la fois",
        "cholesky_rule": "l_ij = (a_ij − Σ_k l_ik l_jk) / l_jj pour j < i, et l_ii = √(a_ii − Σ_k l_ik²)",
        "ldlt_rule": "d_j = a_jj − Σ_k l_jk² d_k, et l_ij = (a_ij − Σ_k l_ik l_jk d_k) / d_j pour i > j",
        "row_n": "Ligne {}",
        "column_n": "Colonne {}",
        "singular": "pivot nul : la matrice est singulière",
        "not_pd": "le terme sous la racine n'est pas positif : la matrice n'est pas définie positive",
        "zero_pivot": "pivot nul : la factorisation sans pivotage s'arrête",
    },
}

# LaTeX versions of the sentences that contain mathematics (the others are
# put in a \text block). Fields are string.Template placeholders ($value).
_LATEX_WORDS: dict[str, dict[str, str]] = {
    "en": {
        "pivot_partial": r"\text{pivot: the largest } |a_{i$col}| \text{ of column $col is } $value\text{, in row $row}",
        "pivot_complete": r"\text{pivot: the largest } |a_{ij}| \text{ of the remaining submatrix is } $value\text{, in row $row, column $col}",
        "lu_rule": r"\text{Gaussian elimination: } R_i \leftarrow R_i - \frac{a_{ik}}{a_{kk}}\, R_k \text{ below each pivot}",
        "forward": r"\text{Forward substitution: } L y = P b",
        "forward_plain": r"\text{Forward substitution: } L x = b",
        "back_plain": r"\text{Back substitution: } U x = b",
        "back": r"\text{Back substitution: } U z = y",
        "back_x": r"\text{Back substitution: } U x = y",
        "unpermute": r"\text{Undo the column permutation: } x = Q z",
        "determinant": r"\text{Determinant: product of the pivots, times } -1 \text{ for each swap}",
        "inverse": r"\text{Inverse: solve } A X = I\text{, all columns at once}",
        "cholesky_rule": r"l_{ij} = \frac{a_{ij} - \sum_k l_{ik} l_{jk}}{l_{jj}} \text{ for } j < i, \quad l_{ii} = \sqrt{a_{ii} - \sum_k l_{ik}^2}",
        "ldlt_rule": r"d_j = a_{jj} - \sum_k l_{jk}^2 d_k, \quad l_{ij} = \frac{a_{ij} - \sum_k l_{ik} l_{jk} d_k}{d_j} \text{ for } i > j",
    },
    "fr": {
        "pivot_partial": r"\text{pivot : le plus grand } |a_{i$col}| \text{ de la colonne $col vaut } $value\text{, ligne $row}",
        "pivot_complete": r"\text{pivot : le plus grand } |a_{ij}| \text{ de la sous-matrice restante vaut } $value\text{, ligne $row, colonne $col}",
        "lu_rule": r"\text{Élimination de Gauss : } L_i \leftarrow L_i - \frac{a_{ik}}{a_{kk}}\, L_k \text{ sous chaque pivot}",
        "forward": r"\text{Descente : } L y = P b",
        "forward_plain": r"\text{Descente : } L x = b",
        "back_plain": r"\text{Remontée : } U x = b",
        "back": r"\text{Remontée : } U z = y",
        "back_x": r"\text{Remontée : } U x = y",
        "unpermute": r"\text{On défait la permutation des colonnes : } x = Q z",
        "determinant": r"\text{Déterminant : produit des pivots, multiplié par } -1 \text{ à chaque échange}",
        "inverse": r"\text{Inverse : on résout } A X = I\text{, toutes les colonnes à la fois}",
        "cholesky_rule": r"l_{ij} = \frac{a_{ij} - \sum_k l_{ik} l_{jk}}{l_{jj}} \text{ pour } j < i, \quad l_{ii} = \sqrt{a_{ii} - \sum_k l_{ik}^2}",
        "ldlt_rule": r"d_j = a_{jj} - \sum_k l_{jk}^2 d_k, \quad l_{ij} = \frac{a_{ij} - \sum_k l_{ik} l_{jk} d_k}{d_j} \text{ pour } i > j",
    },
}

_SUBSCRIPTS = str.maketrans("0123456789,", "₀₁₂₃₄₅₆₇₈₉,")
_TEXT_OPS = {"-": "−", "+": "+", "*": "·", "<-": "←", "<->": "↔"}
_LATEX_OPS = {
    "-": "-",
    "+": "+",
    "*": r"\cdot ",
    "<-": r"\leftarrow",
    "<->": r"\leftrightarrow",
}


def _is_complex(x: Any) -> bool:
    if isinstance(x, complex):
        return True
    mpmath = sys.modules.get("mpmath")
    return mpmath is not None and isinstance(x, mpmath.mpc)


@dataclass(frozen=True)
class _Renderer:
    """Turn the pieces of a step into text, Markdown or LaTeX."""

    fmt: str
    lang: str
    digits: int

    @property
    def latex(self) -> bool:
        return self.fmt == "latex"

    def word(self, key: str, *args: Any, **kwargs: Any) -> str:
        """Return a translated sentence, as a LaTeX text block if needed."""
        if self.latex and key in _LATEX_WORDS[self.lang]:
            return Template(_LATEX_WORDS[self.lang][key]).substitute(kwargs)
        s = _WORDS[self.lang][key].format(*args, **kwargs)
        return r"\text{" + s.replace("_", r"\_") + "}" if self.latex else s

    def heading(self, s: str) -> str:
        """Return a section title."""
        if self.latex:
            return s.replace(r"\text{", r"\textbf{")
        return f"**{s}**" if self.fmt == "markdown" else s

    def sub(self, name: str, *indices: int) -> str:
        """Return `name` with 1-based indices: ``l₂₁`` (text) or ``l_{21}``."""
        sep = "," if any(i >= 9 for i in indices) else ""
        idx = sep.join(str(i + 1) for i in indices)
        return f"{name}_{{{idx}}}" if self.latex else name + idx.translate(_SUBSCRIPTS)

    def name(self, s: str) -> str:
        """Return a matrix or vector name (``A⁻¹`` becomes ``A^{-1}`` in LaTeX)."""
        return s.replace("⁻¹", "^{-1}") if self.latex else s

    def op(self, symbol: str) -> str:
        """Return an operator symbol (``-``, ``+``, ``*``, ``<-``, ``<->``)."""
        return (_LATEX_OPS if self.latex else _TEXT_OPS)[symbol]

    def sqrt(self, s: str) -> str:
        """Return the square root of the expression `s`."""
        return rf"\sqrt{{{s}}}" if self.latex else f"√({s})"

    def group(self, s: str) -> str:
        """Return `s` between parentheses."""
        return rf"\left({s}\right)" if self.latex else f"({s})"

    def num(self, x: Any) -> str:
        """Format a scalar, or a list of scalars as a tuple."""
        if isinstance(x, list | tuple):
            return self.group(", ".join(self.num(v) for v in x))
        if isinstance(x, Fraction):
            if x.denominator == 1:
                return str(x.numerator)
            if self.latex:
                sign = "-" if x < 0 else ""
                return rf"{sign}\frac{{{abs(x.numerator)}}}{{{x.denominator}}}"
            return f"{x.numerator}/{x.denominator}"
        if isinstance(x, float | Decimal):
            return format(x, f".{self.digits}g")
        if isinstance(x, complex):
            re = format(x.real, f".{self.digits}g")
            im = format(abs(x.imag), f".{self.digits}g")
            unit = "i" if self.latex else "j"
            return self.group(f"{re} {'-' if x.imag < 0 else '+'} {im}{unit}")
        mpmath = sys.modules.get("mpmath")
        if mpmath is not None and isinstance(x, mpmath.mpf | mpmath.mpc):
            return str(mpmath.nstr(x, self.digits))
        return str(x)

    def paren(self, x: Any) -> str:
        """Format a number, between parentheses if negative (or a text fraction)."""
        s = self.num(x)
        if s.startswith("-") or (not self.latex and "/" in s):
            return self.group(s)
        return s

    def matrix(self, M: Sequence[Sequence[Any]]) -> str:
        """Format a matrix: aligned rows (text, Markdown) or ``pmatrix``."""
        cells = [[self.num(x) for x in row] for row in M]
        if self.latex:
            body = r" \\ ".join(" & ".join(row) for row in cells)
            return r"\begin{pmatrix} " + body + r" \end{pmatrix}"
        widths = [max(len(row[j]) for row in cells) for j in range(len(cells[0]))]
        lines = [
            "[" + "  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)) + "]"
            for row in cells
        ]
        if self.fmt == "markdown":
            return "```\n" + "\n".join(lines) + "\n```"
        return textwrap.indent("\n".join(lines), "  ")

    def vector(self, v: Sequence[Any]) -> str:
        """Format a vector: a tuple (text, Markdown) or a column ``pmatrix``."""
        return self.matrix([[x] for x in v]) if self.latex else self.num(list(v))


class Step:
    """Base class of the recorded steps; :meth:`render` gives its line(s)."""

    #: Indentation level: 0 for section titles, 1 for the steps.
    level: int = 1

    def render(self, r: _Renderer) -> str:  # pragma: no cover - abstract
        """Render the step with the renderer `r`."""
        raise NotImplementedError


@dataclass(frozen=True)
class Header(Step):
    """A section title; `key` names a sentence of the translation table."""

    key: str
    args: tuple[Any, ...] = ()
    level: int = 0

    def render(self, r: _Renderer) -> str:
        """Render the title."""
        return r.heading(r.word(self.key, *self.args))


@dataclass(frozen=True)
class Note(Step):
    """A sentence of the translation table (`level` 0 for a general rule)."""

    key: str
    args: tuple[Any, ...] = ()
    level: int = 1

    def render(self, r: _Renderer) -> str:
        """Render the sentence."""
        return r.word(self.key, *(r.num(a) for a in self.args))


@dataclass(frozen=True)
class Pivot(Step):
    """Choice of the pivot at (`row`, `col`), 0-based, with its value."""

    row: int
    col: int
    value: Scalar
    complete: bool = False

    def render(self, r: _Renderer) -> str:
        """Render the pivot choice."""
        key = "pivot_complete" if self.complete else "pivot_partial"
        return r.word(key, row=self.row + 1, col=self.col + 1, value=r.num(self.value))


@dataclass(frozen=True)
class Swap(Step):
    """Swap of rows (``axis="row"``) or columns (``axis="col"``) `i` and `j`."""

    axis: str
    i: int
    j: int

    def render(self, r: _Renderer) -> str:
        """Render ``R₁ ↔ R₂``."""
        name = _WORDS[r.lang][self.axis]
        return f"{r.sub(name, self.i)} {r.op('<->')} {r.sub(name, self.j)}"


@dataclass(frozen=True)
class Eliminate(Step):
    """Row operation ``R_target ← R_target − factor·R_source``."""

    target: int
    source: int
    factor: Scalar

    def render(self, r: _Renderer) -> str:
        """Render the row operation, as ``R₂ ← R₂ + 2·R₁`` for a negative factor."""
        name = _WORDS[r.lang]["row"]
        t, s = r.sub(name, self.target), r.sub(name, self.source)
        sign, f = "-", self.factor
        if not _is_complex(f) and f < 0:
            sign, f = "+", -f
        coef = "" if f == 1 else r.paren(f) + r.op("*")
        return f"{t} {r.op('<-')} {t} {r.op(sign)} {coef}{s}"


@dataclass(frozen=True)
class Formula(Step):
    """A computed entry with its numbers: ``name = [√](a − s) [/ div] = value``.

    `s` is omitted when there is no sum to subtract (``None``), and `div` when
    there is no division.
    """

    name: str
    indices: tuple[int, ...]
    a: Scalar
    s: Scalar | None
    div: Scalar | None
    value: Scalar
    sqrt: bool = False

    def render(self, r: _Renderer) -> str:
        """Render the formula."""
        lhs, value = r.sub(self.name, *self.indices), r.num(self.value)
        if self.s is None and self.div is None and not self.sqrt:
            return f"{lhs} = {value}"
        bare = self.s is None and self.div is not None and not self.sqrt
        expr = r.paren(self.a) if bare else r.num(self.a)
        if self.s is not None:
            expr = f"{expr} {r.op('-')} {r.paren(self.s)}"
        if self.sqrt:
            expr = r.sqrt(expr)
        if self.div is not None:
            if self.s is not None and not self.sqrt:
                expr = r.group(expr)
            expr = f"{expr} / {r.paren(self.div)}"
        return f"{lhs} = {expr} = {value}"


@dataclass(frozen=True)
class Determinant(Step):
    """``det A = (−1) · u₁₁ · … · uₙₙ = value``."""

    sign: int
    pivots: tuple[Scalar, ...]
    value: Scalar

    def render(self, r: _Renderer) -> str:
        """Render the product."""
        factors = [r.paren(p) for p in self.pivots]
        if self.sign < 0:
            factors.insert(0, r.group(r.op("-") + "1"))
        times = f" {r.op('*').strip()} "
        det = r"\det A" if r.latex else "det A"
        return f"{det} = {times.join(factors)} = {r.num(self.value)}"


@dataclass(frozen=True)
class MatrixSnapshot(Step):
    """An intermediate or final matrix, with a label (`key`) or a `name`."""

    matrix: tuple[tuple[Scalar, ...], ...]
    key: str | None = None
    args: tuple[Any, ...] = ()
    name: str | None = None

    def render(self, r: _Renderer) -> str:
        """Render the label, then the matrix."""
        if self.name:
            label = f"{r.name(self.name)} ="
        else:
            colon = " :" if r.lang == "fr" else ":"
            label = r.word(self.key or "result", *self.args)
            label += r"\text{" + colon + "}" if r.latex else colon
        return f"{label}{' ' if r.latex else chr(10)}{r.matrix(self.matrix)}"


@dataclass(frozen=True)
class VectorSnapshot(Step):
    """A named vector: ``x = (1, 2)``, with an optional label `key`."""

    name: str
    values: tuple[Scalar, ...]
    key: str | None = None

    def render(self, r: _Renderer) -> str:
        """Render ``label: name = vector``."""
        label = ""
        if self.key:
            label = r.word(self.key) + (
                r"\quad " if r.latex else (" : " if r.lang == "fr" else ": ")
            )
        return f"{label}{r.name(self.name)} = {r.vector(self.values)}"


def _check_lang(lang: str) -> None:
    if lang not in LANGUAGES:
        raise ValueError(f"lang must be one of {LANGUAGES}, got {lang!r}")


@dataclass(frozen=True)
class Steps(Sequence[Step]):
    """The trace of an algorithm: a sequence of :class:`Step`.

    Parameters
    ----------
    items : list of Step
        The recorded steps.
    lang : {"en", "fr"}, optional
        Default language of :func:`str` and of the Jupyter display.

    Examples
    --------
    >>> from numanalysis import cholesky
    >>> print(cholesky([[4, 2], [2, 5]], steps=True).steps)
    l_ij = (a_ij − Σ_k l_ik l_jk) / l_jj for j < i, and l_ii = √(a_ii − Σ_k l_ik²)
    Row 1
      l₁₁ = √(4) = 2
    Row 2
      l₂₁ = 2 / 2 = 1
      l₂₂ = √(5 − 1) = 2
    Result
      L =
        [2  0]
        [1  2]
    """

    items: list[Step] = field(default_factory=list)
    lang: str = "en"

    def __post_init__(self) -> None:
        _check_lang(self.lang)

    @overload
    def __getitem__(self, i: int) -> Step: ...
    @overload
    def __getitem__(self, i: slice) -> list[Step]: ...
    def __getitem__(self, i: int | slice) -> Step | list[Step]:
        return self.items[i]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.items)

    # -- recording (used by the algorithms) ----------------------------------
    def add(self, step: Step) -> None:
        """Append a step."""
        self.items.append(step)

    def snapshot(
        self,
        M: Sequence[Sequence[Scalar]],
        key: str | None = None,
        *args: Any,
        name: str | None = None,
    ) -> None:
        """Append a copy of the matrix `M`."""
        frozen = tuple(tuple(row) for row in M)
        self.items.append(MatrixSnapshot(frozen, key, args, name))

    # -- rendering -------------------------------------------------------------
    def with_lang(self, lang: str) -> "Steps":
        """Return the same trace with another default language."""
        return replace(self, lang=lang)

    def render(
        self, fmt: str = "text", *, lang: str | None = None, digits: int = 6
    ) -> str:
        r"""Render the trace.

        Parameters
        ----------
        fmt : {"text", "markdown", "latex"}
            Output format. The LaTeX output is an ``array`` environment to put in
            math mode (``\[ … \]`` or ``$$ … $$``); it needs ``amsmath``.
        lang : {"en", "fr"}, optional
            Language; defaults to the one of the trace.
        digits : int, optional
            Significant digits of floating-point numbers (fractions are exact).

        Returns
        -------
        str
            The rendered trace.
        """
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
        lang = self.lang if lang is None else lang
        _check_lang(lang)
        r = _Renderer(fmt, lang, digits)
        rendered = [(step.level, step.render(r)) for step in self.items]
        if fmt == "latex":
            body = [(r"\quad " if level else "") + line for level, line in rendered]
            return "\\begin{array}{l}\n" + " \\\\\n".join(body) + "\n\\end{array}"
        if fmt == "markdown":
            out: list[str] = []
            for level, line in rendered:
                head, _, block = line.partition("\n")
                if level == 0:
                    out.append(f"\n{head}\n")
                elif block:  # a label followed by a code block
                    out.append(f"- {head}\n\n{block}\n")
                else:
                    out.append(f"- {head}")
            return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
        return "\n".join(
            textwrap.indent(line, "  " * level) for level, line in rendered
        )

    def to_text(self, *, lang: str | None = None, digits: int = 6) -> str:
        """Render as plain text (with Unicode symbols)."""
        return self.render("text", lang=lang, digits=digits)

    def to_markdown(self, *, lang: str | None = None, digits: int = 6) -> str:
        """Render as Markdown: a bulleted list, matrices in code blocks."""
        return self.render("markdown", lang=lang, digits=digits)

    def to_latex(self, *, lang: str | None = None, digits: int = 6) -> str:
        """Render as a LaTeX ``array``, to put in math mode."""
        return self.render("latex", lang=lang, digits=digits)

    def __str__(self) -> str:
        return self.to_text()

    def _repr_latex_(self) -> str:
        return "$$" + self.to_latex() + "$$"
