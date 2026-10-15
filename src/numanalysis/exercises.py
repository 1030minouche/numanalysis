# ruff: noqa: E501 -- statements are long sentences
r"""Exercise generator: integer matrices whose factorizations "come out nicely".

Each generator returns an :class:`Exercise` whose data and answers are small
integers: the LU factorization with partial pivoting, the Cholesky factor, the
solution of ``A x = b`` or the determinant. The statement and the detailed
solution (the recorded steps of the algorithm) are written in LaTeX, in French
or English, and :func:`exercise_sheet` assembles a complete document.

Examples
--------
>>> from numanalysis.exercises import lu_exercise
>>> ex = lu_exercise(3, seed=4)
>>> ex.A
[[0, 1, -4], [-2, -1, -3], [-2, -2, -2]]
>>> print(ex.statement(lang="en"))
Compute the factorization $PA = LU$ with partial pivoting of
\[ A = \begin{pmatrix} 0 & 1 & -4 \\ -2 & -1 & -3 \\ -2 & -2 & -2 \end{pmatrix}. \]
>>> P, L, U = ex.answer
>>> [[int(x) for x in row] for row in L], ex.answer.perm
([[1, 0, 0], [0, 1, 0], [1, -1, 1]], [1, 0, 2])
>>> print(ex.solution(lang="en").splitlines()[4])
& \quad R_{1} \leftrightarrow R_{2} \\
"""

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from ._steps import Steps, _check_lang
from .linalg import cholesky, det, lu, solve

__all__ = [
    "Exercise",
    "cholesky_exercise",
    "det_exercise",
    "exercise_sheet",
    "lu_exercise",
    "solve_exercise",
]

IntMatrix = list[list[int]]

_STATEMENTS = {
    "lu": {
        "fr": r"Calculer la factorisation $PA = LU$ avec pivot partiel de",
        "en": r"Compute the factorization $PA = LU$ with partial pivoting of",
    },
    "cholesky": {
        "fr": r"Montrer que la matrice suivante est symétrique définie positive en calculant sa factorisation de Cholesky $A = LL^T$ :",
        "en": r"Show that the following matrix is symmetric positive definite by computing its Cholesky factorization $A = LL^T$:",
    },
    "solve": {
        "fr": r"Résoudre le système $Ax = b$ par élimination de Gauss avec pivot partiel, où",
        "en": r"Solve the system $Ax = b$ by Gaussian elimination with partial pivoting, where",
    },
    "det": {
        "fr": r"Calculer le déterminant de la matrice suivante par élimination de Gauss :",
        "en": r"Compute the determinant of the following matrix by Gaussian elimination:",
    },
}

_SHEET = {
    "fr": {
        "title": "Exercices d'analyse numérique",
        "exercise": "Exercice",
        "solution": "Corrigé",
        "solutions": "Corrigés",
    },
    "en": {
        "title": "Numerical analysis exercises",
        "exercise": "Exercise",
        "solution": "Solution",
        "solutions": "Solutions",
    },
}


def _pmatrix(M: Iterable[Iterable[Any]]) -> str:
    rows = [" & ".join(str(x) for x in row) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"


@dataclass(frozen=True)
class Exercise:
    """An exercise with its data, its answer and its detailed solution.

    Attributes
    ----------
    kind : str
        ``"lu"``, ``"cholesky"``, ``"solve"`` or ``"det"``.
    A : list of list of int
        The matrix of the statement.
    b : list of int or None
        The right-hand side (``"solve"`` only).
    answer : object
        The result of the library on the exact data: an ``LUResult``, a
        ``CholeskyResult``, the solution ``x`` or the determinant.
    steps : Steps
        The detailed solution.
    """

    kind: str
    A: IntMatrix
    b: list[int] | None
    answer: Any
    steps: Steps

    def statement(self, *, lang: str = "fr") -> str:
        """Return the statement, in LaTeX."""
        _check_lang(lang)
        text = _STATEMENTS[self.kind][lang]
        if self.kind == "solve":
            assert self.b is not None
            b = _pmatrix([v] for v in self.b)
            return f"{text}\n\\[ A = {_pmatrix(self.A)}, \\quad b = {b}. \\]"
        return f"{text}\n\\[ A = {_pmatrix(self.A)}. \\]"

    def solution(self, *, lang: str = "fr") -> str:
        """Return the detailed solution, as a LaTeX ``align*`` environment."""
        return self.steps.to_latex(lang=lang, environment="align")

    def _repr_latex_(self) -> str:
        return self.statement(lang=self.steps.lang)


def _exact(M: IntMatrix) -> list[list[Fraction]]:
    return [[Fraction(x) for x in row] for row in M]


def _is_small_integer(x: Any, bound: int) -> bool:
    return bool(Fraction(x).denominator == 1 and abs(x) <= bound)


def _check(n: int, bound: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 6:
        raise ValueError(f"n must be an integer between 1 and 6, got {n!r}")
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 1:
        raise ValueError(f"bound must be a positive integer, got {bound!r}")


def _search(
    make: Callable[[random.Random], IntMatrix | None], seed: int | None
) -> IntMatrix:
    """Draw candidates until `make` accepts one (it returns None to reject)."""
    rng = random.Random(seed)
    for _ in range(10_000):
        A = make(rng)
        if A is not None:
            return A
    raise RuntimeError(
        "no suitable matrix found; try a larger bound"
    )  # pragma: no cover


def _nice_lu_matrix(n: int, bound: int, rng: random.Random) -> IntMatrix | None:
    """Draw ``A = Pᵀ L U`` with ``|l_ij| ≤ 1``; keep it if pivoted LU stays integer.

    The candidate is rejected unless pivoting swaps rows (for n ≥ 2) and, for
    n ≥ 3, some multiplier is non-zero. (With n = 2 both cannot hold: multipliers
    ±1 mean a tie between the two candidate pivots, and ties keep the top row.)
    """
    L = [
        [1 if i == j else (rng.randint(-1, 1) if j < i else 0) for j in range(n)]
        for i in range(n)
    ]
    U = [
        [
            rng.choice([v for v in range(-bound, bound + 1) if v])
            if i == j
            else (rng.randint(-bound, bound) if j > i else 0)
            for j in range(n)
        ]
        for i in range(n)
    ]
    LU = [
        [sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)
    ]
    A = [LU[p] for p in rng.sample(range(n), n)]
    limit = n * bound  # also bounds the entries of A, since |l_ij| <= 1
    res = lu(_exact(A))
    if n >= 2 and res.perm == list(range(n)):
        return None
    if n >= 3 and all(res.L[i][k] == 0 for i in range(n) for k in range(i)):
        return None
    if all(
        _is_small_integer(x, limit) for M in (res.L, res.U) for row in M for x in row
    ):
        return A
    return None


def lu_exercise(
    n: int = 3, *, seed: int | None = None, bound: int = 3, lang: str = "fr"
) -> Exercise:
    """Generate an LU exercise whose factors ``P, L, U`` have integer entries.

    Since partial pivoting gives multipliers ``|l_ij| ≤ 1``, an integer `L` has
    entries in {-1, 0, 1}; the rows of `A` are shuffled so that pivoting is
    needed. Every candidate is checked by running :func:`~numanalysis.lu` in exact
    arithmetic.

    Parameters
    ----------
    n : int, optional
        Size of the matrix, from 1 to 6.
    seed : int, optional
        Seed of the random generator, for reproducible exercises.
    bound : int, optional
        Bound of the entries of `U` (those of `A` stay below ``n * bound``).
    lang : {"fr", "en"}, optional
        Default language of the solution.

    Returns
    -------
    Exercise
        ``answer`` is the ``LUResult`` of the exact matrix.
    """
    _check(n, bound)
    _check_lang(lang)
    A = _search(lambda rng: _nice_lu_matrix(n, bound, rng), seed)
    res = lu(_exact(A), steps=True, lang=lang)
    assert res.steps is not None
    return Exercise("lu", A, None, res, res.steps)


def cholesky_exercise(
    n: int = 3, *, seed: int | None = None, bound: int = 3, lang: str = "fr"
) -> Exercise:
    """Generate a Cholesky exercise whose factor `L` has integer entries.

    The matrix is ``A = L Lᵀ`` for a random integer lower triangular `L` with a
    positive diagonal, so its Cholesky factor is exactly `L`.

    Parameters
    ----------
    n : int, optional
        Size of the matrix, from 1 to 6.
    seed : int, optional
        Seed of the random generator, for reproducible exercises.
    bound : int, optional
        Bound of the entries of `L`.
    lang : {"fr", "en"}, optional
        Default language of the solution.

    Returns
    -------
    Exercise
        ``answer`` is the ``CholeskyResult`` of the exact matrix.

    Examples
    --------
    >>> ex = cholesky_exercise(2, seed=0, lang="en")
    >>> ex.A, [[int(x) for x in row] for row in ex.answer.L]
    ([[4, 6], [6, 13]], [[2, 0], [3, 2]])
    """
    _check(n, bound)
    _check_lang(lang)

    def make(rng: random.Random) -> IntMatrix:
        L = [
            [
                rng.randint(1, bound)
                if i == j
                else (rng.randint(-bound, bound) if j < i else 0)
                for j in range(n)
            ]
            for i in range(n)
        ]
        return [
            [sum(L[i][k] * L[j][k] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]

    A = _search(make, seed)
    res = cholesky(_exact(A), steps=True, lang=lang)
    assert res.steps is not None
    return Exercise("cholesky", A, None, res, res.steps)


def solve_exercise(
    n: int = 3, *, seed: int | None = None, bound: int = 3, lang: str = "fr"
) -> Exercise:
    """Generate a linear system ``A x = b`` with an integer solution.

    `A` is built as in :func:`lu_exercise`, so every intermediate value of the
    elimination and of both substitutions is an integer; `x` has entries in
    ``[-bound, bound]``.

    Parameters
    ----------
    n : int, optional
        Size of the system, from 1 to 6.
    seed : int, optional
        Seed of the random generator, for reproducible exercises.
    bound : int, optional
        Bound of the entries of `U` and of the solution.
    lang : {"fr", "en"}, optional
        Default language of the solution.

    Returns
    -------
    Exercise
        ``answer`` is the solution ``x`` (a list of ``Fraction``).
    """
    _check(n, bound)
    _check_lang(lang)
    rng = random.Random(seed)
    A = _search(lambda r: _nice_lu_matrix(n, bound, r), rng.randrange(2**32))
    x = [rng.choice([v for v in range(-bound, bound + 1) if v]) for _ in range(n)]
    b = [sum(a * v for a, v in zip(row, x, strict=True)) for row in A]
    value, steps = solve(_exact(A), b, steps=True, lang=lang)
    return Exercise("solve", A, b, value, steps)


def det_exercise(
    n: int = 3, *, seed: int | None = None, bound: int = 3, lang: str = "fr"
) -> Exercise:
    """Generate a determinant exercise with integer pivots.

    Parameters
    ----------
    n : int, optional
        Size of the matrix, from 1 to 6.
    seed : int, optional
        Seed of the random generator, for reproducible exercises.
    bound : int, optional
        Bound of the pivots.
    lang : {"fr", "en"}, optional
        Default language of the solution.

    Returns
    -------
    Exercise
        ``answer`` is the determinant (a ``Fraction`` with denominator 1).
    """
    _check(n, bound)
    _check_lang(lang)
    A = _search(lambda rng: _nice_lu_matrix(n, bound, rng), seed)
    value, steps = det(_exact(A), steps=True, lang=lang)
    return Exercise("det", A, None, value, steps)


def exercise_sheet(
    exercises: Iterable[Exercise],
    *,
    lang: str = "fr",
    title: str | None = None,
    solutions: str = "end",
) -> str:
    r"""Assemble exercises into a complete LaTeX document.

    The document only needs ``amsmath`` and compiles with ``pdflatex``.

    Parameters
    ----------
    exercises : iterable of Exercise
        The exercises, in order.
    lang : {"fr", "en"}, optional
        Language of the statements and solutions.
    title : str, optional
        Title of the sheet.
    solutions : {"end", "after", "none"}, optional
        Put the solutions at the end of the sheet, after each exercise, or omit
        them.

    Returns
    -------
    str
        The LaTeX source of the document.

    Examples
    --------
    >>> tex = exercise_sheet([lu_exercise(seed=1), cholesky_exercise(seed=2)])
    >>> tex.startswith(r"\documentclass")
    True
    """
    _check_lang(lang)
    if solutions not in ("end", "after", "none"):
        raise ValueError(
            f"solutions must be 'end', 'after' or 'none', got {solutions!r}"
        )
    words = _SHEET[lang]
    items = list(exercises)
    body: list[str] = []
    for i, ex in enumerate(items, 1):
        body.append(
            f"\\subsection*{{{words['exercise']} {i}}}\n{ex.statement(lang=lang)}"
        )
        if solutions == "after":
            body.append(f"\\paragraph{{{words['solution']}}}\n{ex.solution(lang=lang)}")
    if solutions == "end":
        body.append(f"\\section*{{{words['solutions']}}}")
        for i, ex in enumerate(items, 1):
            body.append(
                f"\\subsection*{{{words['exercise']} {i}}}\n{ex.solution(lang=lang)}"
            )
    return (
        "\\documentclass[a4paper,11pt]{article}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{amsmath}\n"
        "\\allowdisplaybreaks\n"
        f"\\title{{{title or words['title']}}}\n"
        "\\date{}\n"
        "\\begin{document}\n"
        "\\maketitle\n" + "\n\n".join(body) + "\n\\end{document}\n"
    )
