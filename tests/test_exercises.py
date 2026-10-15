import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from linalg_helpers import matmul, transpose
from numanalysis import det as det_
from numanalysis._steps import Formula
from numanalysis.exercises import (
    Exercise,
    cholesky_exercise,
    det_exercise,
    exercise_sheet,
    lu_exercise,
    solve_exercise,
)


def is_integer_matrix(M: Any) -> bool:
    return all(Fraction(x).denominator == 1 for row in M for x in row)


seeds = st.integers(0, 10**6)
sizes = st.integers(1, 5)


class TestGenerators:
    @settings(max_examples=40, deadline=None)
    @given(sizes, seeds)
    def test_lu_comes_out_nicely(self, n: int, seed: int) -> None:
        ex = lu_exercise(n, seed=seed)
        P, L, U = ex.answer
        assert is_integer_matrix(L)
        assert is_integer_matrix(U)
        assert matmul(P, ex.A) == matmul(L, U)
        assert all(U[k][k] != 0 for k in range(n))
        if n >= 2:
            assert ex.answer.perm != list(range(n))  # pivoting is needed

    @settings(max_examples=40, deadline=None)
    @given(sizes, seeds)
    def test_cholesky_comes_out_nicely(self, n: int, seed: int) -> None:
        ex = cholesky_exercise(n, seed=seed)
        L = ex.answer.L
        assert is_integer_matrix(L)
        assert matmul(L, transpose(L)) == ex.A

    @settings(max_examples=40, deadline=None)
    @given(sizes, seeds)
    def test_solve_comes_out_nicely(self, n: int, seed: int) -> None:
        ex = solve_exercise(n, seed=seed, bound=2)
        assert ex.b is not None
        x = ex.answer
        assert all(Fraction(v).denominator == 1 and 0 < abs(v) <= 2 for v in x)
        assert [sum(a * v for a, v in zip(row, x, strict=True)) for row in ex.A] == ex.b
        # Every value computed in the solution steps is an integer.
        values = [step.value for step in ex.steps if isinstance(step, Formula)]
        assert values
        assert all(Fraction(v).denominator == 1 for v in values)

    @settings(max_examples=20, deadline=None)
    @given(sizes, seeds)
    def test_det(self, n: int, seed: int) -> None:
        ex = det_exercise(n, seed=seed)
        assert ex.answer == det_([[Fraction(x) for x in row] for row in ex.A])
        assert Fraction(ex.answer).denominator == 1
        assert ex.answer != 0

    def test_entries_stay_small(self) -> None:
        ex = lu_exercise(6, seed=0, bound=1)
        assert max(abs(x) for row in ex.A for x in row) <= 6

    def test_reproducible(self) -> None:
        assert lu_exercise(4, seed=7).A == lu_exercise(4, seed=7).A
        assert solve_exercise(3, seed=7).b == solve_exercise(3, seed=7).b

    @pytest.mark.parametrize("bad", [0, 7, 2.0, True])
    def test_bad_size(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="between 1 and 6"):
            lu_exercise(bad)

    @pytest.mark.parametrize("bad", [0, -1, 1.5])
    def test_bad_bound(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="bound"):
            cholesky_exercise(2, bound=bad)

    def test_bad_lang(self) -> None:
        with pytest.raises(ValueError, match="lang"):
            det_exercise(2, lang="de")


class TestRendering:
    def test_statements(self) -> None:
        ex = solve_exercise(2, seed=3)
        fr = ex.statement()
        assert fr.startswith("Résoudre le système $Ax = b$")
        assert r"b = \begin{pmatrix}" in fr
        assert ex.statement(lang="en").startswith("Solve the system")
        assert cholesky_exercise(2, seed=1).statement(lang="en").startswith("Show that")
        assert det_exercise(2, seed=1).statement().startswith("Calculer le déterminant")
        assert ex._repr_latex_() == fr

    def test_solution(self) -> None:
        ex = lu_exercise(3, seed=5, lang="en")
        sol = ex.solution(lang="fr")
        assert sol.startswith("\\begin{align*}")
        assert "Étape 1" in sol
        assert isinstance(ex, Exercise)

    def test_sheet(self) -> None:
        exs = [lu_exercise(seed=1), cholesky_exercise(seed=2)]
        tex = exercise_sheet(exs, lang="en", title="Week 3")
        assert r"\title{Week 3}" in tex
        assert tex.count(r"\subsection*{Exercise") == 4  # statements + solutions
        assert r"\section*{Solutions}" in tex
        after = exercise_sheet(exs, solutions="after")
        assert r"\paragraph{Corrigé}" in after
        none = exercise_sheet(exs, solutions="none")
        assert "align*" not in none
        assert r"\title{Exercices d'analyse numérique}" in none

    def test_sheet_errors(self) -> None:
        with pytest.raises(ValueError, match="solutions"):
            exercise_sheet([], solutions="inline")
        with pytest.raises(ValueError, match="lang"):
            exercise_sheet([], lang="it")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="needs pdflatex")
@pytest.mark.parametrize("lang", ["fr", "en"])
def test_sheet_compiles(tmp_path: Path, lang: str) -> None:
    exs = [
        lu_exercise(4, seed=1, lang=lang),
        cholesky_exercise(3, seed=2, lang=lang),
        solve_exercise(4, seed=3, lang=lang),
        det_exercise(3, seed=4, lang=lang),
    ]
    (tmp_path / "sheet.tex").write_text(
        exercise_sheet(exs, lang=lang), encoding="utf-8"
    )
    run = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "sheet.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stdout[-3000:]
    assert (tmp_path / "sheet.pdf").exists()
