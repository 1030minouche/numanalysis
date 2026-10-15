import shutil
import subprocess
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings

from linalg_helpers import square_int_matrices
from numanalysis._steps import (
    Determinant,
    Eliminate,
    Formula,
    Header,
    MatrixSnapshot,
    Note,
    Pivot,
    Steps,
    Swap,
    VectorSnapshot,
    _Renderer,
)
from numanalysis.exceptions import (
    InexactResultError,
    NotPositiveDefiniteError,
    SingularMatrixError,
    ZeroPivotError,
)
from numanalysis.linalg import (
    Traced,
    back_substitution,
    cholesky,
    det,
    forward_substitution,
    inv,
    ldlt,
    lu,
    solve,
)

F = Fraction


def r(fmt: str = "text", lang: str = "en", digits: int = 6) -> _Renderer:
    return _Renderer(fmt, lang, digits)


class TestNumbers:
    @pytest.mark.parametrize(
        ("x", "text", "latex"),
        [
            (F(1, 3), "1/3", r"\frac{1}{3}"),
            (F(-1, 3), "-1/3", r"-\frac{1}{3}"),
            (F(4), "4", "4"),
            (2.0, "2", "2"),
            (1 / 3, "0.333333", "0.333333"),
            (Decimal("1.50"), "1.50", "1.50"),
            (1 - 2j, "(1 - 2j)", r"\left(1 - 2i\right)"),
            (3, "3", "3"),
            ([F(1, 2), 1], "(1/2, 1)", r"\left(\frac{1}{2}, 1\right)"),
        ],
    )
    def test_formats(self, x: Any, text: str, latex: str) -> None:
        assert r().num(x) == text
        assert r("latex").num(x) == latex

    def test_digits(self) -> None:
        assert r(digits=3).num(1 / 3) == "0.333"

    def test_mpmath(self) -> None:
        mpmath = pytest.importorskip("mpmath")
        assert r(digits=4).num(mpmath.mpf(1) / 3) == "0.3333"
        assert r(digits=3).num(mpmath.mpc(1, 1) / 3) == "(0.333 + 0.333j)"

    def test_paren(self) -> None:
        assert r().paren(-2.0) == "(-2)"
        assert r().paren(F(1, 2)) == "(1/2)"
        assert r("latex").paren(F(1, 2)) == r"\frac{1}{2}"
        assert r("latex").paren(-1.0) == r"\left(-1\right)"

    def test_subscripts(self) -> None:
        assert r().sub("l", 1, 0) == "l₂₁"
        assert r("latex").sub("l", 1, 0) == "l_{21}"
        assert r().sub("a", 9, 1) == "a₁₀,₂"
        assert r("latex").sub("a", 9, 1) == "a_{10,2}"


class TestStepRendering:
    def test_eliminate(self) -> None:
        assert Eliminate(1, 0, 2.0).render(r()) == "R₂ ← R₂ − 2·R₁"
        assert Eliminate(1, 0, -2.0).render(r()) == "R₂ ← R₂ + 2·R₁"
        assert Eliminate(1, 0, 1).render(r(lang="fr")) == "L₂ ← L₂ − L₁"
        assert Eliminate(2, 0, F(1, 2)).render(r("latex")) == (
            r"R_{3} \leftarrow R_{3} - \frac{1}{2}\cdot L_{1}".replace("L", "R")
        )
        assert Eliminate(1, 0, 1j).render(r()) == "R₂ ← R₂ − (0 + 1j)·R₁"

    def test_swap(self) -> None:
        assert Swap("row", 0, 2).render(r(lang="fr")) == "L₁ ↔ L₃"
        assert Swap("col", 0, 1).render(r("latex")) == r"C_{1} \leftrightarrow C_{2}"

    def test_pivot(self) -> None:
        text = Pivot(2, 0, 8.0).render(r())
        assert text == "pivot: the largest |a_i1| of column 1 is 8, in row 3"
        latex = Pivot(1, 1, F(-3, 4), complete=True).render(r("latex", "fr"))
        assert r"-\frac{3}{4}" in latex
        assert r"\text{" in latex

    def test_formula(self) -> None:
        assert Formula("x", (0,), 3, None, None, 3).render(r()) == "x₁ = 3"
        assert Formula("l", (1, 0), 2, None, 2, 1).render(r()) == "l₂₁ = 2 / 2 = 1"
        assert (
            Formula("l", (1, 1), 5, 1, None, 2, sqrt=True).render(r())
            == "l₂₂ = √(5 − 1) = 2"
        )
        assert (
            Formula("x", (1,), F(7, 2), None, F(5, 2), F(7, 5)).render(r())
            == "x₂ = (7/2) / (5/2) = 7/5"
        )
        assert (
            Formula("x", (0,), 3, -1, 2, 2).render(r("latex"))
            == r"x_{1} = \left(3 - \left(-1\right)\right) / 2 = 2"
        )

    def test_determinant(self) -> None:
        step = Determinant(-1, (3.0, 2.0), -6.0)
        assert step.render(r()) == "det A = (−1) · 3 · 2 = -6"
        assert (
            step.render(r("latex")) == r"\det A = \left(-1\right) \cdot 3 \cdot 2 = -6"
        )

    def test_snapshots(self) -> None:
        snap = MatrixSnapshot(((1, 2), (3, 4)), name="U")
        assert snap.render(r()) == "U =\n  [1  2]\n  [3  4]"
        assert snap.render(r("latex")).startswith(r"U = \begin{pmatrix} 1 & 2")
        assert "```" in snap.render(r("markdown"))
        labelled = MatrixSnapshot(((1,),), "after_step", (1,))
        assert labelled.render(r(lang="fr")).startswith("Matrice après l'étape 1 :")
        vec = VectorSnapshot("x", (1, 2), "solution")
        assert vec.render(r()) == "Solution: x = (1, 2)"
        assert vec.render(r(lang="fr")) == "Solution : x = (1, 2)"
        assert r"\begin{pmatrix} 1 \\ 2" in vec.render(r("latex"))

    def test_header_and_note(self) -> None:
        assert Header("step", (2,)).render(r("markdown", "fr")) == "**Étape 2**"
        assert Header("step", (2,)).render(r("latex")) == r"\textbf{Step 2}"
        assert Note("zero_column").render(r()).startswith("the column is zero")
        assert Note("lu_rule").render(r("latex")).startswith(r"\text{Gaussian")


class TestSteps:
    def test_sequence_protocol(self) -> None:
        steps = lu([[2, 1], [4, 3]], steps=True).steps
        assert steps is not None
        assert isinstance(steps[0], Note)
        assert isinstance(steps[1:3], list)
        assert len(steps) == len(list(steps))
        assert any(isinstance(s, Swap) for s in steps)

    def test_languages_and_formats(self) -> None:
        steps = lu([[F(2), 1], [4, 3]], steps=True).steps
        assert steps is not None
        assert "Step 1" in str(steps)
        assert "Étape 1" in steps.to_text(lang="fr")
        assert "Étape 1" in str(steps.with_lang("fr"))
        md = steps.to_markdown(lang="fr")
        assert "**Étape 1**" in md
        assert "\n\n\n" not in md
        tex = steps.to_latex()
        assert tex.startswith("\\begin{array}{l}")
        assert tex.endswith("\\end{array}")
        assert steps._repr_latex_() == "$$" + tex + "$$"

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="lang"):
            Steps(lang="de")
        steps = Steps()
        with pytest.raises(ValueError, match="lang"):
            steps.to_text(lang="de")
        with pytest.raises(ValueError, match="fmt"):
            steps.render("html")
        with pytest.raises(ValueError, match="environment"):
            steps.to_latex(environment="tabular")
        with pytest.raises(ValueError, match="lang"):
            lu([[1]], steps=True, lang="es")

    def test_no_steps_by_default(self) -> None:
        res = lu([[1, 2], [3, 4]])
        assert res.steps is None
        assert res._repr_latex_() is None


class TestAlgorithms:
    def test_lu_partial(self) -> None:
        steps = lu([[1, 2], [3, 4]], steps=True).steps
        text = str(steps)
        assert "R₁ ↔ R₂" in text
        assert "R₂ ← R₂ − 0.333333·R₁" in text
        assert text.endswith("[0  0.666667]")

    def test_lu_complete(self) -> None:
        text = str(lu([[1, 2], [3, 4]], pivot="complete", steps=True).steps)
        assert "C₁ ↔ C₂" in text
        assert "remaining submatrix is 4" in text
        assert "Q =" in text

    def test_lu_zero_column(self) -> None:
        assert "nothing to eliminate" in str(lu([[0, 1], [0, 1]], steps=True).steps)
        text = str(lu([[0, 0], [0, 0]], pivot="complete", steps=True).steps)
        assert "remaining submatrix is zero" in text

    def test_cholesky(self) -> None:
        res = cholesky([[4, 2], [2, 5]], steps=True, lang="fr")
        text = str(res.steps)
        assert "Ligne 2" in text
        assert "l₂₂ = √(5 − 1) = 2" in text
        assert res._repr_latex_() is not None

    def test_cholesky_failure_keeps_steps(self) -> None:
        with pytest.raises(NotPositiveDefiniteError) as info:
            cholesky([[1, 2], [2, 1]], steps=True)
        assert info.value.steps is not None
        assert "not positive definite" in str(info.value.steps)
        with pytest.raises(NotPositiveDefiniteError) as info:
            cholesky([[-1]])
        assert info.value.steps is None

    def test_cholesky_inexact_keeps_steps(self) -> None:
        with pytest.raises(InexactResultError) as info:
            cholesky([[F(4), 2], [2, F(3)]], steps=True)
        assert info.value.steps is not None
        assert "l₂₁ = 2 / 2 = 1" in str(info.value.steps)

    def test_ldlt(self) -> None:
        text = str(ldlt([[F(4), 2], [2, 5]], steps=True).steps)
        assert "d₁ = 4" in text
        assert "l₂₁ = 2 / 4 = 1/2" in text
        assert "d₂ = 5 − 1 = 4" in text
        assert "D =" in text

    def test_ldlt_failure_keeps_steps(self) -> None:
        with pytest.raises(ZeroPivotError) as info:
            ldlt([[1, 1], [1, 1]], steps=True)
        assert info.value.steps is not None
        assert "zero pivot" in str(info.value.steps)
        with pytest.raises(ZeroPivotError) as info:
            ldlt([[0]])
        assert info.value.steps is None

    def test_solve(self) -> None:
        res = solve([[F(2), 1], [1, 3]], [3, 5], steps=True)
        assert isinstance(res, Traced)
        x, steps = res
        assert x == [F(4, 5), F(7, 5)]
        text = str(steps)
        assert "Forward substitution: L y = P b" in text
        assert "x₁ = (3 − (7/5)) / 2 = 4/5" in text
        assert text.endswith("x = (4/5, 7/5)")
        assert res._repr_latex_() == steps._repr_latex_()

    def test_solve_complete_and_matrix_rhs(self) -> None:
        x, steps = solve(
            [[F(1), 2], [3, 4]], [[1, 0], [0, 1]], pivot="complete", steps=True
        )
        text = str(steps)
        assert "x = Q z" in text
        assert "P B =" in text
        assert "z₂ = " in text
        assert "X =" in text
        assert x == inv([[F(1), 2], [3, 4]])

    def test_solve_singular_keeps_steps(self) -> None:
        with pytest.raises(SingularMatrixError) as info:
            solve([[1, 2], [2, 4]], [1, 1], steps=True)
        assert info.value.steps is not None
        assert "singular" in str(info.value.steps)

    def test_substitutions(self) -> None:
        x, steps = forward_substitution([[2, 0], [1, 1]], [4, 3], steps=True, lang="fr")
        assert x == [2.0, 1.0]
        assert str(steps).startswith("Descente : L x = b")
        y, steps = back_substitution([[F(2), 1], [0, 4]], [[3], [4]], steps=True)
        assert y == [[1], [1]]
        assert "X =" in str(steps)

    def test_substitution_singular_keeps_steps(self) -> None:
        with pytest.raises(SingularMatrixError) as info:
            back_substitution([[1, 1], [0, 0]], [1, 1], steps=True)
        assert info.value.steps is not None

    def test_det(self) -> None:
        d, steps = det([[F(1), 2], [3, 4]], steps=True)
        assert d == -2
        assert str(steps).endswith("det A = (−1) · 3 · (2/3) = -2")
        d2, steps2 = det([[1, 2], [3, 4]], pivot="complete", steps=True)
        assert d2 == -2.0
        assert "C₁ ↔ C₂" in str(steps2)

    def test_inv(self) -> None:
        X, steps = inv([[F(2), 1], [1, 1]], steps=True, lang="fr")
        assert X == [[1, -1], [-1, 2]]
        text = str(steps)
        assert text.startswith("Inverse : on résout A X = I")
        assert "A⁻¹ =" in text

    @settings(max_examples=30)
    @given(square_int_matrices(4, 9))
    def test_property_steps_do_not_change_results(self, A: list[list[int]]) -> None:
        Fr = [[F(x) for x in row] for row in A]
        for pivot in ("partial", "complete"):
            plain = lu(Fr, pivot=pivot)
            traced = lu(Fr, pivot=pivot, steps=True)
            assert list(plain) == list(traced)
            assert det(Fr, pivot=pivot) == det(Fr, pivot=pivot, steps=True).value
            for fmt in ("text", "markdown", "latex"):
                assert traced.steps is not None
                assert traced.steps.render(fmt, lang="fr")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="needs pdflatex")
def test_latex_output_compiles(tmp_path: Path) -> None:
    traces = [
        lu([[F(2), 1, 1], [4, 3, 3], [8, 7, 9]], steps=True).steps,
        lu([[1, 2], [3, 4]], pivot="complete", steps=True, lang="fr").steps,
        cholesky([[4, 2], [2, 5]], steps=True).steps,
        ldlt([[F(4), 2], [2, 5]], steps=True, lang="fr").steps,
        solve([[F(2), 1], [1, 3]], [3, 5], steps=True).steps,
        solve([[1j, 1], [1, 2]], [1, 0], pivot="complete", steps=True, lang="fr").steps,
        det([[F(1), 2], [3, 4]], steps=True).steps,
        inv([[F(2), 1], [1, 1]], steps=True).steps,
    ]
    body = "\n".join(r"\[" + t.to_latex() + r"\]" for t in traces if t is not None)
    doc = (
        "\\documentclass{article}\n\\usepackage[T1]{fontenc}\n\\usepackage{amsmath}\n"
        "\\begin{document}\n" + body + "\n\\end{document}\n"
    )
    (tmp_path / "steps.tex").write_text(doc, encoding="utf-8")
    run = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "steps.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stdout[-3000:]
