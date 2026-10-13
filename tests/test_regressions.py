"""Regression tests for the bugs of numanalysis 0.1.1 (see ROADMAP.md, v0.2.0)."""

import subprocess
import sys
from importlib.metadata import metadata
from pathlib import Path
from typing import Any

import pytest

import numanalysis
from linalg_helpers import matmul
from numanalysis import cholesky, is_positive_definite, lu
from numanalysis.linalg import _cholesky, _lu


def run_python(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_bug1_import_cholesky_outside_the_repository(tmp_path: Path) -> None:
    proc = run_python(
        "from numanalysis import cholesky; print(cholesky([[4]]).L)", tmp_path
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[[2.0]]"


def test_bug2_import_has_no_side_effects(tmp_path: Path) -> None:
    proc = run_python(
        "import numanalysis, numanalysis.linalg, numanalysis.exceptions", tmp_path
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert proc.stderr == ""


def test_bug3_permutation_matrix_needs_pivoting() -> None:
    P, L, U = lu([[0, 1], [1, 0]])
    assert matmul(L, U) == matmul(P, [[0, 1], [1, 0]])


def test_bug3_tiny_pivot_is_avoided() -> None:
    A: list[list[float]] = [[1e-20, 1], [1, 1]]
    P, L, U = lu(A)
    assert matmul(L, U) == matmul(P, A)
    assert P == [[0.0, 1.0], [1.0, 0.0]]


def test_bug4_positive_definite_reads_the_whole_matrix() -> None:
    # xᵀAx = -98 for x = (1, -1)
    assert not is_positive_definite([[1, 100], [0, 1]])


def test_bug5_lu_validates_and_factorizes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = _lu.as_square_matrix  # type: ignore[attr-defined]

    def spy(A: Any, **kwargs: Any) -> Any:
        calls.append(A)
        return original(A, **kwargs)

    monkeypatch.setattr(_lu, "as_square_matrix", spy)
    res = lu([[4, 3], [6, 3]])
    assert len(calls) == 1
    assert res.U[0] == [6.0, 3.0]


def test_bug5_cholesky_factorizes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    sqrt_calls = []
    original = _cholesky._sqrt

    def spy(x: Any) -> Any:
        sqrt_calls.append(x)
        return original(x)

    monkeypatch.setattr(_cholesky, "_sqrt", spy)
    cholesky([[4, 2, 0], [2, 5, 1], [0, 1, 2]])
    assert len(sqrt_calls) == 3


def test_bug6_output_types_are_consistent() -> None:
    P, L, U = lu([[4, 3], [6, 3]])
    for M in (P, L, U):
        assert all(type(x) is float for row in M for x in row)
    L_chol = cholesky([[4, 2], [2, 5]]).L
    assert all(type(x) is float for row in L_chol for x in row)


def test_bug7_metadata() -> None:
    meta = metadata("numanalysis")
    assert meta["Requires-Python"] == ">=3.11"
    assert meta["Version"] == numanalysis.__version__
    classifiers = meta.get_all("Classifier") or []
    assert "Programming Language :: Python :: 3" in classifiers
    assert not any(":: Python :: 2" in c for c in classifiers)
    assert meta["Summary"] != "Package for LU decomp"
    assert meta["License-Expression"] == "MIT"
