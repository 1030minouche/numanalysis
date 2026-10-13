# numanalysis — project instructions

Talk to the maintainer in **French**. Write code, docstrings and commit messages in **English**.

@ROADMAP.md

## Vision

"Glass-box numerical analysis": every algorithm can run in exact or multiprecision arithmetic,
shows its steps, is rigorously verified, and is illustrated by a mechanics/aero example.
We do not compete with NumPy/SciPy on speed: SciPy is the test oracle.

## Design principles

- Core = 100 % standard library, zero runtime dependency. Extras: `plot` (matplotlib, numpy),
  `mp` (mpmath), `dev` (pytest, hypothesis, pytest-cov, numpy, scipy, ruff, mypy, pre-commit,
  build, twine), `docs`.
- From v0.3: algorithms generic over the scalar type (`float`, `Fraction`, `Decimal`, `complex`,
  mpmath); never an implicit `float()`. Square roots and tolerances live in `_scalar.py`
  (zero tolerance for exact types, relative for floats). Until then, `_promote` in
  `_validation.py` turns `int` into `float` and leaves other scalars untouched.
- Inputs: any sequence of sequences (lists, tuples, NumPy arrays), never mutated.
  Outputs: lists of lists with consistent scalar types.
- Every algorithm returns an unpackable result object (`P, L, U = lu(A)`) that also carries its
  trace (`.steps`) or convergence history.
- Dedicated exceptions in `numanalysis.exceptions` (`ShapeError`, `SingularMatrixError`,
  `NotPositiveDefiniteError`, …), all subclasses of `ValueError`.
- NumPy-style docstrings with a runnable doctest example and the complexity; type hints
  everywhere; `py.typed` shipped.
- Clarity over performance. Examples stay fast (ideally < 1 min in pure Python, else document it).

## Definition of done (each roadmap box)

Implementation + tests (nominal and edge cases, regression test for every fixed bug,
`hypothesis` properties, SciPy/NumPy comparison, `Fraction` test when relevant) + docstring +
export in the public API + "Unreleased" CHANGELOG entry + `ruff`, `mypy`, `pytest` green +
box ticked in `ROADMAP.md`.

## Git

- One branch per version from `main`: `dev/v0.2`, `dev/v0.3`, …
- Small atomic Conventional Commits: `feat(linalg): …`, `fix(cholesky): …`, `test(lu): …`,
  `docs:`, `ci:`, `build:`, `refactor:`, `chore(release):`.
- At least one commit per box; never more than ~300 diff lines without committing; every commit
  keeps tests green. Push the version branch after each finished box and at the end of a session.
- **No `Co-Authored-By` trailer and no "Generated with Claude Code" line**: the maintainer is
  the sole author of commits and PRs.
- Use `git mv` to move files. Forbidden: force-push to `main`, rewriting pushed history,
  committing generated files (`__pycache__/`, `dist/`, `build/`, `.venv/`, `docs/_build/`,
  `.coverage`, `.ipynb_checkpoints/`) or secrets.
- Environment: on this Windows machine git/gh are used from Git Bash. The parent folder
  `Projet_Pypi/` contains secret token files: never read, move or commit them.

## CI and publishing

- `ci.yml` (push + PR): ruff check, ruff format --check, mypy; pytest with doctests and
  coverage ≥ 90 % on Python 3.11–3.15; a "minimal" job (package + pytest + hypothesis only,
  optional-dependency tests skipped via `pytest.importorskip`); a "package" job (build, twine
  check, install the wheel in a clean venv and import it **outside the repo**).
- `publish.yml` (tag `v*`): build (+ tag == `__version__` check) → TestPyPI (env `testpypi`,
  `skip-existing`) → PyPI (env `pypi`, manual approval) → GitHub Release with the CHANGELOG
  section and `dist/` files. Trusted Publishing only (`id-token: write`), no PyPI token anywhere.
- `docs.yml` (from v0.3): build docs on PRs, deploy to GitHub Pages on `main`.
- Manual settings (PyPI/TestPyPI publishers, GitHub environments, Pages, Codecov) are done by
  the maintainer: stop, give exact steps, wait for confirmation.

## Release procedure (end of each version)

1. All boxes ticked, CI green on the branch.
2. CHANGELOG "Unreleased" → `[X.Y.0] - YYYY-MM-DD`; bump `__version__`;
   commit `chore(release): prepare vX.Y.0`; push.
3. `gh pr create` `dev/vX.Y` → `main`, title "Release vX.Y.0", body = CHANGELOG section.
4. **Stop**: wait for review. Merge only with approval via `gh pr merge --merge` (never squash).
5. On up-to-date `main`, create annotated tag `vX.Y.0`. **Stop** and ask
   « OK pour publier vX.Y.0 sur PyPI ? ». Push the tag only after an explicit yes.
6. `gh run watch`; the maintainer approves the `pypi` environment.
7. Check in a clean venv from a neutral folder: `pip install numanalysis==X.Y.0`, import, quick test.
8. Mark the version released in `ROADMAP.md`, create the next version branch.

A broken release is never deleted: fix it and publish `X.Y.1`.

## Commands

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash (.venv/bin on Unix)
pip install -e ".[dev]"
pre-commit install

ruff check . && ruff format --check .   # lint / format check (ruff format . to fix)
mypy                                    # type check (config in pyproject.toml)
pytest --cov=numanalysis                # tests + doctests + coverage (≥ 90 %)
python -m build && twine check dist/*   # build and check distributions
```

Docs (from v0.3): `pip install -e ".[docs]"` then `sphinx-build -W docs docs/_build/html`.

## Layout

```
src/numanalysis/
    __init__.py        # __version__, public re-exports
    exceptions.py
    _validation.py     # input normalisation (as_matrix, _promote)
    py.typed
    linalg/            # properties, lu, cholesky, result objects
tests/
```
