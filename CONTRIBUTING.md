# Contributing to numanalysis

Thank you for your interest! Bug reports, documentation fixes and new algorithms are
all welcome. Issues and pull requests may be written in English or French.

## Development setup

```bash
git clone https://github.com/1030minouche/numanalysis
cd numanalysis
python -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
pip install -e ".[dev]"
pre-commit install
```

## Checks

Every pull request must keep these green (the CI runs them on Python 3.11 to 3.15):

```bash
ruff check . && ruff format --check .
mypy
pytest --cov=numanalysis         # tests, doctests (src and READMEs), coverage >= 90 %
```

## Guidelines

- **No runtime dependency** in the core: only the standard library. NumPy, SciPy and
  mpmath may be used in tests (skip with `pytest.importorskip`) or in optional extras.
- **Clarity first**: code is meant to be read by students. Prefer explicit loops and
  descriptive names over clever tricks.
- **Generic scalars**: do not call `float()` on user data; algorithms should work with
  `Fraction` whenever that makes sense.
- **Never modify the input**; accept any sequence of sequences; return lists of lists.
- **Return a result object** that unpacks naturally (`P, L, U = lu(A)`).
- **Raise the dedicated exceptions** from `numanalysis.exceptions`.
- **Docstrings** follow the NumPy convention and include a runnable example and the
  complexity. Public functions are fully type-annotated.

## Tests

A change is complete when it comes with:

- nominal and edge-case tests, and a regression test for every bug fixed;
- `hypothesis` property tests for algebraic identities (for example `P A = L U`);
- a comparison with SciPy or NumPy when an equivalent exists;
- a `Fraction` test when exact arithmetic is relevant.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/):
  `feat(linalg): ...`, `fix(cholesky): ...`, `test(lu): ...`, `docs: ...`, `ci: ...`.
- Keep commits small and focused, and add an entry to the `Unreleased` section of
  `CHANGELOG.md`.
- Open the pull request against `main` (or the current `dev/vX.Y` branch).

## Releases

Releases are published by the maintainer from a `vX.Y.Z` tag through the
`publish.yml` workflow (Trusted Publishing, no API token).
