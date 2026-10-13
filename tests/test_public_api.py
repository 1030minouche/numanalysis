import re

import numanalysis
from numanalysis import linalg


def test_version_is_pep440() -> None:
    assert re.fullmatch(
        r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", numanalysis.__version__
    )


def test_all_names_exist() -> None:
    for module in (numanalysis, linalg):
        for name in module.__all__:
            assert hasattr(module, name), name


def test_top_level_reexports_linalg() -> None:
    assert numanalysis.lu is linalg.lu
    assert numanalysis.cholesky is linalg.cholesky
    assert numanalysis.ShapeError is numanalysis.exceptions.ShapeError
