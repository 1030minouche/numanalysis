"""Sphinx configuration of the numanalysis documentation."""

import numanalysis

project = "numanalysis"
author = "Amîn"
copyright = "2026, Amîn"
release = numanalysis.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "jupyter_execute", ".jupyter_cache"]

# NumPy-style docstrings.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# MyST and notebooks: notebooks are executed at every build and must not fail.
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3
nb_execution_mode = "force"
nb_execution_timeout = 300
nb_execution_raise_on_error = True
nb_merge_streams = True

html_theme = "furo"
html_title = f"numanalysis {release}"
html_theme_options = {
    "source_repository": "https://github.com/1030minouche/numanalysis/",
    "source_branch": "main",
    "source_directory": "docs/",
}
