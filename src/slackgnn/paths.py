"""Filesystem locations for a SlackGNN source checkout.

Scripts and tests resolve data, results, and tool directories here so
callers do not hard-code paths relative to the process working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIRECTORY_NAMES = (
    "docs",
    "data",
    "results",
    "notebooks",
    "scripts",
    "src",
    "tests",
    "tcl",
)


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the checkout root that contains ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "slackgnn").is_dir():
            return candidate
    raise FileNotFoundError(
        "SlackGNN repository root not found. Use a source checkout that contains pyproject.toml."
    )


def project_dir(name: str) -> Path:
    """Return a top-level project directory by its canonical name."""
    if name not in _DIRECTORY_NAMES:
        known = ", ".join(_DIRECTORY_NAMES)
        raise ValueError(f"Unknown project directory {name!r}. Expected one of: {known}.")
    return repo_root() / name


def data_dir() -> Path:
    return project_dir("data")


def results_dir() -> Path:
    return project_dir("results")


def docs_dir() -> Path:
    return project_dir("docs")


def notebooks_dir() -> Path:
    return project_dir("notebooks")


def scripts_dir() -> Path:
    return project_dir("scripts")


def tcl_dir() -> Path:
    return project_dir("tcl")
