"""Scaffold tests. These cover packaging and layout, not learning code."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import slackgnn
from slackgnn.paths import project_dir, repo_root
from slackgnn.runtime import (
    REQUIRED_PYTHON,
    format_interpreter_message,
    interpreter_is_supported,
    main,
)

_ROOT = repo_root()
_REQUIRED_DIRECTORIES = (
    "docs",
    "data",
    "results",
    "notebooks",
    "scripts",
    "src",
    "tests",
    "tcl",
)


def _requirement_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name = stripped.split(">=")[0].split("==")[0].split("<")[0].strip()
        names.add(name.replace("_", "-").lower())
    return names


def test_package_version_matches_project_metadata() -> None:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    assert metadata["project"]["version"] == slackgnn.__version__
    assert metadata["project"]["requires-python"] == ">=3.11,<3.12"


def test_requirements_match_project_metadata() -> None:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    declared = _requirement_names("\n".join(metadata["project"]["dependencies"]))
    declared.update(_requirement_names("\n".join(metadata["project"]["optional-dependencies"]["test"])))
    pinned = _requirement_names((_ROOT / "requirements.txt").read_text(encoding="utf-8"))
    assert pinned == declared
    assert {"torch", "torch-geometric", "networkx", "pytest"} <= pinned


def test_repository_directories_exist() -> None:
    for name in _REQUIRED_DIRECTORIES:
        assert project_dir(name).is_dir()


def test_unknown_project_directory_is_rejected() -> None:
    try:
        project_dir("models")
    except ValueError as exc:
        assert "models" in str(exc)
    else:
        raise AssertionError("project_dir accepted an unknown directory")


def test_required_python_constant() -> None:
    assert REQUIRED_PYTHON == (3, 11)
    assert interpreter_is_supported((3, 11))
    assert not interpreter_is_supported((3, 12))
    assert "Python 3.11" in format_interpreter_message((3, 11))
    assert "Python 3.12" in format_interpreter_message((3, 12))


def test_live_interpreter_is_python_311(capsys) -> None:
    assert sys.version_info[:2] == REQUIRED_PYTHON
    assert main() == 0
    captured = capsys.readouterr()
    assert "Python 3.11" in captured.out
    assert captured.err == ""


def test_py_typed_marker_is_packaged() -> None:
    marker = Path(slackgnn.__file__).resolve().parent / "py.typed"
    assert marker.is_file()
