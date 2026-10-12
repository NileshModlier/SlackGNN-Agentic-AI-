"""Interpreter checks for the SlackGNN toolchain."""

from __future__ import annotations

import sys

REQUIRED_PYTHON = (3, 11)


def interpreter_is_supported(version_info: tuple[int, int] | None = None) -> bool:
    """Return whether ``version_info`` is the required Python release."""
    if version_info is None:
        version_info = sys.version_info[:2]
    return version_info == REQUIRED_PYTHON


def format_interpreter_message(version_info: tuple[int, int] | None = None) -> str:
    """Describe whether the interpreter matches :data:`REQUIRED_PYTHON`."""
    if version_info is None:
        version_info = sys.version_info[:2]
    required = f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}"
    found = f"{version_info[0]}.{version_info[1]}"
    if interpreter_is_supported(version_info):
        return f"Python {found} matches the SlackGNN requirement ({required})."
    return f"SlackGNN requires Python {required}; this interpreter is Python {found}."


def main() -> int:
    """Print the interpreter check and return a process status code."""
    supported = interpreter_is_supported()
    stream = sys.stdout if supported else sys.stderr
    print(format_interpreter_message(), file=stream)
    return 0 if supported else 1
