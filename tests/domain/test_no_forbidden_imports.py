"""Verify domain layer has no forbidden imports."""

from __future__ import annotations

from pathlib import Path

import pytest

DOMAIN_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "audiobook_engine"
    / "domain"
)

FORBIDDEN_PREFIXES = (
    "kokoro",
    "ffmpeg",
    "fastapi",
    "uvicorn",
    "calibre",
    "starlette",
    "httpx",
    "requests",
)


def _domain_py_files() -> list[Path]:
    return sorted(DOMAIN_DIR.rglob("*.py"))


@pytest.mark.parametrize(
    "py_file",
    _domain_py_files(),
    ids=lambda p: str(p.relative_to(DOMAIN_DIR)),
)
def test_domain_module_no_forbidden_imports(py_file: Path) -> None:
    """Scan source file text for forbidden imports."""
    content = py_file.read_text()
    rel = str(py_file.relative_to(DOMAIN_DIR))

    for line_no, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for prefix in FORBIDDEN_PREFIXES:
            if f"import {prefix}" in stripped or f"from {prefix}" in stripped:
                pytest.fail(
                    f"{rel}:{line_no} — forbidden import: {stripped}\n"
                    f"Domain layer must not depend on infrastructure."
                )
