"""Text normalization for audiobook conversion.

Cleans and normalizes text extracted from ebooks before chunking.
"""

from __future__ import annotations

import re

# Patterns for common navigation noise in ebooks
_NOISE_PATTERNS: list[re.Pattern[str]] = [
    # Page numbers: "Page 12", "p. 42", "— 123 —"
    re.compile(r"^[-—–]?\s*[Pp]age\s+\d+\s*[-—–]?$"),
    re.compile(r"^[-—–]?\s*p\.\s*\d+\s*[-—–]?$"),
    re.compile(r"^[-—–]\s*\d+\s*[-—–]$"),
    # Roman numerals often used in prefaces
    re.compile(r"^[ivxlcdmIVXLCDM]+$"),
    # Standalone "Contents" / "Table of Contents" / "Índice"
    re.compile(
        r"^(Contents|Table of Contents|Índice|Índice de contenidos)$",
        re.IGNORECASE,
    ),
    # Standalone "Chapter N" / "Capítulo N" without further text
    re.compile(
        r"^(Chapter|Capítulo|Cap\.?)\s+\d+$",
        re.IGNORECASE,
    ),
    # Copyright / publisher boilerplate lines
    re.compile(r"^Copyright\s+©", re.IGNORECASE),
    re.compile(r"^All rights reserved\.?$", re.IGNORECASE),
    # Separator lines: "***", "---", "• • •"
    re.compile(r"^[\s*•\-]{3,}$"),
]


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace while preserving paragraph structure.

    - Multiple spaces/tabs become a single space
    - Multiple newlines become a single newline
    - Leading/trailing whitespace is stripped
    """
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    # Collapse multiple spaces into one
    text = re.sub(r"[ ]{2,}", " ", text)
    # Collapse multiple newlines into one
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def remove_noise(text: str) -> str:
    """Remove common navigation and boilerplate noise lines."""
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if any(p.match(stripped) for p in _NOISE_PATTERNS):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def normalize(text: str) -> str:
    """Apply full normalization pipeline to text."""
    text = normalize_whitespace(text)
    text = remove_noise(text)
    # Final cleanup: strip each line and remove empty leading/trailing lines
    lines = text.splitlines()
    lines = [ln.strip() for ln in lines]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)
