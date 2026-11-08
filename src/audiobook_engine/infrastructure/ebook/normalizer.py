"""Text normalization for audiobook conversion.

Cleans and normalizes text extracted from ebooks before chunking.
"""

from __future__ import annotations

import re

# Prosody anchors preserved through normalization and chunking.
# They are internal sentinels (never narrated): the synthesis layer
# turns them into silence at the configured scene/title pauses.
SCENE_BREAK_MARKER: str = "\x00SCENE_BREAK\x00"
CHAPTER_TITLE_MARKER: str = "\x00CHAPTER_TITLE\x00"
PROSODY_MARKERS: frozenset[str] = frozenset({SCENE_BREAK_MARKER, CHAPTER_TITLE_MARKER})

# Section separator lines: "***", "---", "• • •" (become scene pauses).
_SCENE_SEPARATOR_RE = re.compile(r"^[\s*•\-\u2013\u2014]{3,}$")

# Standalone "Chapter N" / "Capítulo N" lines (become title pauses).
_CHAPTER_TITLE_RE = re.compile(
    r"^(Chapter|Capítulo|Cap\.?)\s+\d+$",
    re.IGNORECASE,
)


def is_prosody_marker(text: str) -> bool:
    """Return True if the text is a prosody anchor sentinel."""
    return text.strip() in PROSODY_MARKERS


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
    # Copyright / publisher boilerplate lines
    re.compile(r"^Copyright\s+©", re.IGNORECASE),
    re.compile(r"^All rights reserved\.?$", re.IGNORECASE),
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
    """Remove common navigation and boilerplate noise lines.

    Section separators and standalone chapter-title lines are not
    dropped: they are replaced by prosody anchors so the synthesis
    layer can apply the promised scene/title pauses.
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if _SCENE_SEPARATOR_RE.match(stripped):
            cleaned.append(SCENE_BREAK_MARKER)
            continue
        if _CHAPTER_TITLE_RE.match(stripped):
            cleaned.append(CHAPTER_TITLE_MARKER)
            continue
        if any(p.match(stripped) for p in _NOISE_PATTERNS):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def normalize_dialogue(text: str) -> str:
    """Normalize dialogue markers, quotes, and punctuation for TTS.

    - Converts guillemets (« ») and curly quotes to standard quotes
    - Normalizes Spanish dialogue dashes (- / –) to em-dash (—)
    - Fixes spacing around inverted marks (¿, ¡)
    - Standardizes ellipsis (... / …)
    """
    # Normalize guillemets and curly double quotes
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Normalize dialogue dashes at the start of a line/paragraph
    text = re.sub(r"(?m)^[-–]\s*", "— ", text)
    # Normalize dialogue dashes in incises/tags
    text = re.sub(r"\s+[-–]\s+", " — ", text)
    text = re.sub(r"\s+[-–]([.!?])", r" —\1", text)

    # Normalize spaces after inverted question/exclamation marks
    text = re.sub(r"¿\s+", "¿", text)
    text = re.sub(r"¡\s+", "¡", text)

    # Normalize ellipsis
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"(?:\.\s+){2,}\.", "...", text)

    return text


def normalize(text: str) -> str:
    """Apply full normalization pipeline to text."""
    text = normalize_whitespace(text)
    text = remove_noise(text)
    text = normalize_dialogue(text)
    # Final cleanup: strip each line and remove empty leading/trailing lines
    lines = text.splitlines()
    lines = [ln.strip() for ln in lines]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)
