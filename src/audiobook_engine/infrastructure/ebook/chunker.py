"""Sentence-aware deterministic text chunking for TTS.

Splits normalized text into chunks that respect sentence boundaries
and a configurable maximum character limit.
"""

from __future__ import annotations

import re

# Default maximum characters per chunk.
# Most TTS engines handle 200-500 characters well.
DEFAULT_MAX_CHARS: int = 500

# Sentence-ending punctuation followed by whitespace or end of string
_SENTENCE_END_RE = re.compile(r'([.!?]+[\u201d\u2019"»\u2014-]*)(?:\s+|$)')

# Abbreviations that should NOT trigger a sentence break
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "sra", "srta", "jr", "st",
        "vs", "etc", "inc", "ltd", "corp",
        "fig", "eq", "vol", "no", "ch", "pp", "ed", "trans",
        "pág", "págs", "art", "d", "dña",
    }
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at natural boundaries.

    Handles dialogue tags, trailing quotes/dashes, and abbreviations.
    """
    if not text.strip():
        return []

    sentences: list[str] = []
    last_end = 0

    for match in _SENTENCE_END_RE.finditer(text):
        if match.start() == last_end:
            continue

        punct_part = match.group(1)
        before = text[last_end : match.start() + len(punct_part)]
        last_word_match = re.search(r"(\S+)$", before.rstrip())
        if last_word_match:
            word = last_word_match.group(1).rstrip('.!?"»”’\u2014-')
            if word.lower() in _ABBREVIATIONS:
                continue

        # Check for single-letter abbreviations like "U.S.A."
        before_period = text[last_end : match.start()]
        if re.search(r"[A-Z]\.$", before_period):
            continue

        # Avoid breaking if next segment is a dialogue tag (e.g. "—preguntó")
        after_pos = match.end()
        remaining_text = text[after_pos:].lstrip()
        if remaining_text and (
            re.match(r"^(?:—|-|–)\s*[a-záéíóúñ]", remaining_text)
            or re.match(r"^[a-záéíóúñ]", remaining_text)
        ):
            continue

        sentence = text[last_end : match.start() + len(punct_part)].strip()
        if sentence:
            sentences.append(sentence)
        last_end = match.end()

    # Don't forget the last part
    remaining = text[last_end:].strip()
    if remaining:
        sentences.append(remaining)

    return sentences if sentences else [text.strip()]


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[str]:
    """Split text into chunks respecting sentence boundaries and max size.

    Rules:
    - Never split mid-sentence.
    - A single sentence exceeding max_chars becomes its own chunk.
    - Chunks are as large as possible without exceeding max_chars.
    - Deterministic: same input always produces same output.
    """
    if not text.strip():
        return []

    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = (
            (current + " " + sentence).strip() if current else sentence
        )

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks if chunks else [text.strip()]
