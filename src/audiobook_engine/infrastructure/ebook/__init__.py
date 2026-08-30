"""Ebook format adapters and text processing pipeline."""

from audiobook_engine.infrastructure.ebook.chunker import (
    chunk_text,
    split_sentences,
)
from audiobook_engine.infrastructure.ebook.epub_parser import EPUBParser
from audiobook_engine.infrastructure.ebook.mobi_parser import MOBIParser
from audiobook_engine.infrastructure.ebook.normalizer import (
    normalize,
    normalize_whitespace,
    remove_noise,
)

__all__ = [
    "EPUBParser",
    "MOBIParser",
    "chunk_text",
    "normalize",
    "normalize_whitespace",
    "remove_noise",
    "split_sentences",
]
