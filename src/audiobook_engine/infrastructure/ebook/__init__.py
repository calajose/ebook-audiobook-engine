"""EPUB format adapter and text processing pipeline."""

from audiobook_engine.infrastructure.ebook.chunker import (
    chunk_text,
    split_sentences,
)
from audiobook_engine.infrastructure.ebook.epub_parser import EPUBParser
from audiobook_engine.infrastructure.ebook.normalizer import (
    normalize,
    normalize_whitespace,
    remove_noise,
)

__all__ = [
    "EPUBParser",
    "chunk_text",
    "normalize",
    "normalize_whitespace",
    "remove_noise",
    "split_sentences",
]
