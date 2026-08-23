"""TXT parser — extracts Book model from plain text files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from audiobook_engine.domain.exceptions import EbookError
from audiobook_engine.domain.models import Book, Chapter, TextSegment

if TYPE_CHECKING:
    from pathlib import Path


class TXTParser:
    """Parse plain text files into the domain Book model.

    Splits content by blank lines into paragraphs. The filename
    becomes the book title.
    """

    def inspect(self, path: Path) -> Book:
        """Parse a TXT file and return a Book with chapters and segments."""
        self._validate(path)

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="latin-1")
            except Exception as exc:
                raise EbookError(
                    f"Cannot read file encoding: {exc}"
                ) from exc

        title = path.stem
        chapters = self._extract_chapters(text)

        return Book(
            title=title,
            author="",
            language="en",
            chapters=chapters,
            source_path=path,
        )

    @staticmethod
    def _validate(path: Path) -> None:
        if not path.exists():
            raise EbookError(f"File not found: {path}")
        if not path.suffix.lower() == ".txt":
            raise EbookError(f"Not a TXT file: {path}")

    @staticmethod
    def _extract_chapters(text: str) -> tuple[Chapter, ...]:
        """Split text into chapters by blank lines (paragraphs).

        Each group of non-empty lines becomes one paragraph/segment.
        All paragraphs go into a single chapter.
        """
        paragraphs = [
            p.strip() for p in text.split("\n\n") if p.strip()
        ]

        if not paragraphs:
            # Handle single-block text without blank lines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                paragraphs = [" ".join(lines)]

        if not paragraphs:
            return ()

        segments = tuple(
            TextSegment(text=para, index=i)
            for i, para in enumerate(paragraphs)
        )

        return (
            Chapter(
                title="Chapter 1",
                index=0,
                segments=segments,
            ),
        )
