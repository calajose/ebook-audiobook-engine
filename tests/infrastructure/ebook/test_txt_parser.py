"""Tests for TXT parser."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from audiobook_engine.domain.exceptions import EbookError
from audiobook_engine.domain.protocols import EbookParser
from audiobook_engine.infrastructure.ebook.txt_parser import TXTParser

if TYPE_CHECKING:
    from pathlib import Path


class TestTXTParserValidation:
    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        parser = TXTParser()
        with pytest.raises(EbookError, match="File not found"):
            parser.inspect(tmp_path / "nonexistent.txt")

    def test_not_txt_extension_raises(self, tmp_path: Path) -> None:
        epub = tmp_path / "test.epub"
        epub.write_text("content")
        parser = TXTParser()
        with pytest.raises(EbookError, match="Not a TXT file"):
            parser.inspect(epub)

    def test_txt_file_accepted(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("Hello world.")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert book.title == "test"


class TestTXTParserMetadata:
    def test_title_from_filename(self, tmp_path: Path) -> None:
        txt = tmp_path / "My Book.txt"
        txt.write_text("Content here.")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert book.title == "My Book"

    def test_author_empty(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("Content.")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert book.author == ""

    def test_language_default_en(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("Content.")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert book.language == "en"

    def test_source_path_set(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("Content.")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert book.source_path == txt


class TestTXTParserChapters:
    def test_single_paragraph(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("Hello world.")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert len(book.chapters) == 1
        assert book.chapters[0].title == "Chapter 1"
        assert len(book.chapters[0].segments) == 1
        assert book.chapters[0].segments[0].text == "Hello world."

    def test_multiple_paragraphs(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("First paragraph.\n\nSecond paragraph.\n\nThird.")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert len(book.chapters) == 1
        assert len(book.chapters[0].segments) == 3
        assert book.chapters[0].segments[0].text == "First paragraph."
        assert book.chapters[0].segments[1].text == "Second paragraph."
        assert book.chapters[0].segments[2].text == "Third."

    def test_segment_indices_sequential(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("A\n\nB\n\nC")
        parser = TXTParser()
        book = parser.inspect(txt)
        indices = [seg.index for seg in book.chapters[0].segments]
        assert indices == [0, 1, 2]

    def test_empty_file(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert len(book.chapters) == 0

    def test_whitespace_only(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("   \n\n   \n  ")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert len(book.chapters) == 0

    def test_blank_lines_between_paragraphs(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("Para 1\n\n\n\nPara 2")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert len(book.chapters[0].segments) == 2

    def test_single_line_text(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("Line 1\nLine 2\nLine 3")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert len(book.chapters) == 1
        # Without blank lines, all lines become one paragraph
        assert len(book.chapters[0].segments) == 1

    def test_multiline_paragraphs(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("Line 1\nLine 2\n\nLine 3\nLine 4")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert len(book.chapters[0].segments) == 2
        assert "Line 1" in book.chapters[0].segments[0].text
        assert "Line 3" in book.chapters[0].segments[1].text


class TestTXTParserSatisfiesProtocol:
    def test_satisfies_ebook_parser(self) -> None:
        parser = TXTParser()
        assert isinstance(parser, EbookParser)


class TestTXTParserEncoding:
    def test_latin1_fallback(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_bytes(b"Caf\xe9 in Paris.")
        parser = TXTParser()
        book = parser.inspect(txt)
        assert len(book.chapters) == 1
