"""Tests for EPUB parser."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.epub_builder import build_epub

from audiobook_engine.domain.exceptions import EbookError
from audiobook_engine.infrastructure.ebook.epub_parser import EPUBParser


@pytest.fixture()
def sample_epub(tmp_path: Path) -> Path:
    """Create a minimal two-chapter EPUB for testing."""
    path = tmp_path / "sample.epub"
    build_epub(
        path,
        title="Test Book",
        author="Test Author",
        language="en",
        chapters=[
            ("Chapter One", "First paragraph of chapter one.\nSecond paragraph here."),
            ("Chapter Two", "Chapter two opens.\nMore text in chapter two."),
        ],
    )
    return path


@pytest.fixture()
def parser() -> EPUBParser:
    return EPUBParser()


class TestValidation:
    def test_nonexistent_file(self, parser: EPUBParser) -> None:
        with pytest.raises(EbookError, match="not found"):
            parser.inspect(Path("/nonexistent/book.epub"))

    def test_not_epub_extension(
        self, parser: EPUBParser, tmp_path: Path
    ) -> None:
        txt = tmp_path / "book.txt"
        txt.write_text("hello")
        with pytest.raises(EbookError, match="Not an EPUB"):
            parser.inspect(txt)

    def test_corrupt_zip(
        self, parser: EPUBParser, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.epub"
        bad.write_bytes(b"not a zip file at all")
        with pytest.raises(EbookError, match="Corrupt"):
            parser.inspect(bad)


class TestMetadata:
    def test_title(self, parser: EPUBParser, sample_epub: Path) -> None:
        book = parser.inspect(sample_epub)
        assert book.title == "Test Book"

    def test_author(self, parser: EPUBParser, sample_epub: Path) -> None:
        book = parser.inspect(sample_epub)
        assert book.author == "Test Author"

    def test_language(
        self, parser: EPUBParser, sample_epub: Path
    ) -> None:
        book = parser.inspect(sample_epub)
        assert book.language == "en"

    def test_source_path(
        self, parser: EPUBParser, sample_epub: Path
    ) -> None:
        book = parser.inspect(sample_epub)
        assert book.source_path == sample_epub


class TestChapters:
    def test_chapter_count(
        self, parser: EPUBParser, sample_epub: Path
    ) -> None:
        book = parser.inspect(sample_epub)
        assert len(book.chapters) == 2

    def test_chapter_titles(
        self, parser: EPUBParser, sample_epub: Path
    ) -> None:
        book = parser.inspect(sample_epub)
        titles = [ch.title for ch in book.chapters]
        assert titles == ["Chapter One", "Chapter Two"]

    def test_chapter_indices(
        self, parser: EPUBParser, sample_epub: Path
    ) -> None:
        book = parser.inspect(sample_epub)
        indices = [ch.index for ch in book.chapters]
        assert indices == [0, 1]

    def test_segments_exist(
        self, parser: EPUBParser, sample_epub: Path
    ) -> None:
        book = parser.inspect(sample_epub)
        for ch in book.chapters:
            assert len(ch.segments) > 0

    def test_segment_text_content(
        self, parser: EPUBParser, sample_epub: Path
    ) -> None:
        book = parser.inspect(sample_epub)
        ch1 = book.chapters[0]
        texts = [seg.text for seg in ch1.segments]
        assert any("First paragraph" in t for t in texts)

    def test_segment_indices_sequential(
        self, parser: EPUBParser, sample_epub: Path
    ) -> None:
        book = parser.inspect(sample_epub)
        all_indices = []
        for ch in book.chapters:
            for seg in ch.segments:
                all_indices.append(seg.index)
        assert all_indices == list(range(len(all_indices)))


class TestDeterminism:
    def test_same_epub_produces_same_result(
        self, parser: EPUBParser, sample_epub: Path
    ) -> None:
        book1 = parser.inspect(sample_epub)
        book2 = parser.inspect(sample_epub)
        assert book1.title == book2.title
        assert book1.author == book2.author
        assert len(book1.chapters) == len(book2.chapters)
        for c1, c2 in zip(book1.chapters, book2.chapters, strict=True):
            assert c1.title == c2.title
            assert len(c1.segments) == len(c2.segments)
            for s1, s2 in zip(c1.segments, c2.segments, strict=True):
                assert s1.text == s2.text
                assert s1.index == s2.index


class TestEdgeCases:
    def test_empty_chapter_skipped(
        self, parser: EPUBParser, tmp_path: Path
    ) -> None:
        path = tmp_path / "with_empty.epub"
        build_epub(
            path,
            title="Empty Chapter Book",
            author="Author",
            language="en",
            chapters=[
                ("Chapter One", "Some text."),
                ("Empty Chapter", ""),
                ("Chapter Two", "More text."),
            ],
        )
        book = parser.inspect(path)
        titles = [ch.title for ch in book.chapters]
        assert "Empty Chapter" not in titles

    def test_paragraphs_become_segments(
        self, parser: EPUBParser, tmp_path: Path
    ) -> None:
        path = tmp_path / "multi_para.epub"
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        build_epub(
            path,
            title="Multi Para",
            author="Author",
            language="en",
            chapters=[("Ch", text)],
        )
        book = parser.inspect(path)
        ch = book.chapters[0]
        assert len(ch.segments) == 3
        assert ch.segments[0].text == "First paragraph."
        assert ch.segments[1].text == "Second paragraph."
        assert ch.segments[2].text == "Third paragraph."
