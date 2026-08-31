"""Tests for EPUB parser."""

from __future__ import annotations

from pathlib import Path

import pytest
from ebooklib import epub  # type: ignore[import-untyped]
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

    def test_h4_heading_detected(
        self, parser: EPUBParser, tmp_path: Path
    ) -> None:
        """EPUBs generated by Calibre may use <h4> for chapter titles."""
        path = tmp_path / "h4_headings.epub"
        build_epub(
            path,
            title="H4 Book",
            author="Author",
            language="es",
            chapters=[
                ("Chapter One", "Some text."),
                ("Chapter Two", "More text."),
            ],
        )

        # Rewrite chapter XHTML to use <h4> instead of <h1>
        import zipfile

        with zipfile.ZipFile(path, "r") as zin:
            contents = {n: zin.read(n) for n in zin.namelist()}

        for name, data in contents.items():
            if name.startswith("chapter_") and name.endswith(".xhtml"):
                decoded = data.decode("utf-8")
                decoded = decoded.replace("<h1>", "<h4>").replace(
                    "</h1>", "</h4>"
                )
                contents[name] = decoded.encode("utf-8")

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
            for name, data in contents.items():
                zout.writestr(name, data)

        parsed = parser.inspect(path)
        assert len(parsed.chapters) == 2
        assert parsed.chapters[0].title == "Chapter One"
        assert parsed.chapters[1].title == "Chapter Two"


class TestCoverFallback:
    def test_cover_detected_by_id_substring(
        self, parser: EPUBParser, tmp_path: Path
    ) -> None:
        """Cover is detected when ID contains 'cover' even without standard metadata."""
        path = tmp_path / "cover_fallback.epub"
        book = epub.EpubBook()
        book.set_identifier("test-123")
        book.set_title("Cover Fallback")
        book.set_language("en")

        ch = epub.EpubHtml(title="Ch 1", file_name="ch1.xhtml", lang="en")
        ch.id = "ch1"
        ch.set_content("<html><body><p>Hello</p></body></html>")
        book.add_item(ch)
        book.spine = ["nav", ch]

        img = epub.EpubItem(
            uid="cover-image",
            file_name="cover.jpg",
            media_type="image/jpeg",
            content=b"\xff\xd8\xff\xe0fakejpeg",
        )
        book.add_item(img)

        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        epub.write_epub(str(path), book)

        parsed = parser.inspect(path)
        assert parsed.cover_path is not None
        assert parsed.cover_path.exists()
        assert "cover" in parsed.cover_path.name

    def test_cover_extracted_from_xhtml_wrapper(
        self, parser: EPUBParser, tmp_path: Path
    ) -> None:
        """When cover points to an XHTML page, underlying image is extracted."""
        path = tmp_path / "xhtml_cover.epub"
        book = epub.EpubBook()
        book.set_identifier("test-456")
        book.set_title("XHTML Cover")
        book.set_language("en")

        ch = epub.EpubHtml(title="Ch 1", file_name="ch1.xhtml", lang="en")
        ch.id = "ch1"
        ch.set_content("<html><body><p>Hello</p></body></html>")
        book.add_item(ch)
        book.spine = ["nav", ch]

        cover_xhtml = epub.EpubHtml(title="Cover", file_name="cover.xhtml", lang="en")
        cover_xhtml.id = "cover"
        cover_xhtml.set_content('<html><body><img src="cover.jpg"/></body></html>')
        book.add_item(cover_xhtml)

        img = epub.EpubItem(
            uid="img1",
            file_name="cover.jpg",
            media_type="image/jpeg",
            content=b"\xff\xd8\xff\xe0fakejpeg",
        )
        book.add_item(img)

        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        epub.write_epub(str(path), book)

        parsed = parser.inspect(path)
        assert parsed.cover_path is not None
        assert parsed.cover_path.exists()
        assert parsed.cover_path.suffix.lower() in (".jpg", ".jpeg", ".png")
        assert parsed.cover_path.read_bytes() == b"\xff\xd8\xff\xe0fakejpeg"


