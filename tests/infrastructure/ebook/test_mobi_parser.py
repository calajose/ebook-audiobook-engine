"""Tests for MOBI/AZW/AZW3 parser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from tests.fixtures.epub_builder import build_epub

from audiobook_engine.domain.exceptions import EbookError
from audiobook_engine.domain.protocols import EbookParser
from audiobook_engine.infrastructure.ebook.mobi_parser import MOBIParser


@pytest.fixture()
def parser() -> MOBIParser:
    return MOBIParser()


@pytest.fixture()
def sample_mobi(tmp_path: Path) -> Path:
    """Create a minimal MOBI file for testing (just needs correct extension)."""
    path = tmp_path / "sample.azw3"
    path.write_bytes(b"\x00" * 8)
    return path


@pytest.fixture()
def sample_html_mobi(tmp_path: Path) -> Path:
    """Create a minimal MOBI file for HTML extraction testing."""
    path = tmp_path / "sample.mobi"
    path.write_bytes(b"\x00" * 8)
    return path


def _create_opf(
    directory: Path,
    *,
    title: str = "",
    author: str = "",
    language: str = "",
) -> None:
    """Helper: write a content.opf into *directory*."""
    parts: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<package version="2.0" xmlns="http://www.idpf.org/2007/opf">',
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">',
    ]
    if title:
        parts.append(f"<dc:title>{title}</dc:title>")
    if language:
        parts.append(f"<dc:language>{language}</dc:language>")
    if author:
        parts.append(f"<dc:creator>{author}</dc:creator>")
    parts.append("</metadata></package>")
    (directory / "content.opf").write_text("\n".join(parts), encoding="utf-8")


def _create_ncx(
    directory: Path,
    *,
    title: str = "",
    entries: list[tuple[str, int]] | None = None,
) -> None:
    """Helper: write a toc.ncx into *directory*."""
    parts: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">',
    ]
    if title:
        parts.append("<docTitle>")
        parts.append(f"<text>{title}</text>")
        parts.append("</docTitle>")
    parts.append("<navMap>")
    for idx, (label, filepos) in enumerate(entries or [], start=1):
        parts.append(f'<navPoint id="np_{idx}" playOrder="{idx}">')
        parts.append("<navLabel>")
        parts.append(f"<text>{label}</text>")
        parts.append("</navLabel>")
        parts.append(
            f'<content src="book.html#filepos{filepos}"/>'
        )
        parts.append("</navPoint>")
    parts.append("</navMap></ncx>")
    (directory / "toc.ncx").write_text("\n".join(parts), encoding="utf-8")


class TestValidation:
    def test_nonexistent_file(self, parser: MOBIParser) -> None:
        with pytest.raises(EbookError, match="not found"):
            parser.inspect(Path("/nonexistent/book.azw3"))

    def test_not_mobi_extension(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        epub = tmp_path / "book.epub"
        epub.write_bytes(b"not a real file")
        with pytest.raises(EbookError, match="Not a MOBI"):
            parser.inspect(epub)

    def test_azw3_extension_accepted(
        self, parser: MOBIParser, sample_mobi: Path
    ) -> None:
        """AZW3 extension should be accepted."""
        # Will fail at extraction but should pass validation
        with pytest.raises(EbookError, match="Failed to"):
            parser.inspect(sample_mobi)

    def test_azw_extension_accepted(
        self, parser: MOBIParser, sample_html_mobi: Path
    ) -> None:
        """AZW extension should be accepted."""
        with pytest.raises(EbookError, match="Failed to"):
            parser.inspect(sample_html_mobi)

    def test_mobi_extension_accepted(
        self, parser: MOBIParser, sample_html_mobi: Path
    ) -> None:
        """MOBI extension should be accepted."""
        with pytest.raises(EbookError, match="Failed to"):
            parser.inspect(sample_html_mobi)


class TestProtocol:
    def test_satisfies_ebook_parser(self) -> None:
        parser = MOBIParser()
        assert isinstance(parser, EbookParser)


class TestExtractFromEPUB:
    """Test parsing when mobi.extract() returns an EPUB (AZW3/KF8)."""

    def test_epub_extraction(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Test parsing with EPUB extraction (AZW3 flow)."""
        # Create a real EPUB to use as the extracted output
        epub_path = tmp_path / "extracted.epub"
        build_epub(
            epub_path,
            title="AZW3 Book",
            author="AZW3 Author",
            language="es",
            chapters=[
                ("Chapter 1", "First paragraph.\nSecond paragraph."),
                ("Chapter 2", "Chapter two content."),
            ],
        )

        # Create a fake mobi file (just needs correct extension)
        mobi_path = tmp_path / "book.azw3"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(epub_path))
            book = parser.inspect(mobi_path)

        assert book.title == "AZW3 Book"
        assert book.author == "AZW3 Author"
        assert book.language == "es"
        assert book.source_path == mobi_path
        assert len(book.chapters) == 2
        assert book.chapters[0].title == "Chapter 1"
        assert book.chapters[1].title == "Chapter 2"

    def test_epub_segment_content(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Test that paragraph content is correctly extracted."""
        epub_path = tmp_path / "extracted.epub"
        build_epub(
            epub_path,
            title="Content Test",
            chapters=[
                ("Ch", "Hello world.\nGoodbye world."),
            ],
        )

        mobi_path = tmp_path / "book.azw3"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(epub_path))
            book = parser.inspect(mobi_path)

        ch = book.chapters[0]
        texts = [seg.text for seg in ch.segments]
        assert any("Hello world" in t for t in texts)
        assert any("Goodbye world" in t for t in texts)


class TestExtractFromHTML:
    """Test parsing when mobi.extract() returns HTML (MOBI v7/AZW)."""

    def test_html_extraction_single_chapter(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Test parsing with HTML extraction (no headings, no OPF)."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            "<html><body>"
            "<p>First paragraph.</p>"
            "<p>Second paragraph.</p>"
            "</body></html>"
        )

        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        assert book.title == "book"
        assert book.author == ""
        assert book.language == "en"
        assert book.source_path == mobi_path
        assert len(book.chapters) == 1
        assert book.chapters[0].title == "Chapter 1"

    def test_html_extraction_with_headings(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Test parsing with HTML extraction (with chapter headings)."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            "<html><body>"
            "<h1>Chapter One</h1>"
            "<p>First chapter content.</p>"
            "<h1>Chapter Two</h1>"
            "<p>Second chapter content.</p>"
            "</body></html>"
        )

        mobi_path = tmp_path / "book.azw"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        assert book.title == "book"
        assert len(book.chapters) == 2
        assert book.chapters[0].title == "Chapter One"
        assert book.chapters[1].title == "Chapter Two"

    def test_html_title_from_title_tag(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Test that <title> tag is used for book title."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            "<html><head><title>My Great Book</title></head>"
            "<body><p>Content.</p></body></html>"
        )

        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        assert book.title == "My Great Book"

    def test_html_empty_chapter_skipped(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Test that empty chapters are skipped."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            "<html><body>"
            "<h1>Chapter One</h1>"
            "<p>Content here.</p>"
            "<h1>Empty Chapter</h1>"
            "<h1>Chapter Three</h1>"
            "<p>More content.</p>"
            "</body></html>"
        )

        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        titles = [ch.title for ch in book.chapters]
        assert "Empty Chapter" not in titles
        assert len(book.chapters) == 2


class TestOPFMetadata:
    """Test metadata extraction from content.opf in HTML path."""

    def test_opf_language_and_author(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """OPF language and author override HTML defaults."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            "<html><body><p>Content.</p></body></html>"
        )
        _create_opf(
            tmp_path,
            title="Libro en Español",
            author="Autor Famoso",
            language="es",
        )

        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        assert book.language == "es"
        assert book.author == "Autor Famoso"
        assert book.title == "Libro en Español"

    def test_opf_takes_priority_over_html_title(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """OPF title takes priority over <title> tag in HTML."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            "<html><head><title>HTML Title</title></head>"
            "<body><p>Content.</p></body></html>"
        )
        _create_opf(tmp_path, title="OPF Title", language="fr")

        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        assert book.title == "OPF Title"

    def test_missing_opf_falls_back_to_html(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Without OPF, language defaults to 'en' and title from HTML."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            "<html><head><title>My Book</title></head>"
            "<body><p>Content.</p></body></html>"
        )

        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        assert book.language == "en"
        assert book.title == "My Book"
        assert book.author == ""


class TestNCXChapters:
    """Test chapter splitting using NCX toc.ncx entries."""

    def test_ncx_based_chapter_splitting(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """NCX entries with filepos anchors split HTML into chapters."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            '<html><body>'
            '<a id="filepos100" />'
            "<p>Intro text.</p>"
            '<a id="filepos500" />'
            "<p>Chapter one text.</p>"
            '<a id="filepos1000" />'
            "<p>Chapter two text.</p>"
            "</body></html>"
        )
        _create_ncx(
            tmp_path,
            entries=[
                ("Intro", 100),
                ("Chapter 1", 500),
                ("Chapter 2", 1000),
            ],
        )

        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        assert len(book.chapters) == 3
        assert book.chapters[0].title == "Intro"
        assert book.chapters[1].title == "Chapter 1"
        assert book.chapters[2].title == "Chapter 2"

    def test_ncx_chapter_content(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Each chapter contains only text between its anchors."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            '<html><body>'
            '<a id="filepos10" />'
            "<p>First content.</p>"
            '<a id="filepos200" />'
            "<p>Second content.</p>"
            "</body></html>"
        )
        _create_ncx(
            tmp_path,
            entries=[("Part A", 10), ("Part B", 200)],
        )

        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        ch1_texts = [s.text for s in book.chapters[0].segments]
        ch2_texts = [s.text for s in book.chapters[1].segments]
        assert any("First content" in t for t in ch1_texts)
        assert any("Second content" in t for t in ch2_texts)

    def test_ncx_skips_missing_anchors(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """NCX entries whose filepos anchor is missing in HTML are skipped."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            '<html><body>'
            '<a id="filepos10" />'
            "<p>Only content.</p>"
            "</body></html>"
        )
        _create_ncx(
            tmp_path,
            entries=[("Existing", 10), ("Missing", 9999)],
        )

        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        assert len(book.chapters) == 1
        assert book.chapters[0].title == "Existing"

    def test_ncx_fallback_to_headings(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Without NCX, falls back to h1-h3 heading splitting."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            "<html><body>"
            "<h1>Chapter A</h1>"
            "<p>Content A.</p>"
            "<h1>Chapter B</h1>"
            "<p>Content B.</p>"
            "</body></html>"
        )
        # No NCX or OPF in tmp_path

        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        assert len(book.chapters) == 2
        assert book.chapters[0].title == "Chapter A"
        assert book.chapters[1].title == "Chapter B"

    def test_ncx_empty_ncx_falls_back(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Empty/unparseable NCX falls back to heading detection."""
        html_path = tmp_path / "book.html"
        html_path.write_text(
            "<html><body>"
            "<h1>Heading Chapter</h1>"
            "<p>Some content.</p>"
            "</body></html>"
        )
        # Write an invalid NCX
        (tmp_path / "toc.ncx").write_text("not xml", encoding="utf-8")

        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.return_value = (str(tmp_path), str(html_path))
            book = parser.inspect(mobi_path)

        assert len(book.chapters) == 1
        assert book.chapters[0].title == "Heading Chapter"


class TestDRMDetection:
    def test_drm_error_message(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Test that DRM-protected files raise clear error."""
        mobi_path = tmp_path / "drm_book.azw3"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.side_effect = ValueError(
                "DRM protected content"
            )
            with pytest.raises(EbookError, match="DRM"):
                parser.inspect(mobi_path)

    def test_encrypted_error_message(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Test that encrypted files raise clear error."""
        mobi_path = tmp_path / "encrypted.azw"
        mobi_path.write_bytes(b"\x00" * 8)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract"
        ) as mock_extract:
            mock_extract.side_effect = ValueError(
                "Cannot process encrypted file"
            )
            with pytest.raises(EbookError, match="DRM"):
                parser.inspect(mobi_path)


class TestTempDirCleanup:
    def test_temp_dir_cleaned_on_success(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Test that temp directory is cleaned up after successful parse."""
        epub_path = tmp_path / "extracted.epub"
        build_epub(epub_path, title="Cleanup Test")

        mobi_path = tmp_path / "book.azw3"
        mobi_path.write_bytes(b"\x00" * 8)

        captured_tempdir: str | None = None

        def mock_extract(path: str) -> tuple[str, str]:
            nonlocal captured_tempdir
            import tempfile

            tempdir = tempfile.mkdtemp(prefix="test_mobi_")
            captured_tempdir = tempdir
            return tempdir, str(epub_path)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract",
            side_effect=mock_extract,
        ):
            parser.inspect(mobi_path)

        assert captured_tempdir is not None
        assert not Path(captured_tempdir).exists()

    def test_temp_dir_cleaned_on_error(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Test that temp directory is cleaned up after error."""
        mobi_path = tmp_path / "book.azw3"
        mobi_path.write_bytes(b"\x00" * 8)

        captured_tempdir: str | None = None

        def mock_extract(path: str) -> tuple[str, str]:
            nonlocal captured_tempdir
            import tempfile

            tempdir = tempfile.mkdtemp(prefix="test_mobi_err_")
            captured_tempdir = tempdir
            return tempdir, "/nonexistent/file.html"

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract",
            side_effect=mock_extract,
        ), pytest.raises(EbookError):
            parser.inspect(mobi_path)

        assert captured_tempdir is not None
        assert not Path(captured_tempdir).exists()


class TestDeterminism:
    def test_same_input_produces_same_result(
        self, parser: MOBIParser, tmp_path: Path
    ) -> None:
        """Test that parsing the same file twice produces identical results."""
        epub_path = tmp_path / "extracted.epub"
        build_epub(
            epub_path,
            title="Determinism Test",
            author="Test Author",
            chapters=[("Ch", "Some text here.")],
        )

        mobi_path = tmp_path / "book.azw3"
        mobi_path.write_bytes(b"\x00" * 8)

        def mock_extract(path: str) -> tuple[str, str]:
            import shutil

            # Create a fresh copy each time
            import tempfile

            tempdir = tempfile.mkdtemp(prefix="test_det_")
            dest = Path(tempdir) / "extracted.epub"
            shutil.copy2(epub_path, dest)
            return tempdir, str(dest)

        with patch(
            "audiobook_engine.infrastructure.ebook.mobi_parser.mobi.extract",
            side_effect=mock_extract,
        ):
            book1 = parser.inspect(mobi_path)
            book2 = parser.inspect(mobi_path)

        assert book1.title == book2.title
        assert book1.author == book2.author
        assert len(book1.chapters) == len(book2.chapters)
        for c1, c2 in zip(book1.chapters, book2.chapters, strict=True):
            assert c1.title == c2.title
            assert len(c1.segments) == len(c2.segments)
