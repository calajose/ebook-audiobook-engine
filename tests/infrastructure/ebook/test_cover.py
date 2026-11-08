"""Tests for the shared cover extraction logic (A-08)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebooklib import epub  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from pathlib import Path

from audiobook_engine.infrastructure.ebook.cover import (
    OPF_NAMESPACE,
    extract_cover,
)

IMAGE_BYTES = b"\xff\xd8\xff\xe0fakejpeg"


def _image_item(uid: str = "art", name: str = "art.jpg") -> epub.EpubItem:
    return epub.EpubItem(
        uid=uid,
        file_name=name,
        media_type="image/jpeg",
        content=IMAGE_BYTES,
    )


def _book() -> epub.EpubBook:
    book = epub.EpubBook()
    book.set_identifier("cover-test")
    book.set_title("Cover Test")
    book.set_language("en")
    return book


class TestExtractCover:
    def test_opf_meta_with_correct_namespace(self, tmp_path: Path) -> None:
        """EPUB2 <meta name="cover"> must be found under the real OPF namespace."""
        book = _book()
        book.add_item(_image_item(uid="art", name="art.jpg"))
        book.add_metadata(
            OPF_NAMESPACE,
            "meta",
            "",
            {"name": "cover", "content": "art"},
        )

        cover = extract_cover(book, tmp_path / "book.epub")

        assert cover is not None
        assert cover.name == "book_cover.jpg"
        assert cover.read_bytes() == IMAGE_BYTES

    def test_fallback_by_item_id(self, tmp_path: Path) -> None:
        book = _book()
        book.add_item(_image_item(uid="cover", name="art.jpg"))

        cover = extract_cover(book, tmp_path / "book.epub")

        assert cover is not None
        assert cover.read_bytes() == IMAGE_BYTES

    def test_fallback_by_filename(self, tmp_path: Path) -> None:
        book = _book()
        book.add_item(_image_item(uid="img1", name="the-cover.png"))

        cover = extract_cover(book, tmp_path / "book.epub")

        assert cover is not None
        assert cover.name == "book_cover.png"
        assert cover.read_bytes() == IMAGE_BYTES

    def test_xhtml_wrapper_is_unwrapped(self, tmp_path: Path) -> None:
        book = _book()
        wrapper = epub.EpubHtml(title="Cover", file_name="cover.xhtml", lang="en")
        wrapper.id = "cover"
        wrapper.set_content('<html><body><img src="art.jpg"/></body></html>')
        book.add_item(wrapper)
        book.add_item(_image_item(uid="img1", name="art.jpg"))

        cover = extract_cover(book, tmp_path / "book.epub")

        assert cover is not None
        assert cover.suffix.lower() in (".jpg", ".jpeg", ".png")
        assert cover.read_bytes() == IMAGE_BYTES

    def test_no_cover_returns_none(self, tmp_path: Path) -> None:
        book = _book()
        chapter = epub.EpubHtml(title="Ch", file_name="ch1.xhtml", lang="en")
        chapter.set_content("<html><body><p>Text</p></body></html>")
        book.add_item(chapter)

        assert extract_cover(book, tmp_path / "book.epub") is None

    def test_cover_written_next_to_source(self, tmp_path: Path) -> None:
        """Cover location stays next to the user's ebook (BD-2026-10-12-24)."""
        book = _book()
        book.add_item(_image_item(uid="cover", name="art.jpg"))
        source = tmp_path / "library" / "my-book.epub"
        source.parent.mkdir()

        cover = extract_cover(book, source)

        assert cover is not None
        assert cover.parent == source.parent
        assert cover.name == "my-book_cover.jpg"
