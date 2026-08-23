"""Programmatic EPUB builder for tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebooklib import epub

if TYPE_CHECKING:
    from pathlib import Path


def build_epub(
    path: Path,
    *,
    title: str = "Test Book",
    author: str = "Test Author",
    language: str = "en",
    chapters: list[tuple[str, str]] | None = None,
) -> None:
    """Create a minimal EPUB file with the given metadata and chapters.

    Each chapter is a (heading, body_text) tuple.
    """
    if chapters is None:
        chapters = [("Chapter 1", "Default content.")]

    book = epub.EpubBook()
    book.set_identifier("test-epub-001")
    book.set_title(title)
    book.set_language(language)
    book.add_author(author)

    spine_items: list[epub.EpubItem | str] = ["nav"]
    toc: list[epub.Link] = []

    for idx, (heading, body) in enumerate(chapters):
        item_id = f"chapter_{idx}"
        file_name = f"chapter_{idx}.xhtml"

        paragraphs = "".join(
            f"<p>{p}</p>" for p in body.split("\n") if p.strip()
        )
        content = (
            f"<html><body>"
            f"<h1>{heading}</h1>"
            f"{paragraphs}"
            f"</body></html>"
        )

        chapter = epub.EpubHtml(
            title=heading,
            file_name=file_name,
            lang=language,
        )
        chapter.id = item_id
        chapter.set_content(content)

        book.add_item(chapter)
        spine_items.append(chapter)
        toc.append(epub.Link(file_name, heading, item_id))

    book.toc = toc
    book.spine = spine_items

    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(path), book)
