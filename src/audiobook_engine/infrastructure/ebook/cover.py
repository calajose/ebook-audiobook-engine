"""Shared cover extraction for EPUB and MOBI/AZW3 parsers.

Both parsers delegate here so the fallback chain and the XHTML wrapper
handling are identical (A-08 / BD-2026-10-12-02). The extracted cover is
written next to the user's source ebook as ``<stem>_cover.<ext>`` — a
location protected by BD-2026-10-12-24 (deliberate added value, not
contamination).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import ebooklib  # type: ignore[import-untyped]
from ebooklib import epub

if TYPE_CHECKING:
    from ebooklib import epub as epub_types

IMAGE_SUFFIXES: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
)

# Correct OPF namespace. The previous EPUB implementation looked up the
# bogus key "OPF", which is why standard EPUB2 covers were lost.
OPF_NAMESPACE = "http://www.idpf.org/2007/opf"

_IMG_SRC_RE = re.compile(
    r'<img[^>]+src=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def extract_cover(book: epub_types.EpubBook, source_path: Path) -> Path | None:
    """Extract the cover image of an EPUB-derived book, or None.

    Single fallback chain shared by every parser:

    1. EPUB3 ``EpubCover`` items.
    2. OPF2 ``<meta name="cover">`` metadata (correct OPF namespace).
    3. Item whose id is exactly ``cover``.
    4. Item whose id or filename contains ``cover``.
    5. First image item in the book.
    """
    # 1. EPUB3 cover item
    for item in book.get_items():
        if isinstance(item, epub.EpubCover):
            saved = _save_item_as_cover(item, book, source_path)
            if saved is not None:
                return saved

    # 2. EPUB2 OPF metadata (correct namespace)
    meta_entries = book.metadata.get(OPF_NAMESPACE, {}).get("meta", [])
    for _value, others in meta_entries:
        if others and others.get("name") == "cover":
            cover_id = others.get("content", "")
            if cover_id:
                saved = _save_item_as_cover(
                    book.get_item_with_id(cover_id), book, source_path
                )
                if saved is not None:
                    return saved

    # 3. Item with id="cover"
    saved = _save_item_as_cover(book.get_item_with_id("cover"), book, source_path)
    if saved is not None:
        return saved

    # 4. Item id or filename contains "cover" (case-insensitive)
    for item in book.get_items():
        item_id = item.get_id() or ""
        item_name = item.get_name() or ""
        if "cover" in item_id.lower() or "cover" in Path(item_name).name.lower():
            saved = _save_item_as_cover(item, book, source_path)
            if saved is not None:
                return saved

    # 5. First image item in the book
    for item in book.get_items():
        if _is_image_item(item):
            content = item.get_content()
            if content:
                suffix = _image_suffix(item) or ".jpg"
                return _write_cover(source_path, content, suffix)

    return None


def _is_image_item(item: ebooklib.epub.EpubItem) -> bool:
    """Return True when an item holds an image payload."""
    media_type = getattr(item, "media_type", "") or ""
    suffix = Path(item.get_name() or "").suffix.lower()
    return (
        item.get_type() == ebooklib.ITEM_IMAGE
        or media_type.startswith("image/")
        or suffix in IMAGE_SUFFIXES
    )


def _image_suffix(item: ebooklib.epub.EpubItem) -> str:
    """Return the image extension of an item (empty when unknown)."""
    return Path(item.get_name() or "").suffix.lower()


def _write_cover(source_path: Path, content: bytes, suffix: str) -> Path:
    """Write cover bytes next to the source ebook."""
    out = source_path.parent / f"{source_path.stem}_cover{suffix}"
    out.write_bytes(content)
    return out


def _save_item_as_cover(
    item: epub_types.EpubItem | None,
    book: epub_types.EpubBook,
    source_path: Path,
) -> Path | None:
    """Save an item as the cover, unwrapping XHTML wrapper pages.

    A wrapper page (for example ``cover.xhtml`` with an ``<img>`` tag)
    is resolved to the underlying image item, so parsers never write
    HTML bytes into a file with an image extension.
    """
    if item is None:
        return None
    content = item.get_content()
    if not content:
        return None

    if _is_image_item(item):
        return _write_cover(source_path, content, _image_suffix(item) or ".jpg")

    item_name = item.get_name() or ""
    media_type = getattr(item, "media_type", "") or ""
    suffix = Path(item_name).suffix.lower()
    is_html = (
        suffix in (".xhtml", ".html", ".htm")
        or "html" in media_type.lower()
        or b"<html" in content[:100].lower()
    )
    if not is_html:
        return None

    html_text = content.decode("utf-8", errors="replace")
    img_match = _IMG_SRC_RE.search(html_text)
    if img_match is None:
        return None

    img_filename = Path(img_match.group(1)).name.lower()
    for sub_item in book.get_items():
        if not _is_image_item(sub_item):
            continue
        sub_name = (sub_item.get_name() or "").lower()
        if sub_name.endswith(img_filename) or img_filename in sub_name:
            sub_content = sub_item.get_content()
            if sub_content:
                return _write_cover(
                    source_path,
                    sub_content,
                    _image_suffix(sub_item) or ".jpg",
                )
    return None
