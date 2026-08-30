"""EPUB parser — extracts Book model from .epub files."""

from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

import ebooklib  # type: ignore[import-untyped]
from ebooklib import epub

from audiobook_engine.domain.exceptions import EbookError
from audiobook_engine.domain.models import Book, Chapter, TextSegment


class EPUBParser:
    """Parse EPUB files into the domain Book model."""

    def inspect(self, path: Path) -> Book:
        """Parse an EPUB and return a Book with chapters and segments."""
        self._validate(path)

        try:
            book = epub.read_epub(str(path))
        except Exception as exc:
            raise EbookError(f"Failed to parse EPUB: {exc}") from exc

        title = self._extract_title(book)
        author = self._extract_author(book)
        language = self._extract_language(book)
        cover_path = self._extract_cover(book, path)
        chapters = self._extract_chapters(book)

        return Book(
            title=title,
            author=author,
            language=language,
            chapters=chapters,
            cover_path=cover_path,
            source_path=path,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(path: Path) -> None:
        if not path.exists():
            raise EbookError(f"File not found: {path}")
        if not path.suffix.lower() == ".epub":
            raise EbookError(f"Not an EPUB file: {path}")
        try:
            with zipfile.ZipFile(path):
                pass
        except zipfile.BadZipFile as exc:
            raise EbookError(f"Corrupt or invalid ZIP: {exc}") from exc

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_title(book: epub.EpubBook) -> str:
        title = book.get_metadata("DC", "title")
        if title:
            return str(title[0][0]).strip()
        return ""

    @staticmethod
    def _extract_author(book: epub.EpubBook) -> str:
        creator = book.get_metadata("DC", "creator")
        if creator:
            return str(creator[0][0]).strip()
        return ""

    @staticmethod
    def _extract_language(book: epub.EpubBook) -> str:
        lang = book.get_metadata("DC", "language")
        if lang:
            return str(lang[0][0]).strip()
        return "en"

    # ------------------------------------------------------------------
    # Cover extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cover(
        book: epub.EpubBook, source_path: Path
    ) -> Path | None:
        # EPUB3: look for EpubCover items (properties="cover-image")
        for item in book.get_items():
            if isinstance(item, epub.EpubCover):
                content = item.get_content()
                if content:
                    suffix = Path(item.get_name()).suffix or ".jpg"
                    out = (
                        source_path.parent
                        / f"{source_path.stem}_cover{suffix}"
                    )
                    out.write_bytes(content)
                    return out

        # EPUB2 fallback: look in OPF meta metadata
        opf_ns = "OPF"
        meta_entries = book.metadata.get(opf_ns, {}).get("meta", [])
        for _val, others in meta_entries:
            if others and others.get("name") == "cover":
                cover_id = others.get("content", "")
                if cover_id:
                    item = book.get_item_with_id(cover_id)
                    if item is not None:
                        content = item.get_content()
                        if content:
                            suffix = (
                                Path(item.get_name()).suffix or ".jpg"
                            )
                            out = (
                                source_path.parent
                                / f"{source_path.stem}_cover{suffix}"
                            )
                            out.write_bytes(content)
                            return out

        # Direct fallback: try item with id="cover" regardless of metadata
        item = book.get_item_with_id("cover")
        if item is not None:
            content = item.get_content()
            if content:
                suffix = Path(item.get_name()).suffix or ".jpg"
                out = (
                    source_path.parent
                    / f"{source_path.stem}_cover{suffix}"
                )
                out.write_bytes(content)
                return out

        return None

    # ------------------------------------------------------------------
    # Spine-ordered chapter extraction
    # ------------------------------------------------------------------

    def _extract_chapters(
        self, book: epub.EpubBook
    ) -> tuple[Chapter, ...]:
        # Filter out navigation items (nav, ncx) from spine
        skip_ids = {"nav", "ncx"}
        spine_ids = [
            item_id
            for item_id, _ in book.spine
            if item_id not in skip_ids
        ]
        spine_items = {
            item.get_id(): item
            for item in book.get_items()
            if item.get_type() == ebooklib.ITEM_DOCUMENT
        }

        chapters: list[Chapter] = []
        seg_index = 0

        for idx, item_id in enumerate(spine_ids):
            item = spine_items.get(item_id)
            if item is None:
                continue

            content = item.get_content().decode(
                "utf-8", errors="replace"
            )
            heading = self._extract_heading(content)
            segments, seg_index = self._extract_segments(
                content, seg_index
            )

            if segments:
                chapters.append(
                    Chapter(
                        title=heading or f"Section {idx + 1}",
                        index=idx,
                        segments=segments,
                        source_file=item.get_name(),
                    )
                )

        return tuple(chapters)

    # ------------------------------------------------------------------
    # XHTML content processing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_heading(xhtml: str) -> str:
        """Extract the first h1/h2/h3 heading from XHTML content."""
        for tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            match = re.search(
                rf"<{tag}[^>]*>(.*?)</{tag}>",
                xhtml,
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                text = re.sub(r"<[^>]+>", "", match.group(1))
                text = html.unescape(text).strip()
                if text:
                    return text
        return ""

    @staticmethod
    def _text_from_xhtml(xhtml: str) -> str:
        """Extract body text, stripping headings and all tags."""
        # Remove heading elements (title is extracted separately)
        text = re.sub(
            r"<h[1-6][^>]*>.*?</h[1-6]>",
            "",
            xhtml,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)
        return text.strip()

    @staticmethod
    def _normalize_text(raw: str) -> str:
        """Collapse excessive whitespace while preserving paragraph breaks."""
        lines = raw.splitlines()
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned.append(stripped)
        return "\n".join(cleaned)

    @classmethod
    def _extract_segments(
        cls, xhtml: str, start_index: int
    ) -> tuple[tuple[TextSegment, ...], int]:
        """Split XHTML content into TextSegment objects.

        Segmentation is paragraph-based: each non-empty paragraph
        becomes one TextSegment. Sentence-level chunking is done
        later in the normalization/chunking phase.
        """
        plain = cls._text_from_xhtml(xhtml)
        normalized = cls._normalize_text(plain)

        if not normalized:
            return (), start_index

        paragraphs = [
            p for p in normalized.split("\n") if p.strip()
        ]

        segments = tuple(
            TextSegment(text=para, index=start_index + i)
            for i, para in enumerate(paragraphs)
        )

        return segments, start_index + len(paragraphs)
