"""MOBI/AZW/AZW3 parser — extracts Book model from Kindle ebook files."""

from __future__ import annotations

import html
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import ebooklib  # type: ignore[import-untyped]
import mobi  # type: ignore[import-untyped]
from ebooklib import epub

from audiobook_engine.domain.exceptions import EbookError
from audiobook_engine.domain.models import Book, Chapter, TextSegment

_SUPPORTED_EXTENSIONS = frozenset({".mobi", ".azw", ".azw3"})


class MOBIParser:
    """Parse MOBI/AZW/AZW3 files into the domain Book model.

    Uses the ``mobi`` library (KindleUnpack) to unpack the file, then
    parses the resulting EPUB (KF8/AZW3) or HTML (MOBI v7/AZW) output.
    """

    def inspect(self, path: Path) -> Book:
        """Parse a MOBI/AZW/AZW3 file and return a Book."""
        self._validate(path)

        tempdir: Path | None = None
        try:
            tempdir, extracted_path = self._extract(path)
            extracted = Path(extracted_path)

            if extracted.suffix.lower() == ".epub":
                return self._parse_epub(extracted, path)
            elif extracted.suffix.lower() == ".html":
                return self._parse_html(extracted, path)
            else:
                raise EbookError(
                    f"Unsupported extracted format: {extracted.suffix}. "
                    f"Expected .epub or .html"
                )
        except EbookError:
            raise
        except ValueError as exc:
            # mobi.extract raises ValueError for DRM-protected files
            msg = str(exc).lower()
            if "drm" in msg or "encrypt" in msg or "crypto" in msg:
                raise EbookError(
                    f"DRM-protected file not supported: {path}"
                ) from exc
            raise EbookError(
                f"Failed to extract MOBI file: {exc}"
            ) from exc
        except Exception as exc:
            msg = str(exc).lower()
            if "drm" in msg or "encrypt" in msg or "crypto" in msg:
                raise EbookError(
                    f"DRM-protected file not supported: {path}"
                ) from exc
            raise EbookError(
                f"Failed to parse MOBI file: {exc}"
            ) from exc
        finally:
            if tempdir is not None:
                shutil.rmtree(tempdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(path: Path) -> None:
        if not path.exists():
            raise EbookError(f"File not found: {path}")
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise EbookError(
                f"Not a MOBI/AZW/AZW3 file: {path}"
            )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(path: Path) -> tuple[Path, str]:
        """Unpack the MOBI file and return (tempdir, extracted_path)."""
        tempdir_str, filepath_str = mobi.extract(str(path))
        return Path(tempdir_str), filepath_str

    # ------------------------------------------------------------------
    # EPUB parsing (for AZW3/KF8 files)
    # ------------------------------------------------------------------

    def _parse_epub(
        self, epub_path: Path, source_path: Path
    ) -> Book:
        """Parse an extracted EPUB file (from AZW3/KF8)."""
        try:
            book = epub.read_epub(str(epub_path))
        except Exception as exc:
            raise EbookError(
                f"Failed to parse extracted EPUB: {exc}"
            ) from exc

        title = self._extract_epub_title(book) or source_path.stem
        author = self._extract_epub_author(book)
        language = self._extract_epub_language(book)
        cover_path = self._extract_epub_cover(book, source_path)
        chapters = self._extract_epub_chapters(book)

        return Book(
            title=title,
            author=author,
            language=language,
            chapters=chapters,
            cover_path=cover_path,
            source_path=source_path,
        )

    @staticmethod
    def _extract_epub_title(book: epub.EpubBook) -> str:
        title = book.get_metadata("DC", "title")
        if title:
            return str(title[0][0]).strip()
        return ""

    @staticmethod
    def _extract_epub_author(book: epub.EpubBook) -> str:
        creator = book.get_metadata("DC", "creator")
        if creator:
            return str(creator[0][0]).strip()
        return ""

    @staticmethod
    def _extract_epub_language(book: epub.EpubBook) -> str:
        lang = book.get_metadata("DC", "language")
        if lang:
            return str(lang[0][0]).strip()
        return "en"

    @staticmethod
    def _extract_epub_cover(
        book: epub.EpubBook, source_path: Path
    ) -> Path | None:
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

    def _extract_epub_chapters(
        self, book: epub.EpubBook
    ) -> tuple[Chapter, ...]:
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
    # HTML parsing (for MOBI v7/AZW files)
    # ------------------------------------------------------------------

    def _parse_html(
        self, html_path: Path, source_path: Path
    ) -> Book:
        """Parse an extracted HTML file (from MOBI v7/AZW)."""
        try:
            content = html_path.read_text(
                encoding="utf-8", errors="replace"
            )
        except Exception as exc:
            raise EbookError(
                f"Failed to read extracted HTML: {exc}"
            ) from exc

        # Try to read metadata from content.opf (sibling of HTML)
        opf_path = html_path.parent / "content.opf"
        opf_meta = self._parse_opf_metadata(opf_path)

        title = (
            opf_meta.get("title")
            or self._extract_html_title(content)
            or source_path.stem
        )
        author = opf_meta.get("author", "")
        language = opf_meta.get("language", "en")

        # Try to read TOC from toc.ncx (sibling of HTML)
        ncx_path = html_path.parent / "toc.ncx"
        ncx_entries = self._parse_ncx_toc(ncx_path)

        chapters = self._extract_html_chapters(content, ncx_entries)

        return Book(
            title=title,
            author=author,
            language=language,
            chapters=chapters,
            source_path=source_path,
        )

    @staticmethod
    def _extract_html_title(html_content: str) -> str:
        """Extract title from <title> tag or first <h1>/<h2>/<h3>."""
        match = re.search(
            r"<title[^>]*>(.*?)</title>",
            html_content,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            text = re.sub(r"<[^>]+>", "", match.group(1))
            text = html.unescape(text).strip()
            if text:
                return text
        return ""

    # ------------------------------------------------------------------
    # OPF metadata extraction (for MOBI v7 HTML path)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_opf_metadata(
        opf_path: Path,
    ) -> dict[str, str]:
        """Parse OPF file and return metadata dict with title, author, language."""
        try:
            tree = ET.parse(opf_path)
        except (ET.ParseError, OSError):
            return {}

        ns = {
            "dc": "http://purl.org/dc/elements/1.1/",
            "opf": "http://www.idpf.org/2007/opf",
        }
        root = tree.getroot()

        metadata: dict[str, str] = {}

        lang_el = root.find(".//dc:language", ns)
        if lang_el is not None and lang_el.text:
            metadata["language"] = lang_el.text.strip()

        creator_el = root.find(".//dc:creator", ns)
        if creator_el is not None and creator_el.text:
            metadata["author"] = creator_el.text.strip()

        title_el = root.find(".//dc:title", ns)
        if title_el is not None and title_el.text:
            metadata["title"] = title_el.text.strip()

        return metadata

    # ------------------------------------------------------------------
    # NCX table-of-contents extraction (for MOBI v7 HTML path)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ncx_toc(
        ncx_path: Path,
    ) -> list[tuple[str, int]]:
        """Parse NCX file and return ordered (title, filepos) entries.

        Returns a list of (chapter_title, filepos_int) tuples ordered by
        playOrder, suitable for splitting a single HTML file into chapters.
        """
        try:
            tree = ET.parse(ncx_path)
        except (ET.ParseError, OSError):
            return []

        ns = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}
        root = tree.getroot()

        entries: list[tuple[str, int]] = []
        for nav_point in root.findall(".//ncx:navPoint", ns):
            label_el = nav_point.find("ncx:navLabel/ncx:text", ns)
            content_el = nav_point.find("ncx:content", ns)

            title = ""
            if label_el is not None and label_el.text:
                title = label_el.text.strip()

            filepos = -1
            if content_el is not None:
                src = content_el.get("src", "")
                match = re.search(r"filepos(\d+)", src)
                if match:
                    filepos = int(match.group(1))

            if filepos >= 0:
                entries.append((title, filepos))

        return entries

    def _extract_html_chapters(
        self,
        html_content: str,
        ncx_entries: list[tuple[str, int]] | None = None,
    ) -> tuple[Chapter, ...]:
        """Split HTML content into chapters.

        Strategy (in order of preference):
        1. NCX table-of-contents entries with filepos anchors.
        2. Heading tags (h1-h3).
        3. Fallback: single chapter for entire content.
        """
        # Strategy 1: NCX-based splitting
        if ncx_entries:
            return self._split_html_by_ncx(html_content, ncx_entries)

        # Strategy 2: Heading-based splitting
        heading_pattern = re.compile(
            r"<(h[1-3])[^>]*>(.*?)</\1>",
            re.IGNORECASE | re.DOTALL,
        )
        matches = list(heading_pattern.finditer(html_content))

        if matches:
            return self._split_html_by_headings(
                html_content, matches
            )

        # Strategy 3: Single chapter fallback
        segments = self._html_to_segments(html_content, 0)
        if segments:
            return (
                Chapter(
                    title="Chapter 1",
                    index=0,
                    segments=segments,
                ),
            )
        return ()

    # ------------------------------------------------------------------
    # HTML chapter splitting strategies
    # ------------------------------------------------------------------

    def _split_html_by_ncx(
        self,
        html_content: str,
        ncx_entries: list[tuple[str, int]],
    ) -> tuple[Chapter, ...]:
        """Split HTML using NCX filepos anchors as chapter boundaries."""
        chapters: list[Chapter] = []
        seg_index = 0

        for idx, (title, filepos) in enumerate(ncx_entries):
            anchor_id = f"filepos{filepos}"
            start = html_content.find(f'id="{anchor_id}"')
            if start < 0:
                start = html_content.find(f"id='{anchor_id}'")
            if start < 0:
                continue

            # End is the next anchor or end of content
            if idx + 1 < len(ncx_entries):
                next_anchor = f'filepos{ncx_entries[idx + 1][1]}'
                end = html_content.find(next_anchor, start + 1)
                if end < 0:
                    end = len(html_content)
            else:
                end = len(html_content)

            section_html = html_content[start:end]
            segments, seg_index = self._extract_segments(
                section_html, seg_index
            )

            if segments:
                chapters.append(
                    Chapter(
                        title=title or f"Section {idx + 1}",
                        index=idx,
                        segments=segments,
                    )
                )

        return tuple(chapters)

    def _split_html_by_headings(
        self,
        html_content: str,
        matches: list[re.Match[str]],
    ) -> tuple[Chapter, ...]:
        """Split HTML using h1-h3 heading tags as chapter boundaries."""
        chapters: list[Chapter] = []
        seg_index = 0

        for idx, match in enumerate(matches):
            heading = re.sub(
                r"<[^>]+>", "", match.group(2)
            ).strip()

            start = match.end()
            next_start = (
                matches[idx + 1].start()
                if idx + 1 < len(matches)
                else len(html_content)
            )
            section_html = html_content[start:next_start]

            segments, seg_index = self._extract_segments(
                section_html, seg_index
            )

            if segments:
                chapters.append(
                    Chapter(
                        title=heading or f"Section {idx + 1}",
                        index=idx,
                        segments=segments,
                    )
                )

        return tuple(chapters)

    # ------------------------------------------------------------------
    # Shared XHTML/HTML content processing
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
        """Split XHTML/HTML content into TextSegment objects."""
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

    @classmethod
    def _html_to_segments(
        cls, html_content: str, start_index: int
    ) -> tuple[TextSegment, ...]:
        """Convert full HTML to TextSegment objects."""
        segments, _ = cls._extract_segments(html_content, start_index)
        return segments
