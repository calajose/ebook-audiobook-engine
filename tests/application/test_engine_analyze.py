"""Tests for AudiobookEngine.analyze()."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from audiobook_engine.application.engine import AudiobookEngine
from tests.fixtures.epub_builder import build_epub

if TYPE_CHECKING:
    from pathlib import Path


class TestEngineAnalyze:
    def test_analyze_returns_book_and_chapter_analyses(
        self, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        build_epub(
            epub,
            title="My Book",
            author="Jane Doe",
            language="es",
            chapters=[
                ("Capítulo 1", "Primera oración. Segunda oración."),
                ("Capítulo 2", "Hello world. Another sentence."),
            ],
        )

        from audiobook_engine.infrastructure.ebook.epub_parser import (
            EPUBParser,
        )

        engine = AudiobookEngine(EPUBParser(), MagicMock(), work_dir=tmp_path)
        result = engine.analyze(epub)

        assert result.book.title == "My Book"
        assert result.book.author == "Jane Doe"
        assert result.book.language == "es"

        assert len(result.chapter_analyses) == 2
        assert result.chapter_analyses[0].title == "Capítulo 1"
        assert result.chapter_analyses[0].index == 0
        assert result.chapter_analyses[0].chars > 0
        assert result.chapter_analyses[0].words > 0
        assert result.chapter_analyses[0].chunks >= 1

        assert result.chapter_analyses[1].title == "Capítulo 2"

    def test_analyze_with_chapter_filter(
        self, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        build_epub(
            epub,
            chapters=[
                ("Chapter 1", "First chapter content."),
                ("Chapter 2", "Second chapter content."),
                ("Chapter 3", "Third chapter content."),
            ],
        )

        from audiobook_engine.infrastructure.ebook.epub_parser import (
            EPUBParser,
        )

        engine = AudiobookEngine(EPUBParser(), MagicMock(), work_dir=tmp_path)
        result = engine.analyze(epub, chapter_indices=[0, 2])

        assert len(result.chapter_analyses) == 2
        assert result.chapter_analyses[0].index == 0
        assert result.chapter_analyses[0].title == "Chapter 1"
        assert result.chapter_analyses[1].index == 2
        assert result.chapter_analyses[1].title == "Chapter 3"

    def test_analyze_empty_heading_produces_warning(
        self, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        build_epub(
            epub,
            chapters=[
                ("Chapter 1", "Real content here."),
                ("Section 2", "Some content without a proper heading."),
            ],
        )

        from audiobook_engine.infrastructure.ebook.epub_parser import (
            EPUBParser,
        )

        engine = AudiobookEngine(EPUBParser(), MagicMock(), work_dir=tmp_path)
        result = engine.analyze(epub)

        # "Section 2" starts with "Section " so it triggers the warning
        assert any("no heading" in w.lower() for w in result.warnings)

    def test_analyze_single_segment_warning(
        self, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        build_epub(
            epub,
            chapters=[
                ("Chapter 1", "Only one sentence."),
            ],
        )

        from audiobook_engine.infrastructure.ebook.epub_parser import (
            EPUBParser,
        )

        engine = AudiobookEngine(EPUBParser(), MagicMock(), work_dir=tmp_path)
        result = engine.analyze(epub)

        assert any("1 segment" in w for w in result.warnings)

    def test_analyze_no_audio_files_created(
        self, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        build_epub(epub, chapters=[("Ch 1", "Some text.")])

        from audiobook_engine.infrastructure.ebook.epub_parser import (
            EPUBParser,
        )

        engine = AudiobookEngine(EPUBParser(), MagicMock(), work_dir=tmp_path)
        engine.analyze(epub)

        wav_files = list(tmp_path.rglob("*.wav"))
        assert wav_files == []

    def test_analyze_source_file_populated(
        self, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        build_epub(
            epub,
            chapters=[
                ("Chapter 1", "Content one."),
                ("Chapter 2", "Content two."),
            ],
        )

        from audiobook_engine.infrastructure.ebook.epub_parser import (
            EPUBParser,
        )

        engine = AudiobookEngine(EPUBParser(), MagicMock(), work_dir=tmp_path)
        result = engine.analyze(epub)

        for ca in result.chapter_analyses:
            assert ca.source_file is not None
            assert ca.source_file.endswith(".xhtml")

    def test_analyze_max_chars_affects_chunk_count(
        self, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        long_text = "Sentence one. " * 50
        build_epub(epub, chapters=[("Ch 1", long_text)])

        from audiobook_engine.infrastructure.ebook.epub_parser import (
            EPUBParser,
        )

        engine = AudiobookEngine(EPUBParser(), MagicMock(), work_dir=tmp_path)
        result_default = engine.analyze(epub, max_chars=500)
        result_small = engine.analyze(epub, max_chars=100)

        default_chunks = result_default.chapter_analyses[0].chunks
        small_chunks = result_small.chapter_analyses[0].chunks
        assert small_chunks > default_chunks
