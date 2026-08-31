"""Tests for enhanced inspect CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from audiobook_engine.application.engine import (
    BookAnalysis,
    ChapterAnalysis,
)
from audiobook_engine.interfaces.cli.main import (
    _parse_chapter_indices,
    app,
)

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


class TestParseChapterIndices:
    def test_single_index(self) -> None:
        assert _parse_chapter_indices("0") == [0]

    def test_multiple_indices(self) -> None:
        assert _parse_chapter_indices("0,2,5") == [0, 2, 5]

    def test_range(self) -> None:
        assert _parse_chapter_indices("1-3") == [1, 2, 3]

    def test_mixed(self) -> None:
        assert _parse_chapter_indices("0,2-4,7") == [0, 2, 3, 4, 7]

    def test_deduplicates(self) -> None:
        assert _parse_chapter_indices("0,0,1") == [0, 1]

    def test_sorts_output(self) -> None:
        assert _parse_chapter_indices("5,1,3") == [1, 3, 5]


def _make_book_analysis(
    title: str = "Test",
    author: str = "",
    language: str = "en",
    chapters: list[ChapterAnalysis] | None = None,
    warnings: list[str] | None = None,
    cover_path: Path | None = None,
) -> BookAnalysis:
    mock_book = MagicMock()
    mock_book.title = title
    mock_book.author = author
    mock_book.language = language
    mock_book.cover_path = cover_path
    mock_book.chapters = [MagicMock() for _ in (chapters or [])]
    return BookAnalysis(
        book=mock_book,
        chapter_analyses=tuple(chapters or []),
        warnings=tuple(warnings or []),
    )


class TestInspectCommand:
    def test_missing_file_shows_error(self) -> None:
        result = runner.invoke(app, ["inspect", "nonexistent.epub"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    @patch("audiobook_engine.interfaces.cli.main.AudiobookEngine")
    def test_inspect_displays_chapter_table(
        self, mock_engine_cls: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        epub.write_text("fake epub")

        mock_engine = mock_engine_cls.return_value
        mock_engine.analyze.return_value = _make_book_analysis(
            title="Test Book",
            author="Author",
            chapters=[
                ChapterAnalysis(
                    index=0,
                    title="Chapter 1",
                    source_file="ch1.xhtml",
                    segments=5,
                    chars=1200,
                    words=200,
                    chunks=3,
                ),
                ChapterAnalysis(
                    index=1,
                    title="Chapter 2",
                    source_file="ch2.xhtml",
                    segments=3,
                    chars=800,
                    words=130,
                    chunks=2,
                ),
            ],
        )

        result = runner.invoke(app, ["inspect", str(epub)])
        assert result.exit_code == 0
        assert "Test Book" in result.output
        assert "Author" in result.output
        assert "Chapter 1" in result.output
        assert "Chapter 2" in result.output
        assert "1,200" in result.output
        assert "800" in result.output

    @patch("audiobook_engine.interfaces.cli.main.AudiobookEngine")
    def test_inspect_shows_warnings(
        self, mock_engine_cls: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        epub.write_text("fake epub")

        mock_engine = mock_engine_cls.return_value
        mock_engine.analyze.return_value = _make_book_analysis(
            chapters=[
                ChapterAnalysis(
                    index=0,
                    title="Section 1",
                    source_file=None,
                    segments=1,
                    chars=100,
                    words=20,
                    chunks=1,
                ),
            ],
            warnings=["Chapter 0: no heading detected"],
        )

        result = runner.invoke(app, ["inspect", str(epub)])
        assert result.exit_code == 0
        assert "Warning" in result.output
        assert "no heading detected" in result.output

    @patch("audiobook_engine.interfaces.cli.main.AudiobookEngine")
    def test_inspect_with_chapters_filter(
        self, mock_engine_cls: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        epub.write_text("fake epub")

        mock_engine = mock_engine_cls.return_value
        mock_engine.analyze.return_value = _make_book_analysis(
            chapters=[
                ChapterAnalysis(
                    index=2,
                    title="Chapter 3",
                    source_file="ch3.xhtml",
                    segments=4,
                    chars=900,
                    words=150,
                    chunks=2,
                ),
            ],
        )

        result = runner.invoke(
            app, ["inspect", str(epub), "--chapters", "2"]
        )
        assert result.exit_code == 0
        mock_engine.analyze.assert_called_once()
        call_args = mock_engine.analyze.call_args
        assert call_args[0][1] == [2]

    @patch("audiobook_engine.interfaces.cli.main.AudiobookEngine")
    def test_inspect_with_range_filter(
        self, mock_engine_cls: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        epub.write_text("fake epub")

        mock_engine = mock_engine_cls.return_value
        mock_engine.analyze.return_value = _make_book_analysis()

        result = runner.invoke(
            app, ["inspect", str(epub), "--chapters", "1-3"]
        )
        assert result.exit_code == 0
        call_args = mock_engine.analyze.call_args
        assert call_args[0][1] == [1, 2, 3]

    @patch("audiobook_engine.interfaces.cli.main.AudiobookEngine")
    def test_inspect_no_audio_files_created(
        self, mock_engine_cls: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        epub.write_text("fake epub")

        mock_engine = mock_engine_cls.return_value
        mock_engine.analyze.return_value = _make_book_analysis()

        runner.invoke(app, ["inspect", str(epub)])
        mock_engine.create_job.assert_not_called()
        mock_engine.run.assert_not_called()

    @patch("audiobook_engine.interfaces.cli.main.AudiobookEngine")
    def test_inspect_displays_totals(
        self, mock_engine_cls: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        epub.write_text("fake epub")

        mock_engine = mock_engine_cls.return_value
        mock_engine.analyze.return_value = _make_book_analysis(
            chapters=[
                ChapterAnalysis(
                    index=0,
                    title="Ch 1",
                    source_file=None,
                    segments=2,
                    chars=500,
                    words=80,
                    chunks=2,
                ),
                ChapterAnalysis(
                    index=1,
                    title="Ch 2",
                    source_file=None,
                    segments=3,
                    chars=700,
                    words=110,
                    chunks=3,
                ),
            ],
        )

        result = runner.invoke(app, ["inspect", str(epub)])
        assert result.exit_code == 0
        assert "1,200" in result.output  # total chars
        assert "190" in result.output  # total words
        assert "5" in result.output  # total chunks

    @patch("audiobook_engine.interfaces.cli.main.AudiobookEngine")
    def test_inspect_shows_no_cover(
        self, mock_engine_cls: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        epub.write_text("fake epub")

        mock_engine = mock_engine_cls.return_value
        mock_engine.analyze.return_value = _make_book_analysis(cover_path=None)

        result = runner.invoke(app, ["inspect", str(epub)])
        assert result.exit_code == 0
        assert "Cover: None" in result.output

    @patch("audiobook_engine.interfaces.cli.main.AudiobookEngine")
    def test_inspect_shows_cover_filename(
        self, mock_engine_cls: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        epub.write_text("fake epub")
        cover = tmp_path / "test_cover.jpg"
        cover.write_bytes(b"fake image")

        mock_engine = mock_engine_cls.return_value
        mock_engine.analyze.return_value = _make_book_analysis(cover_path=cover)

        result = runner.invoke(app, ["inspect", str(epub)])
        assert result.exit_code == 0
        assert "Cover: test_cover.jpg" in result.output
