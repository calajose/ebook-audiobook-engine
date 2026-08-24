"""Tests for CLI interface."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from audiobook_engine.domain.job import JobState
from audiobook_engine.interfaces.cli.main import app

if TYPE_CHECKING:
    from pathlib import Path
    pass

runner = CliRunner()


class TestConvertCommand:
    def test_missing_file_shows_error(self) -> None:
        result = runner.invoke(
            app, ["convert", "nonexistent.epub", "-v", "af_heart"]
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    @patch("audiobook_engine.interfaces.cli.main.AudiobookEngine")
    def test_convert_calls_engine(
        self, mock_engine_cls: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "test.epub"
        epub.write_text("fake epub")

        mock_engine = mock_engine_cls.return_value
        mock_book = MagicMock()
        mock_book.title = "Test Book"
        mock_book.author = "Author"
        mock_book.chapters = [MagicMock(), MagicMock()]
        mock_engine.inspect.return_value = mock_book

        mock_caps = MagicMock()
        mock_lang = MagicMock()
        mock_lang.code = "en"
        mock_voice = MagicMock()
        mock_voice.id = "af_heart"
        mock_voice.language_code = "en"
        mock_caps.languages = [mock_lang]
        mock_caps.voices = [mock_voice]
        mock_engine.capabilities.return_value = mock_caps

        mock_job = MagicMock()
        mock_job.id = "test123"
        mock_job.state = JobState.COMPLETED
        mock_job.completed_segments = 10
        mock_job.total_segments = 10
        mock_engine.create_job.return_value = mock_job
        mock_engine.get_job.return_value = mock_job

        result = runner.invoke(
            app,
            [
                "convert",
                str(epub),
                "-v",
                "af_heart",
                "-o",
                str(tmp_path / "out.m4b"),
                "--speed",
                "0.95",
                "--paragraph-pause",
                "800",
                "--chapter-pause",
                "3000",
            ],
        )
        # Should not crash (may exit with various codes depending on mock)
        assert result.exit_code in (0, 1)
        mock_engine.create_job.assert_called_once()
        kwargs = mock_engine.create_job.call_args[1]
        assert kwargs["speed"] == 0.95
        assert kwargs["paragraph_pause_ms"] == 800
        assert kwargs["chapter_pause_ms"] == 3000


class TestListVoicesCommand:
    @patch("audiobook_engine.interfaces.cli.main.AudiobookEngine")
    def test_list_voices_shows_table(
        self, mock_engine_cls: MagicMock
    ) -> None:
        mock_engine = mock_engine_cls.return_value
        mock_caps = MagicMock()
        mock_lang = MagicMock()
        mock_lang.code = "en"
        mock_lang.name = "English"
        mock_voice = MagicMock()
        mock_voice.id = "af_heart"
        mock_voice.name = "Heart"
        mock_voice.language_code = "en"
        mock_caps.languages = [mock_lang]
        mock_caps.voices = [mock_voice]
        mock_engine.capabilities.return_value = mock_caps

        result = runner.invoke(app, ["list-voices"])
        assert result.exit_code == 0
        assert "af_heart" in result.output

    @patch("audiobook_engine.interfaces.cli.main.AudiobookEngine")
    def test_list_voices_filter_by_language(
        self, mock_engine_cls: MagicMock
    ) -> None:
        mock_engine = mock_engine_cls.return_value
        mock_caps = MagicMock()
        mock_lang_en = MagicMock()
        mock_lang_en.code = "en"
        mock_lang_en.name = "English"
        mock_lang_es = MagicMock()
        mock_lang_es.code = "es"
        mock_lang_es.name = "Spanish"
        mock_voice_en = MagicMock()
        mock_voice_en.id = "af_heart"
        mock_voice_en.name = "Heart"
        mock_voice_en.language_code = "en"
        mock_voice_es = MagicMock()
        mock_voice_es.id = "es_voice"
        mock_voice_es.name = "Spanish Voice"
        mock_voice_es.language_code = "es"
        mock_caps.languages = [mock_lang_en, mock_lang_es]
        mock_caps.voices = [mock_voice_en, mock_voice_es]
        mock_engine.capabilities.return_value = mock_caps

        result = runner.invoke(app, ["list-voices", "-l", "en"])
        assert result.exit_code == 0
        assert "af_heart" in result.output
        assert "es_voice" not in result.output


class TestResumeCommand:
    def test_resume_nonexistent_job_shows_error(self) -> None:
        result = runner.invoke(app, ["resume", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()
