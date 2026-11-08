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
        result = runner.invoke(app, ["convert", "nonexistent.epub", "-v", "af_heart"])
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
                "-l",
                "en-us",
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
    def test_list_voices_shows_table(self, mock_engine_cls: MagicMock) -> None:
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
    def test_list_voices_filter_by_language(self, mock_engine_cls: MagicMock) -> None:
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

    def test_resume_list_no_jobs(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["resume", "--work-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No jobs found" in result.output

    def test_resume_list_with_resumable_jobs(self, tmp_path: Path) -> None:
        from audiobook_engine.domain.job import AudiobookJob, JobState
        from audiobook_engine.infrastructure.persistence.job_store import save_job

        job_dir = tmp_path / "jobs" / "job123"
        job = AudiobookJob(
            id="job123",
            state=JobState.SYNTHESIZING,
            book_title="My Test Book",
            work_dir=job_dir,
            total_segments=10,
            completed_segments=3,
        )
        save_job(job, job_dir)

        result = runner.invoke(app, ["resume", "--work-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "job123" in result.output
        assert "My Test Book" in result.output
        assert "synthesizing" in result.output


class TestCleanCommand:
    def test_clean_existing_work_dir(self, tmp_path: Path) -> None:
        work = tmp_path / "work"
        work.mkdir()
        (work / "some_file.txt").write_text("debris")

        result = runner.invoke(app, ["clean", "--work-dir", str(work)])
        assert result.exit_code == 0
        assert not work.exists()
        assert "Cleaned working directory" in result.output

    def test_clean_nonexistent_work_dir(self, tmp_path: Path) -> None:
        work = tmp_path / "nonexistent"
        result = runner.invoke(app, ["clean", "--work-dir", str(work)])
        assert result.exit_code == 0
        assert "does not exist" in result.output

    def test_clean_specific_job(self, tmp_path: Path) -> None:
        work = tmp_path / "work"
        jobs = work / "jobs"
        job1 = jobs / "job-aaa"
        job2 = jobs / "job-bbb"
        job1.mkdir(parents=True)
        job2.mkdir(parents=True)
        (job1 / "job.json").write_text("{}")
        (job2 / "job.json").write_text("{}")

        result = runner.invoke(app, ["clean", "--work-dir", str(work), "job-aaa"])
        assert result.exit_code == 0
        assert not job1.exists()
        assert job2.exists()
        assert "Cleaned job" in result.output

    def test_clean_nonexistent_job(self, tmp_path: Path) -> None:
        work = tmp_path / "work"
        work.mkdir()

        result = runner.invoke(app, ["clean", "--work-dir", str(work), "no-such-job"])
        assert result.exit_code == 0
        assert "not found" in result.output


def _mock_convert_engine(
    mock_create: MagicMock,
    *,
    completed: bool = True,
    total_segments: int = 10,
) -> tuple[MagicMock, MagicMock]:
    """Configure a mocked engine for convert validation tests."""
    engine = mock_create.return_value
    book = MagicMock()
    book.title = "Test Book"
    book.author = "Author"
    book.chapters = [MagicMock(index=0), MagicMock(index=1)]
    engine.inspect.return_value = book

    caps = MagicMock()
    lang = MagicMock(code="en-us")
    voice = MagicMock()
    voice.id = "af_heart"
    voice.language_code = "en-us"
    caps.languages = [lang]
    caps.voices = [voice]
    engine.capabilities.return_value = caps

    job = MagicMock()
    job.id = "job123"
    job.state = JobState.COMPLETED if completed else JobState.FAILED
    job.completed_segments = total_segments
    job.total_segments = total_segments
    job.error_message = None
    engine.create_job.return_value = job
    engine.get_job.return_value = job
    return engine, job


class TestConvertValidation:
    @patch("audiobook_engine.interfaces.cli.main._create_engine")
    def test_out_of_range_chapters_rejected(
        self, mock_create: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "book.epub"
        epub.write_text("fake")
        engine, _ = _mock_convert_engine(mock_create)

        result = runner.invoke(
            app,
            ["convert", str(epub), "-v", "af_heart", "-l", "en-us", "-c", "99"],
        )

        assert result.exit_code == 1
        assert "out of range" in result.output.lower()
        engine.create_job.assert_not_called()

    @patch("audiobook_engine.interfaces.cli.main._create_engine")
    def test_empty_chapter_selection_rejected(
        self, mock_create: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "book.epub"
        epub.write_text("fake")
        engine, _ = _mock_convert_engine(mock_create)

        result = runner.invoke(
            app,
            ["convert", str(epub), "-v", "af_heart", "-l", "en-us", "-c", "3-1"],
        )

        assert result.exit_code == 1
        assert "empty" in result.output.lower()
        engine.create_job.assert_not_called()

    @patch("audiobook_engine.interfaces.cli.main._create_engine")
    def test_unsupported_output_extension_rejected(
        self, mock_create: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "book.epub"
        epub.write_text("fake")
        engine, _ = _mock_convert_engine(mock_create)

        result = runner.invoke(
            app,
            [
                "convert",
                str(epub),
                "-v",
                "af_heart",
                "-l",
                "en-us",
                "-o",
                str(tmp_path / "out.mp3"),
            ],
        )

        assert result.exit_code == 1
        assert "unsupported output extension" in result.output.lower()
        engine.create_job.assert_not_called()

    @patch(
        "audiobook_engine.interfaces.cli.main.ffmpeg.is_available",
        return_value=False,
    )
    @patch("audiobook_engine.interfaces.cli.main._create_engine")
    def test_m4b_without_ffmpeg_rejected(
        self,
        mock_create: MagicMock,
        mock_ffmpeg: MagicMock,
        tmp_path: Path,
    ) -> None:
        epub = tmp_path / "book.epub"
        epub.write_text("fake")
        engine, _ = _mock_convert_engine(mock_create)

        result = runner.invoke(
            app,
            [
                "convert",
                str(epub),
                "-v",
                "af_heart",
                "-l",
                "en-us",
                "-o",
                str(tmp_path / "out.m4b"),
            ],
        )

        assert result.exit_code == 1
        assert "requires ffmpeg" in result.output.lower()
        engine.create_job.assert_not_called()

    @patch("audiobook_engine.interfaces.cli.main._create_engine")
    def test_max_chars_forwarded_to_create_job(
        self, mock_create: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "book.epub"
        epub.write_text("fake")
        engine, _ = _mock_convert_engine(mock_create)

        runner.invoke(
            app,
            [
                "convert",
                str(epub),
                "-v",
                "af_heart",
                "-l",
                "en-us",
                "-o",
                str(tmp_path / "out.wav"),
                "--max-chars",
                "120",
            ],
        )

        kwargs = engine.create_job.call_args[1]
        assert kwargs["max_chars"] == 120

    @patch("audiobook_engine.interfaces.cli.main._create_engine")
    def test_warns_when_no_audible_content(
        self, mock_create: MagicMock, tmp_path: Path
    ) -> None:
        epub = tmp_path / "book.epub"
        epub.write_text("fake")
        _, job = _mock_convert_engine(mock_create, total_segments=0)
        job.total_segments = 0

        result = runner.invoke(
            app,
            [
                "convert",
                str(epub),
                "-v",
                "af_heart",
                "-l",
                "en-us",
                "-o",
                str(tmp_path / "out.wav"),
            ],
        )

        assert result.exit_code == 0
        assert "no audible content" in result.output.lower()
