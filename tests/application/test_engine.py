"""Tests for AudiobookEngine application services."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from audiobook_engine.application.engine import AudiobookEngine
from audiobook_engine.domain.exceptions import JobError
from audiobook_engine.domain.job import JobState
from audiobook_engine.domain.models import (
    BackendCapabilities,
    Language,
    Voice,
)
from audiobook_engine.infrastructure.ebook.epub_parser import EPUBParser
from tests.fixtures.epub_builder import build_epub


def _make_fake_wav(path: Path) -> None:
    """Write a minimal valid WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(struct.pack("<3h", 0, 0, 0))


class FakeTTSBackend:
    """Mock TTS backend that writes fake WAV files."""

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            languages=(Language(code="en-us", name="English"),),
            voices=(
                Voice(
                    id="af_sarah",
                    name="af_sarah",
                    language_code="en-us",
                ),
            ),
        )

    def synthesize(
        self,
        text: str,
        language: str,
        voice: str,
        output_path: Path,
        speed: float = 1.0,
    ) -> None:
        _make_fake_wav(output_path)


@pytest.fixture()
def sample_epub(tmp_path: Path) -> Path:
    path = tmp_path / "test.epub"
    build_epub(
        path,
        title="Test Book",
        author="Test Author",
        language="en",
        chapters=[
            (
                "Chapter One",
                "First sentence. Second sentence.\nThird paragraph.",
            ),
            (
                "Chapter Two",
                "Another chapter with some text.",
            ),
        ],
    )
    return path


@pytest.fixture()
def engine(tmp_path: Path) -> AudiobookEngine:
    return AudiobookEngine(
        parser=EPUBParser(),
        tts_backend=FakeTTSBackend(),
        work_dir=tmp_path / "work",
    )


class TestInspect:
    def test_inspect_returns_book(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        book = engine.inspect(sample_epub)
        assert book.title == "Test Book"
        assert book.author == "Test Author"
        assert len(book.chapters) == 2


class TestCapabilities:
    def test_returns_capabilities(
        self, engine: AudiobookEngine
    ) -> None:
        caps = engine.capabilities()
        assert len(caps.languages) == 1
        assert caps.languages[0].code == "en-us"
        assert len(caps.voices) == 1
        assert caps.voices[0].id == "af_sarah"


class TestCreateJob:
    def test_creates_job(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        job = engine.create_job(
            sample_epub, "en-us", "af_sarah", Path("out.m4b")
        )
        assert job.state == JobState.CREATED
        assert job.book_title == "Test Book"
        assert job.language == "en-us"
        assert job.voice == "af_sarah"
        assert job.source_path == sample_epub

    def test_get_job(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        job = engine.create_job(
            sample_epub, "en-us", "af_sarah", Path("out.m4b")
        )
        found = engine.get_job(job.id)
        assert found.id == job.id

    def test_get_job_not_found(
        self, engine: AudiobookEngine
    ) -> None:
        with pytest.raises(JobError, match="not found"):
            engine.get_job("nonexistent")


class TestRun:
    def test_full_pipeline(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        job = engine.create_job(
            sample_epub, "en-us", "af_sarah", Path("out.m4b")
        )
        result = engine.run(job.id)
        assert result.state == JobState.COMPLETED
        assert result.total_segments > 0
        assert result.completed_segments == result.total_segments

    def test_creates_output_file(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        output = engine._work_dir.parent / "final_output.wav"
        job = engine.create_job(
            sample_epub, "en-us", "af_sarah", output
        )
        engine.run(job.id)
        assert output.exists()

    def test_creates_work_directory(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        job = engine.create_job(
            sample_epub,
            "en-us",
            "af_sarah",
            Path("out.wav"),
            keep_intermediates=True,
        )
        engine.run(job.id)
        work_dir = engine._work_dir / job.id
        assert work_dir.exists()
        assert (work_dir / "segments").exists()
        assert (work_dir / "chapters").exists()
        assert (work_dir / "output").exists()

    def test_cleanup_intermediates_by_default(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        job = engine.create_job(
            sample_epub, "en-us", "af_sarah", Path("out.wav")
        )
        engine.run(job.id)
        work_dir = engine._work_dir / job.id
        assert work_dir.exists()
        assert not (work_dir / "segments").exists()
        assert not (work_dir / "chapters").exists()
        assert not (work_dir / "output").exists()


class TestCancel:
    def test_cancel_created_job(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        job = engine.create_job(
            sample_epub, "en-us", "af_sarah", Path("out.wav")
        )
        result = engine.cancel(job.id)
        assert result.state == JobState.CANCELLED

    def test_cancel_completed_job_raises(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        job = engine.create_job(
            sample_epub, "en-us", "af_sarah", Path("out.wav")
        )
        engine.run(job.id)
        with pytest.raises(JobError, match="Cannot cancel"):
            engine.cancel(job.id)


class TestResume:
    def test_resume_not_resumable_raises(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        job = engine.create_job(
            sample_epub, "en-us", "af_sarah", Path("out.wav")
        )
        with pytest.raises(JobError, match="cannot be resumed"):
            engine.resume(job.id)

    def test_resume_from_synthesizing(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        job = engine.create_job(
            sample_epub, "en-us", "af_sarah", Path("out.wav")
        )
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        assert job.can_resume()
        result = engine.resume(job.id)
        assert result.state == JobState.COMPLETED

    def test_resume_from_assembling(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        """Test that resume retries assembly when state is ASSEMBLING."""
        job = engine.create_job(
            sample_epub, "en-us", "af_sarah", Path("out.wav")
        )
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        job.transition(JobState.ASSEMBLING)
        assert job.can_resume()
        result = engine.resume(job.id)
        assert result.state == JobState.COMPLETED
