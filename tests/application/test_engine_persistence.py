"""Tests for engine persistence integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from audiobook_engine.application.engine import AudiobookEngine
from audiobook_engine.domain.job import JobState
from audiobook_engine.infrastructure.ebook.epub_parser import EPUBParser
from audiobook_engine.infrastructure.persistence.job_store import (
    load_job,
    save_job,
)
from tests.fixtures.epub_builder import build_epub

if TYPE_CHECKING:
    from pathlib import Path


def _make_engine(tmp_path: Path) -> AudiobookEngine:
    """Create an engine with a mock TTS backend."""
    from unittest.mock import MagicMock

    parser = EPUBParser()
    tts = MagicMock()

    def _synthesize(
        text: str, language: str, voice: str, output: Path
    ) -> None:
        import wave

        with wave.open(str(output), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * 22050)

    tts.synthesize.side_effect = _synthesize
    return AudiobookEngine(parser, tts, work_dir=tmp_path)


class TestEnginePersistence:
    def test_create_job_persists_to_disk(
        self, tmp_path: Path
    ) -> None:
        engine = _make_engine(tmp_path)
        epub = tmp_path / "test.epub"
        build_epub(
            epub,
            chapters=[("Ch1", "Hello world.")],
        )

        job = engine.create_job(
            epub, "en", "af_heart", tmp_path / "out.wav"
        )
        loaded = load_job(tmp_path / "jobs" / job.id)
        assert loaded.state == JobState.CREATED
        assert loaded.book_title == "Test Book"

    def test_run_persists_completed_state(
        self, tmp_path: Path
    ) -> None:
        engine = _make_engine(tmp_path)
        epub = tmp_path / "test.epub"
        build_epub(
            epub,
            chapters=[("Ch1", "Hello world.")],
        )

        job = engine.create_job(
            epub, "en", "af_heart", tmp_path / "out.wav"
        )
        engine.run(job.id)

        loaded = load_job(tmp_path / "jobs" / job.id)
        assert loaded.state == JobState.COMPLETED

    def test_resume_from_disk_after_new_engine(
        self, tmp_path: Path
    ) -> None:
        """Simulate process restart: create job in one engine,
        resume in a fresh engine using only disk state."""
        epub = tmp_path / "test.epub"
        build_epub(
            epub,
            chapters=[("Ch1", "Hello world.")],
        )

        # First engine: create and persist a synthesizing job
        engine1 = _make_engine(tmp_path)
        job = engine1.create_job(
            epub, "en", "af_heart", tmp_path / "out.wav"
        )
        # Force into synthesizing state and persist
        job.transition(JobState.ANALYZING)
        job.total_segments = 1
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        job.completed_segments = 0
        save_job(job, job.work_dir)

        # Second engine: fresh instance, no in-memory state
        engine2 = _make_engine(tmp_path)
        resumed = engine2.resume(job.id)
        assert resumed.state == JobState.COMPLETED

    def test_resume_raises_for_unknown_job(
        self, tmp_path: Path
    ) -> None:
        engine = _make_engine(tmp_path)
        from audiobook_engine.domain.exceptions import JobError

        with pytest.raises(JobError, match="Job not found"):
            engine.resume("nonexistent")
