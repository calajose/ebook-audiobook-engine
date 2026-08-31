"""Tests for job persistence layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from audiobook_engine.domain.job import AudiobookJob, JobState
from audiobook_engine.infrastructure.persistence.job_store import (
    find_resumable_jobs,
    load_job,
    save_job,
)


def _make_job(
    *,
    job_id: str = "test123",
    state: JobState = JobState.CREATED,
    work_dir: Path | None = None,
) -> AudiobookJob:
    return AudiobookJob(
        id=job_id,
        state=state,
        source_path=Path("/books/test.epub"),
        book_title="Test Book",
        backend="kokoro",
        language="en",
        voice="af_heart",
        work_dir=work_dir or Path("work/jobs/test123"),
        output_path=Path("output/test.wav"),
        total_segments=10,
        completed_segments=5,
    )


class TestSaveAndLoad:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        job = _make_job(work_dir=tmp_path)
        result = save_job(job, tmp_path)
        assert result == tmp_path / "job.json"
        assert result.exists()

    def test_load_returns_correct_job(self, tmp_path: Path) -> None:
        job = _make_job(work_dir=tmp_path)
        save_job(job, tmp_path)
        loaded = load_job(tmp_path)

        assert loaded.id == job.id
        assert loaded.state == job.state
        assert loaded.book_title == job.book_title
        assert loaded.backend == job.backend
        assert loaded.language == job.language
        assert loaded.voice == job.voice
        assert loaded.speed == job.speed
        assert loaded.paragraph_pause_ms == job.paragraph_pause_ms
        assert loaded.chapter_pause_ms == job.chapter_pause_ms
        assert loaded.total_segments == job.total_segments
        assert loaded.completed_segments == job.completed_segments

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        job = _make_job(work_dir=nested)
        save_job(job, nested)
        assert (nested / "job.json").exists()

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No job.json"):
            load_job(tmp_path)

    def test_round_trip_preserves_none_fields(
        self, tmp_path: Path
    ) -> None:
        job = AudiobookJob(
            id="minimal",
            source_path=None,
            output_path=None,
            work_dir=tmp_path,
        )
        save_job(job, tmp_path)
        loaded = load_job(tmp_path)
        assert loaded.source_path is None
        assert loaded.output_path is None
        assert loaded.error_message is None


class TestFindResumableJobs:
    def test_finds_synthesizing_job(self, tmp_path: Path) -> None:
        work = tmp_path / "jobs" / "j1"
        job = _make_job(
            state=JobState.SYNTHESIZING, work_dir=work
        )
        save_job(job, work)
        results = find_resumable_jobs(tmp_path / "jobs")
        assert len(results) == 1
        assert results[0][0] == job.id

    def test_finds_assembling_job(self, tmp_path: Path) -> None:
        work = tmp_path / "jobs" / "j1"
        job = _make_job(
            state=JobState.ASSEMBLING, work_dir=work
        )
        save_job(job, work)
        results = find_resumable_jobs(tmp_path / "jobs")
        assert len(results) == 1

    def test_ignores_completed_job(self, tmp_path: Path) -> None:
        work = tmp_path / "jobs" / "j1"
        job = _make_job(
            state=JobState.COMPLETED, work_dir=work
        )
        save_job(job, work)
        results = find_resumable_jobs(tmp_path / "jobs")
        assert len(results) == 0

    def test_ignores_failed_job(self, tmp_path: Path) -> None:
        work = tmp_path / "jobs" / "j1"
        job = _make_job(state=JobState.FAILED, work_dir=work)
        save_job(job, work)
        results = find_resumable_jobs(tmp_path / "jobs")
        assert len(results) == 0

    def test_nonexistent_dir_returns_empty(
        self, tmp_path: Path
    ) -> None:
        results = find_resumable_jobs(tmp_path / "nope")
        assert results == []

    def test_ignores_dirs_without_job_json(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "not-a-job").mkdir()
        results = find_resumable_jobs(tmp_path)
        assert results == []


class TestFindAllJobs:
    def test_finds_all_states(self, tmp_path: Path) -> None:
        states = [
            JobState.CREATED,
            JobState.SYNTHESIZING,
            JobState.COMPLETED,
            JobState.FAILED,
        ]
        for i, state in enumerate(states):
            work = tmp_path / "jobs" / f"j{i}"
            job = _make_job(job_id=f"j{i}", state=state, work_dir=work)
            save_job(job, work)

        from audiobook_engine.infrastructure.persistence.job_store import (
            find_all_jobs,
        )

        results = find_all_jobs(tmp_path / "jobs")
        assert len(results) == len(states)

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        from audiobook_engine.infrastructure.persistence.job_store import (
            find_all_jobs,
        )

        results = find_all_jobs(tmp_path / "nope")
        assert results == []

    def test_ignores_dirs_without_job_json(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-job").mkdir()
        from audiobook_engine.infrastructure.persistence.job_store import (
            find_all_jobs,
        )

        results = find_all_jobs(tmp_path)
        assert results == []
