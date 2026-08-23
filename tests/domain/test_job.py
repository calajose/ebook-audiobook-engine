"""Tests for job state machine and AudiobookJob."""

from __future__ import annotations

import pytest

from audiobook_engine.domain.job import (
    AudiobookJob,
    InvalidStateTransition,
    JobState,
)


class TestJobState:
    def test_all_states_exist(self) -> None:
        expected = {
            "created",
            "analyzing",
            "ready",
            "synthesizing",
            "assembling",
            "completed",
            "failed",
            "cancelled",
        }
        assert {s.value for s in JobState} == expected


class TestAudiobookJob:
    def test_default_state(self) -> None:
        job = AudiobookJob()
        assert job.state == JobState.CREATED
        assert job.id
        assert job.completed_segments == 0
        assert job.total_segments == 0

    def test_valid_transition_created_to_analyzing(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.ANALYZING)
        assert job.state == JobState.ANALYZING

    def test_valid_full_lifecycle(self) -> None:
        job = AudiobookJob()
        for state in [
            JobState.ANALYZING,
            JobState.READY,
            JobState.SYNTHESIZING,
            JobState.ASSEMBLING,
            JobState.COMPLETED,
        ]:
            job.transition(state)
        assert job.state == JobState.COMPLETED

    def test_invalid_transition(self) -> None:
        job = AudiobookJob()
        with pytest.raises(InvalidStateTransition):
            job.transition(JobState.COMPLETED)

    def test_invalid_transition_from_completed(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        job.transition(JobState.ASSEMBLING)
        job.transition(JobState.COMPLETED)
        with pytest.raises(InvalidStateTransition):
            job.transition(JobState.SYNTHESIZING)

    def test_can_resume_from_synthesizing(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        assert job.can_resume()

    def test_can_resume_from_assembling(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        job.transition(JobState.ASSEMBLING)
        assert job.can_resume()

    def test_cannot_resume_from_created(self) -> None:
        job = AudiobookJob()
        assert not job.can_resume()

    def test_cannot_resume_from_completed(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        job.transition(JobState.ASSEMBLING)
        job.transition(JobState.COMPLETED)
        assert not job.can_resume()

    def test_failed_from_synthesizing(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        job.transition(JobState.FAILED)
        assert job.state == JobState.FAILED

    def test_failed_from_analyzing(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.ANALYZING)
        job.transition(JobState.FAILED)
        assert job.state == JobState.FAILED

    def test_failed_from_ready(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.FAILED)
        assert job.state == JobState.FAILED

    def test_cancelled_from_created(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.CANCELLED)
        assert job.state == JobState.CANCELLED

    def test_cancelled_from_analyzing(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.ANALYZING)
        job.transition(JobState.CANCELLED)
        assert job.state == JobState.CANCELLED

    def test_cancelled_from_synthesizing(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        job.transition(JobState.CANCELLED)
        assert job.state == JobState.CANCELLED

    def test_cancelled_from_ready(self) -> None:
        job = AudiobookJob()
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.CANCELLED)
        assert job.state == JobState.CANCELLED

    def test_timestamps_updated(self) -> None:
        job = AudiobookJob()
        t0 = job.created_at
        job.transition(JobState.ANALYZING)
        assert job.updated_at >= t0
