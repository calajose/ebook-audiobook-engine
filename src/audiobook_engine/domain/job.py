"""Job state machine and audiobook job model."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class JobState(enum.Enum):
    """Possible states of an audiobook conversion job."""

    CREATED = "created"
    ANALYZING = "analyzing"
    READY = "ready"
    SYNTHESIZING = "synthesizing"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Valid state transitions: from_state -> set of possible next states
_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.CREATED: {JobState.ANALYZING, JobState.CANCELLED},
    JobState.ANALYZING: {
        JobState.READY,
        JobState.FAILED,
        JobState.CANCELLED,
    },
    JobState.READY: {
        JobState.SYNTHESIZING,
        JobState.FAILED,
        JobState.CANCELLED,
    },
    JobState.SYNTHESIZING: {
        JobState.ASSEMBLING,
        JobState.FAILED,
        JobState.CANCELLED,
    },
    JobState.ASSEMBLING: {JobState.COMPLETED, JobState.FAILED},
    JobState.COMPLETED: {JobState.SYNTHESIZING},
    JobState.FAILED: set(),
    JobState.CANCELLED: set(),
}

_RESUMABLE_STATES: frozenset[JobState] = frozenset(
    {JobState.SYNTHESIZING, JobState.ASSEMBLING}
)


@dataclass
class AudiobookJob:
    """Represents a single audiobook conversion job."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: JobState = JobState.CREATED

    # Source
    source_path: Path | None = None
    book_title: str = ""

    # Backend selection
    backend: str = ""
    language: str = ""
    voice: str = ""

    # Prosody & pauses
    speed: float = 1.0
    paragraph_pause_ms: int = 700
    chapter_pause_ms: int = 2500
    scene_break_pause_ms: int = 1500
    chapter_title_pause_ms: int = 1200

    # Paths
    work_dir: Path | None = None
    output_path: Path | None = None

    # Progress
    total_segments: int = 0
    completed_segments: int = 0
    chapter_indices: list[int] | None = None

    # Cleanup control
    keep_intermediates: bool = False

    # Error tracking
    error_message: str | None = None

    # Timestamps
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def transition(self, target: JobState) -> None:
        """Transition to a new state, raising on invalid transitions."""
        allowed = _TRANSITIONS.get(self.state, set())
        if target not in allowed:
            raise InvalidStateTransition(
                f"Cannot transition from {self.state.value} "
                f"to {target.value}"
            )
        self.state = target
        self.updated_at = datetime.now(timezone.utc)

    def can_resume(self) -> bool:
        """Return True if the job is in a state that supports resume."""
        return self.state in _RESUMABLE_STATES


class InvalidStateTransition(Exception):
    """Raised when a job state transition is not allowed."""
