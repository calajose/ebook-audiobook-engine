"""Domain layer — data models, protocols, and exceptions.

This layer must NOT import from infrastructure (Kokoro, FFmpeg, HTTP, Calibre).
"""

from audiobook_engine.domain.exceptions import (
    AudioAssemblyError,
    AudiobookEngineError,
    ConfigurationError,
    EbookError,
    JobError,
    TTSBackendError,
)
from audiobook_engine.domain.job import AudiobookJob, InvalidStateTransition, JobState
from audiobook_engine.domain.models import (
    BackendCapabilities,
    Book,
    Chapter,
    Language,
    TextSegment,
    Voice,
)
from audiobook_engine.domain.protocols import EbookParser, TTSBackend

__all__ = [
    "AudioAssemblyError",
    "AudiobookEngineError",
    "AudiobookJob",
    "BackendCapabilities",
    "Book",
    "Chapter",
    "ConfigurationError",
    "EbookError",
    "EbookParser",
    "InvalidStateTransition",
    "JobError",
    "JobState",
    "Language",
    "TextSegment",
    "TTSBackend",
    "TTSBackendError",
    "Voice",
]
