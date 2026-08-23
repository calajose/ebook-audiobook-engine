"""Domain exceptions for audiobook-engine."""

from __future__ import annotations


class AudiobookEngineError(Exception):
    """Base exception for all engine errors."""


class EbookError(AudiobookEngineError):
    """Raised when ebook parsing fails."""


class TTSBackendError(AudiobookEngineError):
    """Raised when TTS synthesis fails."""


class AudioAssemblyError(AudiobookEngineError):
    """Raised when audio assembly (FFmpeg/M4B) fails."""


class ConfigurationError(AudiobookEngineError):
    """Raised for invalid or missing configuration."""


class JobError(AudiobookEngineError):
    """Raised for job-related errors."""
