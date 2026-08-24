"""Protocols defining contracts for ebook parsers and TTS backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from audiobook_engine.domain.models import BackendCapabilities, Book


@runtime_checkable
class EbookParser(Protocol):
    """Contract for ebook format parsers."""

    def inspect(self, path: Path) -> Book:
        """Parse an ebook and return its structured representation."""
        ...


@runtime_checkable
class TTSBackend(Protocol):
    """Contract for text-to-speech backends."""

    def capabilities(self) -> BackendCapabilities:
        """Return the languages and voices this backend supports."""
        ...

    def synthesize(
        self,
        text: str,
        language: str,
        voice: str,
        output_path: Path,
        speed: float = 1.0,
    ) -> None:
        """Synthesize text to a WAV file at the given path."""
        ...
