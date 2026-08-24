"""Domain data models for audiobook conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class TextSegment:
    """A single chunk of text ready for TTS synthesis."""

    text: str
    index: int


@dataclass(frozen=True)
class Chapter:
    """A book chapter containing ordered text segments."""

    title: str
    index: int
    segments: tuple[TextSegment, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Book:
    """Metadata and content extracted from an ebook file."""

    title: str
    author: str
    language: str
    chapters: tuple[Chapter, ...] = field(default_factory=tuple)
    cover_path: Path | None = None
    source_path: Path | None = None


@dataclass(frozen=True)
class Language:
    """A language supported by a TTS backend."""

    code: str
    name: str


@dataclass(frozen=True)
class Voice:
    """A voice available in a TTS backend."""

    id: str
    name: str
    language_code: str


@dataclass(frozen=True)
class BackendCapabilities:
    """Languages and voices a TTS backend supports."""

    languages: tuple[Language, ...] = field(default_factory=tuple)
    voices: tuple[Voice, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProsodyConfig:
    """Prosody and pause durations configuration for audio synthesis."""

    paragraph_pause_ms: int = 700
    chapter_pause_ms: int = 2500
    scene_break_pause_ms: int = 1500
    chapter_title_pause_ms: int = 1200
    speed: float = 1.0

