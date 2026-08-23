"""Tests for domain data models."""

from __future__ import annotations

from pathlib import Path

import pytest

from audiobook_engine.domain.models import (
    BackendCapabilities,
    Book,
    Chapter,
    Language,
    TextSegment,
    Voice,
)


class TestTextSegment:
    def test_creation(self) -> None:
        seg = TextSegment(text="Hello world", index=0)
        assert seg.text == "Hello world"
        assert seg.index == 0

    def test_frozen(self) -> None:
        seg = TextSegment(text="x", index=0)
        with pytest.raises(AttributeError):
            seg.text = "y"  # type: ignore[misc]


class TestChapter:
    def test_creation_with_segments(self) -> None:
        segs = (TextSegment(text="a", index=0), TextSegment(text="b", index=1))
        ch = Chapter(title="Ch1", index=0, segments=segs)
        assert ch.title == "Ch1"
        assert len(ch.segments) == 2

    def test_empty_segments_default(self) -> None:
        ch = Chapter(title="Ch", index=0)
        assert ch.segments == ()


class TestBook:
    def test_creation(self) -> None:
        book = Book(title="T", author="A", language="en")
        assert book.title == "T"
        assert book.author == "A"
        assert book.language == "en"
        assert book.chapters == ()
        assert book.cover_path is None
        assert book.source_path is None

    def test_with_chapters(self) -> None:
        ch = Chapter(title="C", index=0)
        book = Book(title="T", author="A", language="en", chapters=(ch,))
        assert len(book.chapters) == 1

    def test_with_cover(self) -> None:
        book = Book(
            title="T",
            author="A",
            language="en",
            cover_path=Path("/tmp/cover.jpg"),
        )
        assert book.cover_path == Path("/tmp/cover.jpg")


class TestLanguage:
    def test_creation(self) -> None:
        lang = Language(code="es", name="Spanish")
        assert lang.code == "es"
        assert lang.name == "Spanish"


class TestVoice:
    def test_creation(self) -> None:
        voice = Voice(id="v1", name="Voice 1", language_code="es")
        assert voice.id == "v1"
        assert voice.language_code == "es"


class TestBackendCapabilities:
    def test_empty(self) -> None:
        caps = BackendCapabilities()
        assert caps.languages == ()
        assert caps.voices == ()

    def test_with_data(self) -> None:
        langs = (Language(code="en", name="English"),)
        voices = (Voice(id="v1", name="V1", language_code="en"),)
        caps = BackendCapabilities(languages=langs, voices=voices)
        assert len(caps.languages) == 1
        assert len(caps.voices) == 1
