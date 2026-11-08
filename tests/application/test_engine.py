"""Tests for AudiobookEngine application services."""

from __future__ import annotations

import struct
import wave
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    name: str = "fake"

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
    def test_returns_capabilities(self, engine: AudiobookEngine) -> None:
        caps = engine.capabilities()
        assert len(caps.languages) == 1
        assert caps.languages[0].code == "en-us"
        assert len(caps.voices) == 1
        assert caps.voices[0].id == "af_sarah"


class TestCreateJob:
    def test_creates_job(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        job = engine.create_job(sample_epub, "en-us", "af_sarah", tmp_path / "out.m4b")
        assert job.state == JobState.CREATED
        assert job.book_title == "Test Book"
        assert job.language == "en-us"
        assert job.voice == "af_sarah"
        assert job.backend == "fake"
        assert job.source_path == sample_epub

    def test_get_job(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        job = engine.create_job(sample_epub, "en-us", "af_sarah", tmp_path / "out.m4b")
        found = engine.get_job(job.id)
        assert found.id == job.id

    def test_get_job_not_found(self, engine: AudiobookEngine) -> None:
        with pytest.raises(JobError, match="not found"):
            engine.get_job("nonexistent")


class TestRun:
    def test_full_pipeline(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        job = engine.create_job(sample_epub, "en-us", "af_sarah", tmp_path / "out.m4b")
        result = engine.run(job.id)
        assert result.state == JobState.COMPLETED
        assert result.total_segments > 0
        assert result.completed_segments == result.total_segments

    def test_creates_output_file(
        self, engine: AudiobookEngine, sample_epub: Path
    ) -> None:
        output = engine._work_dir.parent / "final_output.wav"
        job = engine.create_job(sample_epub, "en-us", "af_sarah", output)
        engine.run(job.id)
        assert output.exists()

    def test_creates_work_directory(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        job = engine.create_job(
            sample_epub,
            "en-us",
            "af_sarah",
            tmp_path / "out.wav",
            keep_intermediates=True,
        )
        engine.run(job.id)
        work_dir = engine._work_dir / job.id
        assert work_dir.exists()
        assert (work_dir / "segments").exists()
        assert (work_dir / "chapters").exists()
        assert (work_dir / "output").exists()

    def test_cleanup_intermediates_by_default(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        job = engine.create_job(sample_epub, "en-us", "af_sarah", tmp_path / "out.wav")
        engine.run(job.id)
        work_dir = engine._work_dir / job.id
        assert work_dir.exists()
        assert not (work_dir / "segments").exists()
        assert not (work_dir / "chapters").exists()
        assert not (work_dir / "output").exists()


class TestCancel:
    def test_cancel_created_job(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        job = engine.create_job(sample_epub, "en-us", "af_sarah", tmp_path / "out.wav")
        result = engine.cancel(job.id)
        assert result.state == JobState.CANCELLED

    def test_cancel_completed_job_raises(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        job = engine.create_job(sample_epub, "en-us", "af_sarah", tmp_path / "out.wav")
        engine.run(job.id)
        with pytest.raises(JobError, match="Cannot cancel"):
            engine.cancel(job.id)


class TestResume:
    def test_resume_not_resumable_raises(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        job = engine.create_job(sample_epub, "en-us", "af_sarah", tmp_path / "out.wav")
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        job.transition(JobState.ASSEMBLING)
        job.transition(JobState.COMPLETED)
        with pytest.raises(JobError, match="cannot be resumed"):
            engine.resume(job.id)

    def test_resume_from_created(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        job = engine.create_job(sample_epub, "en-us", "af_sarah", tmp_path / "out.wav")
        assert job.state == JobState.CREATED
        assert job.can_resume()
        result = engine.resume(job.id)
        assert result.state == JobState.COMPLETED

    def test_resume_from_ready(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        job = engine.create_job(sample_epub, "en-us", "af_sarah", tmp_path / "out.wav")
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        assert job.can_resume()
        result = engine.resume(job.id)
        assert result.state == JobState.COMPLETED

    def test_resume_from_synthesizing(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        job = engine.create_job(sample_epub, "en-us", "af_sarah", tmp_path / "out.wav")
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        assert job.can_resume()
        result = engine.resume(job.id)
        assert result.state == JobState.COMPLETED

    def test_resume_from_assembling(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        """Test that resume retries assembly when state is ASSEMBLING."""
        job = engine.create_job(sample_epub, "en-us", "af_sarah", tmp_path / "out.wav")
        job.transition(JobState.ANALYZING)
        job.transition(JobState.READY)
        job.transition(JobState.SYNTHESIZING)
        job.transition(JobState.ASSEMBLING)
        assert job.can_resume()
        result = engine.resume(job.id)
        assert result.state == JobState.COMPLETED


class _FailingTTSBackend(FakeTTSBackend):
    def synthesize(
        self,
        text: str,
        language: str,
        voice: str,
        output_path: Path,
        speed: float = 1.0,
    ) -> None:
        from audiobook_engine.domain.exceptions import TTSBackendError

        raise TTSBackendError("boom")


class _RecordingTTSBackend(FakeTTSBackend):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def synthesize(
        self,
        text: str,
        language: str,
        voice: str,
        output_path: Path,
        speed: float = 1.0,
    ) -> None:
        self.texts.append(text)
        _make_fake_wav(output_path)


class _FixedBookParser:
    def __init__(self, book: object) -> None:
        self._book = book

    def inspect(self, path: Path) -> object:
        return self._book


def _fixed_book(title: str, segment_texts: list[str]) -> object:
    from audiobook_engine.domain.models import Book, Chapter, TextSegment

    segments = tuple(
        TextSegment(text=text, index=index) for index, text in enumerate(segment_texts)
    )
    return Book(
        title="Libro",
        author="Autor",
        language="en",
        chapters=(Chapter(title=title, index=0, segments=segments),),
    )


def _wav_duration_ms(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate() * 1000.0


class TestFailureHandling:
    def test_non_recoverable_error_marks_job_failed(self, tmp_path: Path) -> None:
        from audiobook_engine.domain.exceptions import TTSBackendError
        from audiobook_engine.infrastructure.persistence.job_store import (
            load_job,
        )

        engine = AudiobookEngine(
            parser=EPUBParser(),
            tts_backend=_FailingTTSBackend(),
            work_dir=tmp_path / "work",
        )
        epub = tmp_path / "book.epub"
        build_epub(epub, chapters=[("Ch1", "Some text.")])
        job = engine.create_job(epub, "en-us", "af_sarah", tmp_path / "out.wav")

        with pytest.raises(JobError, match="Conversion failed"):
            engine.run(job.id)

        assert job.state == JobState.FAILED
        assert job.error_message
        assert "boom" in job.error_message
        assert job.work_dir is not None
        persisted = load_job(job.work_dir)
        assert persisted.state == JobState.FAILED
        assert persisted.error_message
        assert not (tmp_path / "out.wav").exists()
        assert isinstance(TTSBackendError("x"), Exception)

    def test_m4b_assembly_error_fails_without_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from audiobook_engine.domain.exceptions import AudioAssemblyError

        def _failing_assemble(**kwargs: object) -> None:
            raise AudioAssemblyError("ffmpeg exploded")

        monkeypatch.setattr(
            "audiobook_engine.infrastructure.audio.ffmpeg.is_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "audiobook_engine.application.engine.assemble_m4b",
            _failing_assemble,
        )

        engine = AudiobookEngine(
            parser=EPUBParser(),
            tts_backend=FakeTTSBackend(),
            work_dir=tmp_path / "work",
        )
        epub = tmp_path / "book.epub"
        build_epub(epub, chapters=[("Ch1", "Some text.")])
        output = tmp_path / "out.m4b"
        job = engine.create_job(epub, "en-us", "af_sarah", output)

        with pytest.raises(JobError, match="Conversion failed"):
            engine.run(job.id)

        assert job.state == JobState.FAILED
        assert not output.exists()

    def test_m4b_without_ffmpeg_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "audiobook_engine.infrastructure.audio.ffmpeg.is_available",
            lambda: False,
        )
        engine = AudiobookEngine(
            parser=EPUBParser(),
            tts_backend=FakeTTSBackend(),
            work_dir=tmp_path / "work",
        )
        epub = tmp_path / "book.epub"
        build_epub(epub, chapters=[("Ch1", "Some text.")])
        output = tmp_path / "out.m4b"
        job = engine.create_job(epub, "en-us", "af_sarah", output)

        with pytest.raises(JobError):
            engine.run(job.id)

        assert job.state == JobState.FAILED
        assert not output.exists()

    def test_unsupported_output_extension_fails(self, tmp_path: Path) -> None:
        engine = AudiobookEngine(
            parser=EPUBParser(),
            tts_backend=FakeTTSBackend(),
            work_dir=tmp_path / "work",
        )
        epub = tmp_path / "book.epub"
        build_epub(epub, chapters=[("Ch1", "Some text.")])
        job = engine.create_job(epub, "en-us", "af_sarah", tmp_path / "out.mp3")

        with pytest.raises(JobError):
            engine.run(job.id)

        assert job.state == JobState.FAILED


class TestProsodyAnchors:
    def test_scene_break_pause_is_applied(
        self, engine: AudiobookEngine, tmp_path: Path
    ) -> None:
        epub = tmp_path / "scene.epub"
        build_epub(
            epub,
            chapters=[("Chapter One", "Antes.\n***\nDespués.")],
        )
        job = engine.create_job(
            epub,
            "en-us",
            "af_sarah",
            tmp_path / "out.wav",
            keep_intermediates=True,
        )
        engine.run(job.id)

        marker_wav = (
            tmp_path / "work" / job.id / "chapters" / "ch_0000" / "seg_000001.wav"
        )
        assert marker_wav.exists()
        assert abs(_wav_duration_ms(marker_wav) - 1500) < 20

    def test_consecutive_separators_collapse_to_one_pause(
        self, engine: AudiobookEngine, tmp_path: Path
    ) -> None:
        epub = tmp_path / "scene2.epub"
        build_epub(
            epub,
            chapters=[("Chapter One", "Antes.\n***\n---\nDespués.")],
        )
        job = engine.create_job(
            epub,
            "en-us",
            "af_sarah",
            tmp_path / "out.wav",
            keep_intermediates=True,
        )
        engine.run(job.id)

        ch_dir = tmp_path / "work" / job.id / "chapters" / "ch_0000"
        assert (ch_dir / "seg_000001.wav").exists()
        assert not (ch_dir / "seg_000002.wav").exists()

    def test_marker_only_segment_has_no_paragraph_pause(
        self, engine: AudiobookEngine, tmp_path: Path
    ) -> None:
        epub = tmp_path / "scene3.epub"
        build_epub(
            epub,
            chapters=[("Chapter One", "Antes.\n***\nDespués.")],
        )
        job = engine.create_job(
            epub,
            "en-us",
            "af_sarah",
            tmp_path / "out.wav",
            keep_intermediates=True,
            paragraph_pause_ms=700,
        )
        engine.run(job.id)

        marker_wav = (
            tmp_path / "work" / job.id / "chapters" / "ch_0000" / "seg_000001.wav"
        )
        # Exactly the scene pause, not scene + paragraph pause.
        assert abs(_wav_duration_ms(marker_wav) - 1500) < 20

    def test_chapter_title_anchor_helper(self) -> None:
        from audiobook_engine.infrastructure.ebook.normalizer import (
            CHAPTER_TITLE_MARKER,
        )

        anchored = AudiobookEngine._apply_chapter_title_anchor(
            "Capítulo Uno\nEl bosque.", "Capítulo Uno", True
        )
        assert CHAPTER_TITLE_MARKER in anchored.splitlines()

        not_first = AudiobookEngine._apply_chapter_title_anchor(
            "Capítulo Uno\nEl bosque.", "Capítulo Uno", False
        )
        assert CHAPTER_TITLE_MARKER not in not_first

        not_title = AudiobookEngine._apply_chapter_title_anchor(
            "Otra cosa.", "Capítulo Uno", True
        )
        assert CHAPTER_TITLE_MARKER not in not_title

    def test_chapter_title_pause_applied_after_narrated_title(
        self, tmp_path: Path
    ) -> None:
        from audiobook_engine.domain.models import Book, Chapter, TextSegment

        book = Book(
            title="Libro",
            author="Autor",
            language="en",
            chapters=(
                Chapter(
                    title="Capítulo Uno",
                    index=0,
                    segments=(
                        TextSegment(text="Capítulo Uno", index=0),
                        TextSegment(text="El bosque.", index=1),
                    ),
                ),
            ),
        )
        engine = AudiobookEngine(
            parser=_FixedBookParser(book),  # type: ignore[arg-type]
            tts_backend=FakeTTSBackend(),
            work_dir=tmp_path / "work",
        )
        job = engine.create_job(
            tmp_path / "book.txt",
            "en-us",
            "af_sarah",
            tmp_path / "out.wav",
            keep_intermediates=True,
        )
        engine.run(job.id)

        pause_wav = (
            tmp_path / "work" / job.id / "segments" / "ch0000_seg000000_chunk1.wav"
        )
        assert pause_wav.exists()
        assert abs(_wav_duration_ms(pause_wav) - 1200) < 20

    def test_pause_events_coalesce_and_classify(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        from audiobook_engine.domain.job import AudiobookJob
        from audiobook_engine.infrastructure.ebook.normalizer import (
            CHAPTER_TITLE_MARKER,
            SCENE_BREAK_MARKER,
        )

        job = AudiobookJob(scene_break_pause_ms=1500, chapter_title_pause_ms=1200)
        events = AudiobookEngine._build_segment_events(
            [SCENE_BREAK_MARKER, SCENE_BREAK_MARKER], job
        )
        assert len(events) == 1
        assert events[0].text is None
        assert events[0].pause_ms == 1500

        events = AudiobookEngine._build_segment_events(
            [SCENE_BREAK_MARKER, CHAPTER_TITLE_MARKER], job
        )
        assert len(events) == 1
        assert events[0].pause_ms == 1500

        events = AudiobookEngine._build_segment_events(
            ["Texto.", SCENE_BREAK_MARKER, "Más texto."], job
        )
        assert [event.text for event in events] == ["Texto.", None, "Más texto."]


class TestMaxChars:
    def test_create_job_persists_max_chars(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        from audiobook_engine.infrastructure.persistence.job_store import (
            load_job,
        )

        job = engine.create_job(
            sample_epub,
            "en-us",
            "af_sarah",
            tmp_path / "out.wav",
            max_chars=200,
        )
        assert job.max_chars == 200
        assert job.work_dir is not None
        assert load_job(job.work_dir).max_chars == 200

    def test_synthesis_honours_persisted_max_chars(self, tmp_path: Path) -> None:
        tts = _RecordingTTSBackend()
        book = _fixed_book("Cap", ["Uno. Dos. Tres. Cuatro."])
        engine = AudiobookEngine(
            parser=_FixedBookParser(book),  # type: ignore[arg-type]
            tts_backend=tts,
            work_dir=tmp_path / "work",
        )
        job = engine.create_job(
            tmp_path / "book.txt",
            "en-us",
            "af_sarah",
            tmp_path / "out.wav",
            max_chars=8,
        )
        engine.run(job.id)

        assert tts.texts
        assert all(len(text) <= 8 for text in tts.texts)

    def test_total_segments_matches_analyze(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        analysis = engine.analyze(sample_epub, max_chars=200)
        expected = sum(chapter.chunks for chapter in analysis.chapter_analyses)
        job = engine.create_job(
            sample_epub,
            "en-us",
            "af_sarah",
            tmp_path / "out.wav",
            max_chars=200,
        )
        engine.run(job.id)
        assert job.total_segments == expected

    def test_create_job_rejects_invalid_max_chars(
        self, engine: AudiobookEngine, sample_epub: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(JobError, match="max_chars"):
            engine.create_job(
                sample_epub,
                "en-us",
                "af_sarah",
                tmp_path / "out.wav",
                max_chars=0,
            )


class TestProtectedProsody:
    def test_chapter_pause_replaces_paragraph_pause(
        self, engine: AudiobookEngine, tmp_path: Path
    ) -> None:
        """Protected rule A-45: chapter pause replaces paragraph pause."""
        epub = tmp_path / "prosody.epub"
        build_epub(epub, chapters=[("Chapter One", "Uno.\nDos.")])
        job = engine.create_job(
            epub,
            "en-us",
            "af_sarah",
            tmp_path / "out.wav",
            keep_intermediates=True,
            paragraph_pause_ms=700,
            chapter_pause_ms=2500,
        )
        engine.run(job.id)

        ch_dir = tmp_path / "work" / job.id / "chapters" / "ch_0000"
        first_ms = _wav_duration_ms(ch_dir / "seg_000000.wav")
        last_ms = _wav_duration_ms(ch_dir / "seg_000001.wav")

        assert abs(first_ms - 700) < 20
        assert abs(last_ms - 2500) < 20
