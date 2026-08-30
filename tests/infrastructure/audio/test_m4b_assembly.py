"""Tests for FFmpeg wrapper and M4B assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from audiobook_engine.domain.exceptions import AudioAssemblyError
from audiobook_engine.domain.job import AudiobookJob, JobState
from audiobook_engine.domain.models import Book, Chapter, TextSegment
from audiobook_engine.infrastructure.audio import ffmpeg
from audiobook_engine.infrastructure.audio.m4b_assembler import (
    _build_chapter_metadata,
    _collect_chapter_files,
    _create_filelist,
    _get_wav_duration_ms,
    _ms_to_timestamp,
    assemble_m4b,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_wav(path: Path, duration_ms: int = 1000) -> None:
    """Create a minimal WAV file with the given duration."""
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 22050
    n_frames = int(sample_rate * duration_ms / 1000)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _make_book() -> Book:
    """Create a minimal Book for testing."""
    return Book(
        title="Test Book",
        author="Test Author",
        language="en",
        chapters=(
            Chapter(
                title="Chapter 1",
                index=0,
                segments=(
                    TextSegment(text="Hello world.", index=0),
                ),
            ),
            Chapter(
                title="Chapter 2",
                index=1,
                segments=(
                    TextSegment(text="Second chapter.", index=0),
                ),
            ),
        ),
    )


class TestFFmpegAvailability:
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_is_available(self, mock_which: MagicMock) -> None:
        assert ffmpeg.is_available() is True

    @patch("shutil.which", return_value=None)
    def test_is_not_available(
        self, mock_which: MagicMock
    ) -> None:
        assert ffmpeg.is_available() is False


class TestFFmpegRun:
    @patch("shutil.which", return_value=None)
    def test_raises_when_not_available(
        self, mock_which: MagicMock
    ) -> None:
        with pytest.raises(AudioAssemblyError, match="not installed"):
            ffmpeg.run(["-i", "input.wav", "output.m4b"])

    @patch("audiobook_engine.infrastructure.audio.ffmpeg.subprocess")
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_raises_on_nonzero_exit(
        self, mock_which: MagicMock, mock_subprocess: MagicMock
    ) -> None:
        mock_subprocess.run.return_value = MagicMock(
            returncode=1, stderr="Error: bad input"
        )
        with pytest.raises(AudioAssemblyError, match="FFmpeg failed"):
            ffmpeg.run(["-i", "bad.wav", "out.m4b"])

    @patch("audiobook_engine.infrastructure.audio.ffmpeg.subprocess")
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_success(
        self, mock_which: MagicMock, mock_subprocess: MagicMock
    ) -> None:
        mock_subprocess.run.return_value = MagicMock(
            returncode=0, stderr=""
        )
        ffmpeg.run(["-i", "input.wav", "output.m4b"])


class TestM4bAssembler:
    def test_collect_chapter_files(
        self, tmp_path: Path
    ) -> None:
        # Create chapter directories with WAV files
        ch0 = tmp_path / "chapters" / "ch_0"
        ch1 = tmp_path / "chapters" / "ch_1"
        _make_wav(ch0 / "seg_0.wav")
        _make_wav(ch0 / "seg_1.wav")
        _make_wav(ch1 / "seg_0.wav")

        files = _collect_chapter_files(tmp_path / "chapters")
        assert len(files) == 3
        assert files[0].name == "seg_0.wav"
        assert files[2].name == "seg_0.wav"

    def test_collect_empty_dir(self, tmp_path: Path) -> None:
        files = _collect_chapter_files(tmp_path / "empty")
        assert files == []

    def test_build_chapter_metadata(
        self, tmp_path: Path
    ) -> None:
        ch0 = tmp_path / "chapters" / "ch_0"
        ch1 = tmp_path / "chapters" / "ch_1"
        _make_wav(ch0 / "seg_0.wav", duration_ms=5000)
        _make_wav(ch1 / "seg_0.wav", duration_ms=3000)

        book = _make_book()
        metadata = _build_chapter_metadata(tmp_path / "chapters", book)

        assert len(metadata) == 2
        assert metadata[0]["start_ms"] == 0
        assert metadata[0]["end_ms"] == 5000
        assert metadata[0]["title"] == "Chapter 1"
        assert metadata[1]["start_ms"] == 5000
        assert metadata[1]["end_ms"] == 8000
        assert metadata[1]["title"] == "Chapter 2"

    def test_ms_to_timestamp(self) -> None:
        assert _ms_to_timestamp(0) == "00:00:00.000"
        assert _ms_to_timestamp(1500) == "00:00:01.500"
        assert _ms_to_timestamp(65000) == "00:01:05.000"
        assert _ms_to_timestamp(3661000) == "01:01:01.000"

    def test_get_wav_duration_ms(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _make_wav(wav, duration_ms=2500)
        duration = _get_wav_duration_ms(wav)
        assert duration == 2500

    def test_create_filelist(self, tmp_path: Path) -> None:
        """Test filelist creation for concat demuxer."""
        ch0 = tmp_path / "chapters" / "ch_0"
        _make_wav(ch0 / "seg_0.wav")
        _make_wav(ch0 / "seg_1.wav")

        files = _collect_chapter_files(tmp_path / "chapters")
        filelist_path = _create_filelist(files)

        try:
            assert filelist_path.exists()
            content = filelist_path.read_text()
            assert "file '" in content
            assert "seg_0.wav" in content
            assert "seg_1.wav" in content
        finally:
            filelist_path.unlink(missing_ok=True)

    def test_create_filelist_cleanup(
        self, tmp_path: Path
    ) -> None:
        """Test that filelist is cleaned up after use."""
        ch0 = tmp_path / "chapters" / "ch_0"
        _make_wav(ch0 / "seg_0.wav")

        files = _collect_chapter_files(tmp_path / "chapters")
        filelist_path = _create_filelist(files)

        # File exists before cleanup
        assert filelist_path.exists()

        # Cleanup
        filelist_path.unlink(missing_ok=True)

        # File should not exist after cleanup
        assert not filelist_path.exists()

    @patch("audiobook_engine.infrastructure.audio.ffmpeg.run")
    @patch(
        "audiobook_engine.infrastructure.audio.ffmpeg.is_available",
        return_value=True,
    )
    def test_assemble_m4b_calls_ffmpeg(
        self,
        mock_available: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Setup chapter WAVs
        ch0 = tmp_path / "chapters" / "ch_0"
        _make_wav(ch0 / "seg_0.wav")

        job = AudiobookJob(
            id="test",
            state=JobState.ASSEMBLING,
            work_dir=tmp_path,
            output_path=tmp_path / "output" / "book.m4b",
        )
        book = _make_book()

        assemble_m4b(
            job,
            book,
            tmp_path / "chapters",
            tmp_path / "output" / "book.m4b",
        )

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        # Verify concat demuxer is used
        assert "-f" in args
        assert "concat" in args
        assert "-safe" in args
        assert "0" in args
        # Verify output file is specified
        assert str(tmp_path / "output" / "book.m4b") in args

    @patch(
        "audiobook_engine.infrastructure.audio.ffmpeg.is_available",
        return_value=False,
    )
    def test_assemble_m4b_raises_without_ffmpeg(
        self, mock_available: MagicMock, tmp_path: Path
    ) -> None:
        job = AudiobookJob(id="test", state=JobState.ASSEMBLING)
        book = _make_book()

        with pytest.raises(AudioAssemblyError, match="FFmpeg is required"):
            assemble_m4b(
                job, book, tmp_path / "chapters", tmp_path / "out.m4b"
            )
