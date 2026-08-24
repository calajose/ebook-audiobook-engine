"""Tests for WAV file merger."""

from __future__ import annotations

import struct
import wave
from typing import TYPE_CHECKING

import pytest

from audiobook_engine.domain.exceptions import AudioAssemblyError
from audiobook_engine.infrastructure.audio.wav_merger import merge_wav_files

if TYPE_CHECKING:
    from pathlib import Path


def _make_wav(path: Path, duration_frames: int = 100) -> None:
    """Write a minimal valid WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        frames = struct.pack(
            f"<{duration_frames}h", *([0] * duration_frames)
        )
        w.writeframes(frames)


class TestMergeWavFiles:
    def test_merge_two_files(self, tmp_path: Path) -> None:
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        out = tmp_path / "merged.wav"
        _make_wav(a, 100)
        _make_wav(b, 200)

        merge_wav_files([a, b], out)

        assert out.exists()
        with wave.open(str(out), "rb") as w:
            assert w.getnframes() == 300

    def test_merge_single_file(self, tmp_path: Path) -> None:
        a = tmp_path / "a.wav"
        out = tmp_path / "merged.wav"
        _make_wav(a, 50)

        merge_wav_files([a], out)

        with wave.open(str(out), "rb") as w:
            assert w.getnframes() == 50

    def test_merge_empty_list_raises(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.wav"
        with pytest.raises(AudioAssemblyError, match="No WAV files"):
            merge_wav_files([], out)

    def test_preserves_parameters(
        self, tmp_path: Path
    ) -> None:
        a = tmp_path / "a.wav"
        out = tmp_path / "merged.wav"
        _make_wav(a, 10)

        merge_wav_files([a], out)

        with wave.open(str(out), "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() == 24000

    def test_merge_with_silence_between(self, tmp_path: Path) -> None:
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        out = tmp_path / "merged.wav"
        _make_wav(a, 240)  # 10ms at 24000Hz
        _make_wav(b, 240)  # 10ms at 24000Hz

        # Insert 100ms silence (2400 frames) between a and b
        merge_wav_files([a, b], out, silence_between_ms=100)

        with wave.open(str(out), "rb") as w:
            assert w.getnframes() == 240 + 2400 + 240

