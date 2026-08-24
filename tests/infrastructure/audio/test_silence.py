"""Tests for silence generation and audio pause utilities."""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING

import pytest

from audiobook_engine.domain.exceptions import AudioAssemblyError
from audiobook_engine.infrastructure.audio.silence import (
    append_silence_to_wav,
    create_silence_matching,
    generate_silence_wav,
    get_wav_params,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_sample_wav(
    path: Path,
    sample_rate: int = 24000,
    channels: int = 1,
    sampwidth: int = 2,
    num_frames: int = 2400,  # 100 ms
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x01\x00" * (num_frames * channels))


class TestGenerateSilenceWav:
    def test_generate_silence(self, tmp_path: Path) -> None:
        out = tmp_path / "silence.wav"
        generate_silence_wav(out, duration_ms=500, sample_rate=24000)

        assert out.exists()
        with wave.open(str(out), "rb") as wf:
            assert wf.getframerate() == 24000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 12000  # 24000 * 0.5

    def test_negative_duration_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "silence.wav"
        with pytest.raises(AudioAssemblyError, match="cannot be negative"):
            generate_silence_wav(out, duration_ms=-100)


class TestGetWavParams:
    def test_get_params(self, tmp_path: Path) -> None:
        wav = tmp_path / "sample.wav"
        _make_sample_wav(wav, sample_rate=22050, channels=2, sampwidth=2)
        rate, channels, width = get_wav_params(wav)
        assert rate == 22050
        assert channels == 2
        assert width == 2


class TestCreateSilenceMatching:
    def test_matching_silence(self, tmp_path: Path) -> None:
        ref = tmp_path / "ref.wav"
        out = tmp_path / "silence.wav"
        _make_sample_wav(ref, sample_rate=16000, channels=1, sampwidth=2)

        create_silence_matching(ref, duration_ms=250, output_path=out)

        with wave.open(str(out), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 4000  # 16000 * 0.25


class TestAppendSilenceToWav:
    def test_append_silence_creates_new_file(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.wav"
        out = tmp_path / "output.wav"
        _make_sample_wav(inp, sample_rate=24000, num_frames=2400)  # 100ms

        append_silence_to_wav(inp, duration_ms=200, output_path=out)

        with wave.open(str(out), "rb") as wf:
            # 2400 frames (100ms) + 4800 frames (200ms) = 7200 frames (300ms)
            assert wf.getnframes() == 7200

    def test_append_zero_duration(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.wav"
        out = tmp_path / "output.wav"
        _make_sample_wav(inp, sample_rate=24000, num_frames=2400)

        append_silence_to_wav(inp, duration_ms=0, output_path=out)

        with wave.open(str(out), "rb") as wf:
            assert wf.getnframes() == 2400
