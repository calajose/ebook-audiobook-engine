"""WAV silence generation and audio pause utilities.

Provides functions to create silent WAV audio and insert configurable
pauses between paragraphs, sections, and chapters.
"""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING

from audiobook_engine.domain.exceptions import AudioAssemblyError

if TYPE_CHECKING:
    from pathlib import Path


def generate_silence_wav(
    output_path: Path,
    duration_ms: int,
    sample_rate: int = 24000,
    channels: int = 1,
    sampwidth: int = 2,
) -> Path:
    """Generate a silent WAV file with specific audio parameters.

    Args:
        output_path: Target path for the generated WAV file.
        duration_ms: Duration of silence in milliseconds.
        sample_rate: Audio sample rate in Hz (default: 24000).
        channels: Number of audio channels (1 for mono, 2 for stereo).
        sampwidth: Sample width in bytes (2 for 16-bit PCM).

    Returns:
        The path to the generated WAV file.
    """
    if duration_ms < 0:
        raise AudioAssemblyError(
            f"Silence duration cannot be negative: {duration_ms} ms"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(sample_rate * (duration_ms / 1000.0))
    silence_bytes = b"\x00" * (num_frames * channels * sampwidth)

    try:
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(silence_bytes)
    except Exception as exc:
        raise AudioAssemblyError(
            f"Failed to generate silence WAV: {exc}"
        ) from exc

    return output_path


def get_wav_params(wav_path: Path) -> tuple[int, int, int]:
    """Extract (sample_rate, channels, sampwidth) from an existing WAV file."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
    except Exception as exc:
        raise AudioAssemblyError(
            f"Failed to read WAV parameters from {wav_path}: {exc}"
        ) from exc


def create_silence_matching(
    reference_wav: Path,
    duration_ms: int,
    output_path: Path,
) -> Path:
    """Generate silence matching the parameters of a reference WAV file."""
    sample_rate, channels, sampwidth = get_wav_params(reference_wav)
    return generate_silence_wav(
        output_path=output_path,
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        channels=channels,
        sampwidth=sampwidth,
    )


def append_silence_to_wav(
    input_wav: Path,
    duration_ms: int,
    output_path: Path | None = None,
) -> Path:
    """Append silence frames to the end of a WAV file.

    If output_path is None or identical to input_wav, modifies in place.
    """
    if duration_ms <= 0:
        if output_path is not None and output_path != input_wav:
            import shutil

            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(input_wav), str(output_path))
            return output_path
        return input_wav

    dest = output_path or input_wav
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with wave.open(str(input_wav), "rb") as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())

        num_silent_frames = int(params.framerate * (duration_ms / 1000.0))
        silence_bytes = (
            b"\x00" * (num_silent_frames * params.nchannels * params.sampwidth)
        )

        with wave.open(str(dest), "wb") as out:
            out.setparams(params)
            out.writeframes(frames)
            out.writeframes(silence_bytes)

        return dest
    except Exception as exc:
        raise AudioAssemblyError(
            f"Failed to append silence to {input_wav}: {exc}"
        ) from exc
