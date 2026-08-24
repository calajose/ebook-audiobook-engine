"""WAV audio merger — concatenates multiple WAV files into one."""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING

from audiobook_engine.domain.exceptions import AudioAssemblyError

if TYPE_CHECKING:
    from pathlib import Path


def merge_wav_files(
    input_paths: list[Path],
    output_path: Path,
    silence_between_ms: int = 0,
) -> None:
    """Concatenate multiple WAV files into a single WAV file.

    All input files must have the same sample rate, channels, and
    sample width. Uses raw PCM concatenation — no re-encoding.
    If silence_between_ms > 0, inserts silent PCM frames between files.
    """
    if not input_paths:
        raise AudioAssemblyError("No WAV files to merge")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with wave.open(str(input_paths[0]), "rb") as first:
            params = first.getparams()
            frames = first.readframes(first.getnframes())

        silence_bytes = b""
        if silence_between_ms > 0:
            num_silent_frames = int(
                params.framerate * (silence_between_ms / 1000.0)
            )
            silence_bytes = (
                b"\x00" * (num_silent_frames * params.nchannels * params.sampwidth)
            )

        with wave.open(str(output_path), "wb") as out:
            out.setparams(params)
            out.writeframes(frames)

            for path in input_paths[1:]:
                if silence_bytes:
                    out.writeframes(silence_bytes)
                with wave.open(str(path), "rb") as w:
                    # Compare channels, sample width, frame rate
                    # (not nframes — that's what we're combining)
                    if w.getparams()[:3] != params[:3]:
                        raise AudioAssemblyError(
                            f"WAV format mismatch: {path} "
                            f"has different parameters"
                        )
                    out.writeframes(w.readframes(w.getnframes()))

    except AudioAssemblyError:
        raise
    except Exception as exc:
        raise AudioAssemblyError(
            f"Failed to merge WAV files: {exc}"
        ) from exc
