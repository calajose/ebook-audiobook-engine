"""FFmpeg wrapper — availability check, command execution, error handling."""

from __future__ import annotations

import logging
import shutil
import subprocess

from audiobook_engine.domain.exceptions import AudioAssemblyError

logger = logging.getLogger(__name__)

_FFMPEG_BIN = "ffmpeg"


def is_available() -> bool:
    """Return True if FFmpeg is found on PATH."""
    return shutil.which(_FFMPEG_BIN) is not None


def get_version() -> str | None:
    """Return the FFmpeg version string, or None if unavailable."""
    if not is_available():
        return None
    try:
        result = subprocess.run(
            [_FFMPEG_BIN, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = result.stdout.split("\n", 1)[0]
        return first_line.strip()
    except Exception:
        return None


def run(args: list[str], *, timeout: int = 600) -> None:
    """Run an FFmpeg command, raising AudioAssemblyError on failure.

    Parameters
    ----------
    args:
        Arguments to pass to FFmpeg (without the binary name).
    timeout:
        Maximum seconds to wait for completion.
    """
    if not is_available():
        raise AudioAssemblyError("FFmpeg is not installed or not on PATH")

    cmd = [_FFMPEG_BIN, *args]
    logger.debug("Running FFmpeg: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioAssemblyError(
            f"FFmpeg timed out after {timeout}s"
        ) from exc
    except FileNotFoundError as exc:
        raise AudioAssemblyError(
            "FFmpeg executable not found"
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()[-500:] if result.stderr else ""
        raise AudioAssemblyError(
            f"FFmpeg failed (exit {result.returncode}): {stderr}"
        )

    logger.debug("FFmpeg completed successfully")
