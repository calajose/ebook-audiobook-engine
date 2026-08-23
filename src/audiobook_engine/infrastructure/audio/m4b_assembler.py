"""M4B assembler — builds M4B audiobooks from WAV segments using FFmpeg.

Supports chapter markers, metadata embedding (title, author, language),
and optional cover image embedding.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from audiobook_engine.domain.exceptions import AudioAssemblyError
from audiobook_engine.infrastructure.audio import ffmpeg

if TYPE_CHECKING:
    from pathlib import Path

    from audiobook_engine.domain.job import AudiobookJob
    from audiobook_engine.domain.models import Book

logger = logging.getLogger(__name__)


def assemble_m4b(
    job: AudiobookJob,
    book: Book,
    chapters_dir: Path,
    output_path: Path,
) -> None:
    """Assemble WAV segments into an M4B audiobook.

    Creates an M4B with:
    - Concatenated audio from all chapter WAV files
    - Chapter markers matching the book structure
    - Metadata tags (title, author, language)
    - Optional cover image from book metadata

    Raises AudioAssemblyError if FFmpeg is unavailable or fails.
    """
    if not ffmpeg.is_available():
        raise AudioAssemblyError(
            "FFmpeg is required for M4B assembly but was not found"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect chapter WAV files in order
    chapter_files = _collect_chapter_files(chapters_dir)
    if not chapter_files:
        raise AudioAssemblyError(
            f"No WAV files found in {chapters_dir}"
        )

    # Create chapter metadata file
    chapter_metadata = _build_chapter_metadata(chapters_dir, book)

    # Build FFmpeg command
    args = _build_ffmpeg_args(
        chapter_files,
        chapter_metadata,
        book,
        output_path,
    )

    ffmpeg.run(args, timeout=1800)


def _extract_chapter_index_from_dir(dir_name: str) -> int:
    """Extract chapter index from directory name like 'ch_0'."""
    try:
        return int(dir_name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _extract_segment_index_from_file(stem: str) -> int:
    """Extract segment index from filename like 'seg_10'."""
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _collect_chapter_files(chapters_dir: Path) -> list[Path]:
    """Collect all chapter WAV files sorted by chapter index."""
    files: list[Path] = []
    if not chapters_dir.exists():
        return files

    for ch_dir in sorted(
        chapters_dir.iterdir(),
        key=lambda p: _extract_chapter_index_from_dir(p.name)
    ):
        if not ch_dir.is_dir():
            continue
        for wav in sorted(
            ch_dir.glob("seg_*.wav"),
            key=lambda p: _extract_segment_index_from_file(p.stem)
        ):
            files.append(wav)
    return files


def _build_chapter_metadata(
    chapters_dir: Path,
    book: Book,
) -> list[dict[str, object]]:
    """Build chapter metadata with timestamps for M4B chapter markers."""
    if not chapters_dir.exists():
        return []

    metadata: list[dict[str, object]] = []
    current_time_ms = 0

    for ch_dir in sorted(
        chapters_dir.iterdir(),
        key=lambda p: _extract_chapter_index_from_dir(p.name)
    ):
        if not ch_dir.is_dir():
            continue

        ch_index = _extract_chapter_index(ch_dir.name)
        ch_title = _get_chapter_title(book, ch_index)

        # Calculate duration of all segments in this chapter
        chapter_duration_ms = 0
        for seg_wav in sorted(
            ch_dir.glob("seg_*.wav"),
            key=lambda p: _extract_segment_index_from_file(p.stem)
        ):
            duration = _get_wav_duration_ms(seg_wav)
            chapter_duration_ms += duration

        if chapter_duration_ms > 0:
            metadata.append(
                {
                    "start_ms": current_time_ms,
                    "end_ms": current_time_ms + chapter_duration_ms,
                    "title": ch_title,
                }
            )
            current_time_ms += chapter_duration_ms

    return metadata


def _extract_chapter_index(dir_name: str) -> int:
    """Extract chapter index from directory name like 'ch_0' or 'ch_0000'."""
    try:
        # Handle both 'ch_0' and 'ch_0000' formats
        parts = dir_name.split("_", 1)
        if len(parts) == 2:
            return int(parts[1])
    except (IndexError, ValueError):
        pass
    return 0


def _get_chapter_title(book: Book, index: int) -> str:
    """Get chapter title from book metadata, or generate a default."""
    for ch in book.chapters:
        if ch.index == index:
            return ch.title
    return f"Chapter {index + 1}"


def _get_wav_duration_ms(wav_path: Path) -> int:
    """Get duration of a WAV file in milliseconds."""
    import wave

    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return 0
            return int((frames / rate) * 1000)
    except Exception:
        return 0


def _build_ffmpeg_args(
    chapter_files: list[Path],
    chapter_metadata: list[dict[str, object]],
    book: Book,
    output_path: Path,
) -> list[str]:
    """Build the complete FFmpeg argument list for M4B assembly."""
    args: list[str] = ["-y", "-nostdin"]

    # Input files — one per chapter WAV
    for wav in chapter_files:
        args.extend(["-i", str(wav)])

    # Build filter complex for concatenation
    n = len(chapter_files)
    if n == 1:
        filter_complex = "[0:a]acopy[out]"
    else:
        inputs = "".join(f"[{i}:a]" for i in range(n))
        filter_complex = f"{inputs}concat=n={n}:v=0:a=1[out]"

    args.extend(["-filter_complex", filter_complex])
    args.extend(["-map", "[out]"])

    # Output format and codec
    args.extend([
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-ac", "1",
        "-f", "ipod",
    ])

    # Metadata tags
    args.extend(["-metadata", f"title={book.title}"])
    args.extend(["-metadata", f"artist={book.author}"])
    args.extend(["-metadata", f"album={book.title}"])
    args.extend(["-metadata", f"language={book.language}"])
    args.extend(["-metadata", "genre=Audiobook"])

    # Chapter markers via metadata file
    if chapter_metadata:
        chapter_args = _build_chapter_ffmpeg_args(chapter_metadata)
        args.extend(chapter_args)

    # Cover image if available
    if book.cover_path is not None and book.cover_path.exists():
        args.extend([
            "-i", str(book.cover_path),
            "-map", f"{n}:v",
            "-c:v", "mjpeg",
            "-disposition:v:0", "attached_pic",
        ])

    args.append(str(output_path))
    return args


def _build_chapter_ffmpeg_args(
    chapter_metadata: list[dict[str, object]],
) -> list[str]:
    """Build FFmpeg chapter metadata arguments."""
    # Format: metadata:s:v chapter_key=value pairs
    args: list[str] = []
    for ch in chapter_metadata:
        start_ms = int(str(ch["start_ms"]))
        end_ms = int(str(ch["end_ms"]))
        title = str(ch["title"])

        # FFmpeg uses time in format HH:MM:SS.mmm
        start_time = _ms_to_timestamp(start_ms)
        end_time = _ms_to_timestamp(end_ms)

        args.extend([
            "-metadata:s:a",
            f"chapter_start={start_time}",
            "-metadata:s:a",
            f"chapter_end={end_time}",
            "-metadata:s:a",
            f"chapter_title={title}",
        ])

    return args


def _ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS.mmm format."""
    total_seconds = ms / 1000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
