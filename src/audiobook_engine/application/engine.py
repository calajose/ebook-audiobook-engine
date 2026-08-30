"""Application services — orchestrates the audiobook conversion pipeline.

CLI, Python API, and HTTP all invoke this layer. It coordinates
the ebook parser, normalizer, chunker, TTS backend, and audio assembly.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from audiobook_engine.domain.exceptions import JobError
from audiobook_engine.domain.job import AudiobookJob, JobState
from audiobook_engine.infrastructure.audio import ffmpeg
from audiobook_engine.infrastructure.audio.m4b_assembler import (
    _extract_chapter_index_from_dir,
    _extract_segment_index_from_file,
    assemble_m4b,
)
from audiobook_engine.infrastructure.audio.silence import append_silence_to_wav
from audiobook_engine.infrastructure.audio.wav_merger import merge_wav_files
from audiobook_engine.infrastructure.ebook.chunker import chunk_text
from audiobook_engine.infrastructure.ebook.normalizer import normalize
from audiobook_engine.infrastructure.persistence.job_store import (
    load_job,
    save_job,
)

if TYPE_CHECKING:
    from audiobook_engine.domain.models import BackendCapabilities, Book
    from audiobook_engine.domain.protocols import EbookParser, TTSBackend

logger = logging.getLogger(__name__)


class AudiobookEngine:
    """Core engine that orchestrates audiobook conversion.

    All three interfaces (CLI, Python API, HTTP) delegate to this class.
    """

    def __init__(
        self,
        parser: EbookParser,
        tts_backend: TTSBackend,
        work_dir: Path | None = None,
    ) -> None:
        self._parser = parser
        self._tts = tts_backend
        self._work_dir = work_dir or Path("work")
        self._jobs: dict[str, AudiobookJob] = {}
        self._books: dict[str, Book] = {}

    def _persist(self, job: AudiobookJob) -> None:
        """Save job state to disk for crash recovery."""
        if job.work_dir is not None:
            save_job(job, job.work_dir)

    def inspect(self, path: Path) -> Book:
        """Parse an ebook and return its structured representation."""
        return self._parser.inspect(path)

    def capabilities(self) -> BackendCapabilities:
        """Return available languages and voices from the TTS backend."""
        return self._tts.capabilities()

    def create_job(
        self,
        source_path: Path,
        language: str,
        voice: str,
        output_path: Path,
        speed: float = 1.0,
        paragraph_pause_ms: int = 700,
        chapter_pause_ms: int = 2500,
        scene_break_pause_ms: int = 1500,
        chapter_title_pause_ms: int = 1200,
        keep_intermediates: bool = False,
    ) -> AudiobookJob:
        """Create a new conversion job from an ebook file."""
        book = self.inspect(source_path)
        job = AudiobookJob(
            source_path=source_path,
            book_title=book.title,
            language=language,
            voice=voice,
            speed=speed,
            paragraph_pause_ms=paragraph_pause_ms,
            chapter_pause_ms=chapter_pause_ms,
            scene_break_pause_ms=scene_break_pause_ms,
            chapter_title_pause_ms=chapter_title_pause_ms,
            output_path=output_path,
            keep_intermediates=keep_intermediates,
            work_dir=self._work_dir / "jobs",
        )
        job.work_dir = self._work_dir / "jobs" / job.id
        self._jobs[job.id] = job
        self._books[job.id] = book
        self._persist(job)
        return job

    def get_job(self, job_id: str) -> AudiobookJob:
        """Retrieve a job by ID."""
        if job_id not in self._jobs:
            raise JobError(f"Job not found: {job_id}")
        return self._jobs[job_id]

    def cancel(self, job_id: str) -> AudiobookJob:
        """Cancel a running or pending job."""
        job = self.get_job(job_id)
        terminal = {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        }
        if job.state in terminal:
            raise JobError(
                f"Cannot cancel job in state {job.state.value}"
            )
        job.transition(JobState.CANCELLED)
        self._persist(job)
        return job

    def run(self, job_id: str) -> AudiobookJob:
        """Execute the full conversion pipeline for a job.

        Pipeline:
        CREATED -> ANALYZING -> READY -> SYNTHESIZING -> ASSEMBLING -> COMPLETED

        Supports resume: if the job is already in SYNTHESIZING or
        ASSEMBLING state, earlier phases are skipped. The synthesis
        phase is idempotent — completed segments are not re-generated.
        """
        job = self.get_job(job_id)
        book = self._books[job_id]

        # Phase 1: Analyze (skip if resuming)
        if job.state == JobState.CREATED:
            job.transition(JobState.ANALYZING)
            job.total_segments = sum(
                len(chunk_text(normalize(seg.text)))
                for ch in book.chapters
                for seg in ch.segments
            )
            job.transition(JobState.READY)
            self._persist(job)

        # Phase 2: Synthesize (skip analyze/ready if resuming)
        if job.state in (JobState.READY, JobState.SYNTHESIZING):
            if job.state == JobState.READY:
                job.transition(JobState.SYNTHESIZING)
                self._persist(job)
            self._synthesize(job, book)

        # Phase 3: Assemble (retry if previously failed at ASSEMBLING)
        if job.state in (JobState.SYNTHESIZING, JobState.ASSEMBLING):
            if job.state == JobState.SYNTHESIZING:
                job.transition(JobState.ASSEMBLING)
                self._persist(job)
            self._assemble(job)

        # Phase 4: Cleanup intermediates (unless keep_intermediates is set)
        self._cleanup_intermediates(job)

        job.transition(JobState.COMPLETED)
        self._persist(job)
        return job

    def _synthesize(
        self, job: AudiobookJob, book: Book
    ) -> None:
        """Synthesize all chapters and segments to WAV files."""
        work_dir = self._work_dir / job.id
        segments_dir = work_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        for ch in book.chapters:
            ch_dir = work_dir / "chapters" / f"ch_{ch.index:04d}"
            ch_dir.mkdir(parents=True, exist_ok=True)

            num_segments = len(ch.segments)
            for seg_idx, seg in enumerate(ch.segments):
                if job.state == JobState.CANCELLED:
                    return

                cleaned = normalize(seg.text)
                chunks = chunk_text(cleaned)

                seg_wavs: list[Path] = []
                for i, chunk in enumerate(chunks):
                    wav_path = (
                        segments_dir
                        / f"ch{ch.index:04d}_seg{seg.index:06d}_"
                        f"chunk{i}.wav"
                    )
                    if not wav_path.exists():
                        self._tts.synthesize(
                            chunk,
                            job.language,
                            job.voice,
                            wav_path,
                            speed=job.speed,
                        )
                    seg_wavs.append(wav_path)
                    job.completed_segments += 1

                # Merge chunk WAVs into one segment WAV
                if len(seg_wavs) > 1:
                    merged = (
                        segments_dir
                        / f"ch{ch.index:04d}_seg{seg.index:06d}.wav"
                    )
                    merge_wav_files(seg_wavs, merged)
                elif seg_wavs:
                    merged = seg_wavs[0]
                else:
                    continue

                dest = ch_dir / f"seg_{seg.index:06d}.wav"
                is_last_in_chapter = seg_idx == num_segments - 1
                pause_ms = (
                    job.chapter_pause_ms
                    if is_last_in_chapter
                    else job.paragraph_pause_ms
                )
                append_silence_to_wav(merged, pause_ms, dest)

            # Persist at end of each chapter
            self._persist(job)

    def _assemble(self, job: AudiobookJob) -> None:
        """Assemble WAV segments into the final output.

        Uses FFmpeg to produce M4B with chapters and metadata when
        available. Falls back to WAV merge when FFmpeg is missing.
        """
        work_dir = self._work_dir / job.id
        chapters_dir = work_dir / "chapters"
        output_dir = work_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        if job.output_path is None:
            return

        # Try M4B assembly with FFmpeg
        if ffmpeg.is_available():
            book = self._books[job.id]
            try:
                assemble_m4b(
                    job=job,
                    book=book,
                    chapters_dir=chapters_dir,
                    output_path=job.output_path,
                )
                return
            except Exception as exc:
                logger.warning(
                    "M4B assembly failed, falling back to WAV: %s",
                    exc,
                )

        # Fallback: merge WAV files
        all_wavs: list[Path] = []
        if chapters_dir.exists():
            for ch_dir in sorted(
                chapters_dir.iterdir(),
                key=lambda p: _extract_chapter_index_from_dir(p.name)
            ):
                if ch_dir.is_dir():
                    all_wavs.extend(
                        sorted(
                            ch_dir.glob("seg_*.wav"),
                            key=lambda p: _extract_segment_index_from_file(p.stem)
                        )
                    )

        if all_wavs:
            final_output = output_dir / "final.wav"
            merge_wav_files(all_wavs, final_output)

            job.output_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            shutil.copy2(
                str(final_output), str(job.output_path)
            )

    def _cleanup_intermediates(self, job: AudiobookJob) -> None:
        """Remove intermediate WAV files after successful assembly.

        Preserves job metadata (work/jobs/<id>/) and the final output.
        Skipped when job.keep_intermediates is True.
        """
        if job.keep_intermediates:
            return

        work_dir = self._work_dir / job.id
        if not work_dir.exists():
            return

        for child in work_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
                logger.debug("Removed intermediate directory: %s", child)
            elif child.is_file():
                child.unlink()
                logger.debug("Removed intermediate file: %s", child)

    def resume(self, job_id: str) -> AudiobookJob:
        """Resume a previously interrupted job.

        If the job is not in memory (e.g. after process restart),
        attempts to load it from disk. The synthesis phase is
        idempotent — completed segments are not re-synthesized.
        """
        # Try in-memory first
        if job_id not in self._jobs:
            work_dir = self._work_dir / "jobs" / job_id
            if not work_dir.exists():
                raise JobError(f"Job not found: {job_id}")
            job = load_job(work_dir)
            self._jobs[job.id] = job
            # Re-inspect the book so segments are available
            if job.source_path is not None:
                self._books[job.id] = self._parser.inspect(job.source_path)

        job = self.get_job(job_id)
        if not job.can_resume():
            raise JobError(
                f"Job cannot be resumed from state {job.state.value}"
            )
        # Reset completed_segments for accurate recount
        job.completed_segments = 0
        return self.run(job_id)

    def reassemble(
        self,
        job_id: str,
        output_path: Path | None = None,
    ) -> AudiobookJob:
        """Re-assemble a completed job with a potentially different output.

        Only re-runs the assembly phase using existing WAV segments.
        Does not re-synthesize. Useful for testing different output formats
        or re-generating output after assembly issues.

        The job must be in COMPLETED state and have intermediate WAV files.
        """
        # Try in-memory first
        if job_id not in self._jobs:
            work_dir = self._work_dir / "jobs" / job_id
            if not work_dir.exists():
                raise JobError(f"Job not found: {job_id}")
            job = load_job(work_dir)
            self._jobs[job.id] = job
            # Re-inspect the book so segments are available
            if job.source_path is not None:
                self._books[job.id] = self._parser.inspect(job.source_path)

        job = self.get_job(job_id)

        if job.state != JobState.COMPLETED:
            raise JobError(
                f"Job must be in 'completed' state to reassemble, "
                f"got '{job.state.value}'"
            )

        # Verify intermediate WAV files exist
        work_dir = self._work_dir / job.id
        chapters_dir = work_dir / "chapters"
        if not chapters_dir.exists():
            raise JobError(
                f"No intermediate files found for job {job_id}. "
                f"Cannot reassemble without WAV segments."
            )

        # Update output path if provided
        if output_path is not None:
            job.output_path = output_path

        # Reset state to ASSEMBLING and re-run assembly
        job.transition(JobState.SYNTHESIZING)
        job.transition(JobState.ASSEMBLING)
        self._persist(job)

        self._assemble(job)

        job.transition(JobState.COMPLETED)
        self._persist(job)
        return job

    def list_jobs(self) -> list[AudiobookJob]:
        """Return all jobs known to this engine instance."""
        return list(self._jobs.values())

    def get_resumable_jobs(self) -> list[AudiobookJob]:
        """Return jobs that are in a state eligible for resume."""
        return [j for j in self._jobs.values() if j.can_resume()]
