"""Job persistence — save and load job state to/from JSON.

Job state is saved to <workdir>/<job-id>/job.json after each
pipeline phase, enabling resume after process interruption.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from audiobook_engine.domain.job import AudiobookJob, JobState


def _serialize(job: AudiobookJob) -> dict[str, object]:
    """Convert an AudiobookJob to a JSON-serializable dict."""
    return {
        "id": job.id,
        "state": job.state.value,
        "source_path": str(job.source_path) if job.source_path else None,
        "book_title": job.book_title,
        "backend": job.backend,
        "language": job.language,
        "voice": job.voice,
        "speed": job.speed,
        "paragraph_pause_ms": job.paragraph_pause_ms,
        "chapter_pause_ms": job.chapter_pause_ms,
        "scene_break_pause_ms": job.scene_break_pause_ms,
        "chapter_title_pause_ms": job.chapter_title_pause_ms,
        "keep_intermediates": job.keep_intermediates,
        "work_dir": str(job.work_dir) if job.work_dir else None,
        "output_path": str(job.output_path) if job.output_path else None,
        "total_segments": job.total_segments,
        "completed_segments": job.completed_segments,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


def _deserialize(data: dict[str, object]) -> AudiobookJob:
    """Reconstruct an AudiobookJob from a JSON dict."""
    return AudiobookJob(
        id=str(data["id"]),
        state=JobState(str(data["state"])),
        source_path=Path(str(data["source_path"])) if data.get("source_path") else None,
        book_title=str(data.get("book_title", "")),
        backend=str(data.get("backend", "")),
        language=str(data.get("language", "")),
        voice=str(data.get("voice", "")),
        speed=float(str(data.get("speed", 1.0))),
        paragraph_pause_ms=int(str(data.get("paragraph_pause_ms", 700))),
        chapter_pause_ms=int(str(data.get("chapter_pause_ms", 2500))),
        scene_break_pause_ms=int(str(data.get("scene_break_pause_ms", 1500))),
        chapter_title_pause_ms=int(str(data.get("chapter_title_pause_ms", 1200))),
        keep_intermediates=bool(data.get("keep_intermediates", False)),
        work_dir=Path(str(data["work_dir"])) if data.get("work_dir") else None,
        output_path=Path(str(data["output_path"])) if data.get("output_path") else None,
        total_segments=int(str(data.get("total_segments", 0))),
        completed_segments=int(str(data.get("completed_segments", 0))),
        error_message=str(data["error_message"]) if data.get("error_message") else None,
        created_at=datetime.fromisoformat(str(data["created_at"])),
        updated_at=datetime.fromisoformat(str(data["updated_at"])),
    )


def save_job(job: AudiobookJob, work_dir: Path) -> Path:
    """Save job state to <work_dir>/job.json.

    Returns the path to the saved file.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    job_path = work_dir / "job.json"
    data = _serialize(job)
    job_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return job_path


def load_job(work_dir: Path) -> AudiobookJob:
    """Load job state from <work_dir>/job.json."""
    job_path = work_dir / "job.json"
    if not job_path.exists():
        raise FileNotFoundError(f"No job.json found in {work_dir}")
    data = json.loads(job_path.read_text())
    return _deserialize(data)


def find_resumable_jobs(work_dir: Path) -> list[tuple[str, AudiobookJob]]:
    """Find all jobs in the work directory that can be resumed.

    Returns a list of (job_id, job) tuples for jobs in resumable states.
    """
    if not work_dir.exists():
        return []

    results: list[tuple[str, AudiobookJob]] = []
    for entry in sorted(work_dir.iterdir()):
        if not entry.is_dir():
            continue
        job_json = entry / "job.json"
        if not job_json.exists():
            continue
        try:
            job = load_job(entry)
            if job.can_resume():
                results.append((job.id, job))
        except (FileNotFoundError, ValueError, KeyError):
            continue

    return results
