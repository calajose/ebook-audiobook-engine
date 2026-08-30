"""CLI entry point for audiobook-engine.

Provides commands for converting ebooks to audiobooks, listing
available voices, and resuming interrupted jobs.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from audiobook_engine.application.engine import AudiobookEngine
from audiobook_engine.domain.exceptions import (
    EbookError,
    JobError,
    TTSBackendError,
)
from audiobook_engine.domain.job import JobState
from audiobook_engine.infrastructure.ebook.epub_parser import EPUBParser
from audiobook_engine.infrastructure.ebook.txt_parser import TXTParser
from audiobook_engine.infrastructure.tts.kokoro.backend import (
    _voice_to_language,
)

if TYPE_CHECKING:
    from audiobook_engine.domain.protocols import EbookParser

app = typer.Typer(
    name="audiobook-engine",
    help="Local audiobook-conversion engine with pluggable TTS backends.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _get_parser(source: Path) -> EbookParser:
    """Return the appropriate parser for the file extension."""
    ext = source.suffix.lower()
    if ext == ".epub":
        return EPUBParser()
    elif ext == ".txt":
        return TXTParser()
    else:
        raise EbookError(
            f"Unsupported format: {ext}. "
            f"Supported: .epub, .txt"
        )


def _create_engine(
    work_dir: str | None,
    source: Path | None = None,
) -> AudiobookEngine:
    """Create an AudiobookEngine with appropriate parser and TTS backend."""
    from audiobook_engine.infrastructure.tts.kokoro.backend import (
        KokoroBackend,
    )

    parser = _get_parser(source) if source else EPUBParser()
    tts = KokoroBackend()
    work = Path(work_dir) if work_dir else Path("work")
    return AudiobookEngine(parser, tts, work_dir=work)


def _display_progress(
    job_id: str,
    engine: AudiobookEngine,
    run_thread: threading.Thread | None = None,
) -> None:
    """Display progress while the job runs.

    If run_thread is provided, waits for it to complete and
    captures any exception.
    """
    job = engine.get_job(job_id)
    total = job.total_segments or 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting...", total=total)

        while True:
            job = engine.get_job(job_id)
            progress.update(
                task,
                completed=job.completed_segments,
                total=job.total_segments or 1,
                description=f"[{job.state.value}]",
            )

            if job.state in (
                JobState.COMPLETED,
                JobState.FAILED,
                JobState.CANCELLED,
            ):
                break

            # Check if run thread finished
            if run_thread is not None and not run_thread.is_alive():
                # Thread done, do one final update
                job = engine.get_job(job_id)
                progress.update(
                    task,
                    completed=job.completed_segments,
                    total=job.total_segments or 1,
                    description=f"[{job.state.value}]",
                )
                break

            time.sleep(0.2)


@app.command()
def convert(
    book: str = typer.Argument(..., help="Path to the ebook file."),
    output: str = typer.Option(
        "output.m4b", "--output", "-o", help="Output audiobook path."
    ),
    language: str = typer.Option(
        "en", "--language", "-l", help="Target language code."
    ),
    voice: str = typer.Option(
        ..., "--voice", "-v", help="Voice identifier."
    ),
    speed: float = typer.Option(
        1.0, "--speed", "-s", help="Speech speed multiplier (0.5 to 2.0)."
    ),
    paragraph_pause: int = typer.Option(
        700,
        "--paragraph-pause",
        help="Pause between paragraphs in milliseconds (default: 700).",
    ),
    chapter_pause: int = typer.Option(
        2500,
        "--chapter-pause",
        help="Pause between chapters in milliseconds (default: 2500).",
    ),
    scene_break_pause: int = typer.Option(
        1500,
        "--scene-break-pause",
        help="Pause for scene breaks in milliseconds (default: 1500).",
    ),
    work_dir: str | None = typer.Option(
        None, "--work-dir", "-w", help="Working directory for temp files."
    ),
    keep_intermediates: bool = typer.Option(
        False,
        "--keep-intermediates",
        help="Keep intermediate WAV files after conversion.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-V", help="Enable verbose logging."
    ),
) -> None:
    """Convert an ebook to an audiobook with configurable prosody and pauses."""
    _setup_logging(verbose)

    source = Path(book)
    if not source.exists():
        console.print(f"[red]Error:[/red] File not found: {book}")
        raise typer.Exit(1)

    try:
        engine = _create_engine(work_dir, source)
    except EbookError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    with console.status("[bold green]Inspecting ebook..."):
        book_data = engine.inspect(source)

    console.print(
        f"[bold]Book:[/bold] {book_data.title} "
        f"by {book_data.author}"
    )
    console.print(
        f"[bold]Chapters:[/bold] {len(book_data.chapters)}"
    )

    try:
        with console.status("[bold green]Creating job..."):
            job = engine.create_job(
                source_path=source,
                language=language,
                voice=voice,
                output_path=Path(output),
                speed=speed,
                paragraph_pause_ms=paragraph_pause,
                chapter_pause_ms=chapter_pause,
                scene_break_pause_ms=scene_break_pause,
                keep_intermediates=keep_intermediates,
            )
    except (TTSBackendError, JobError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    try:
        with console.status("[bold green]Validating voice..."):
            caps = engine.capabilities()

            # Auto-detect language from voice ID
            detected_lang = _voice_to_language(voice)
            if detected_lang and detected_lang != language:
                console.print(
                    f"[yellow]Voice '{voice}' uses {detected_lang}, "
                    f"overriding --language {language}[/yellow]"
                )
                language = detected_lang
                # Update job with corrected language
                job.language = language

            valid_voice = any(
                v.id == voice
                for v in caps.voices
                if v.language_code == language
            )
            if not valid_voice:
                # Try any voice for the language
                lang_voices = [
                    v for v in caps.voices
                    if v.language_code == language
                ]
                if lang_voices:
                    console.print(
                        f"[yellow]Voice '{voice}' not found. "
                        f"Using {lang_voices[0].id} instead.[/yellow]"
                    )
                    job.voice = lang_voices[0].id
                else:
                    console.print(
                        f"[red]Error:[/red] No voices for "
                        f"language '{language}'"
                    )
                    raise typer.Exit(1)
    except TTSBackendError as exc:
        console.print(f"[yellow]Warning:[/yellow] {exc}")
        console.print("[yellow]Proceeding anyway...[/yellow]")

    console.print(f"[bold]Job ID:[/bold] {job.id}")

    # Run engine in background thread
    run_error: Exception | None = None

    def _run_job() -> None:
        nonlocal run_error
        try:
            engine.run(job.id)
        except Exception as exc:
            run_error = exc

    run_thread = threading.Thread(target=_run_job, daemon=True)
    run_thread.start()

    try:
        _display_progress(job.id, engine, run_thread)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Job saved.[/yellow]")
        console.print(
            f"[dim]Resume with: audiobook-engine resume {job.id}[/dim]"
        )
        raise typer.Exit(0) from None

    # Check for errors from the run thread
    if run_error is not None:
        console.print(f"\n[red]Error:[/red] {run_error}")
        raise typer.Exit(1) from None

    final_job = engine.get_job(job.id)

    if final_job.state == JobState.COMPLETED:
        console.print(
            f"\n[bold green]Done![/bold green] "
            f"Output: {output}"
        )
    else:
        console.print(
            f"\n[red]Failed:[/red] {final_job.error_message}"
        )
        raise typer.Exit(1)


@app.command()
def capabilities(
    work_dir: str | None = typer.Option(
        None, "--work-dir", "-w", help="Working directory."
    ),
) -> None:
    """Display supported languages and voice counts."""
    engine = _create_engine(work_dir)
    try:
        caps = engine.capabilities()
        console.print(f"[bold]Languages:[/bold] {', '.join(lang.code for lang in caps.languages)}")
        console.print(f"[bold]Total voices:[/bold] {len(caps.voices)}")
    except TTSBackendError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None


@app.command()
def inspect(
    book: str = typer.Argument(..., help="Path to the ebook file."),
    work_dir: str | None = typer.Option(
        None, "--work-dir", "-w", help="Working directory."
    ),
) -> None:
    """Analyze an ebook and show its structure."""
    source = Path(book)
    if not source.exists():
        console.print(f"[red]Error:[/red] File not found: {book}")
        raise typer.Exit(1)
    
    engine = _create_engine(work_dir, source)
    with console.status("[bold green]Inspecting ebook..."):
        try:
            book_data = engine.inspect(source)
            total_segments = sum(len(ch.segments) for ch in book_data.chapters)
            console.print(f"[bold]Title:[/bold] {book_data.title}")
            console.print(f"[bold]Author:[/bold] {book_data.author}")
            console.print(f"[bold]Language:[/bold] {book_data.language}")
            console.print(f"[bold]Chapters:[/bold] {len(book_data.chapters)}")
            console.print(f"[bold]Total segments:[/bold] {total_segments}")
        except EbookError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from None


@app.command()
def status(
    job_id: str = typer.Argument(..., help="Job ID to check."),
    work_dir: str | None = typer.Option(
        None, "--work-dir", "-w", help="Working directory."
    ),
) -> None:
    """Show the status of a job."""
    engine = _create_engine(work_dir)
    try:
        # Load from disk if not in memory
        work = Path(work_dir or "work") / "jobs" / job_id
        if work.exists():
            from audiobook_engine.infrastructure.persistence.job_store import load_job
            job = load_job(work)
            engine._jobs[job.id] = job
        
        job = engine.get_job(job_id)
        console.print(f"[bold]State:[/bold] {job.state.value}")
        progress = (job.completed_segments / job.total_segments * 100) if job.total_segments > 0 else 0
        console.print(f"[bold]Progress:[/bold] {job.completed_segments}/{job.total_segments} segments ({progress:.1f}%)")
        console.print(f"[bold]Created:[/bold] {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    except JobError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None


@app.command()
def cancel(
    job_id: str = typer.Argument(..., help="Job ID to cancel."),
    work_dir: str | None = typer.Option(
        None, "--work-dir", "-w", help="Working directory."
    ),
) -> None:
    """Cancel a running job."""
    engine = _create_engine(work_dir)
    try:
        # Load from disk if not in memory
        work = Path(work_dir or "work") / "jobs" / job_id
        if work.exists():
            from audiobook_engine.infrastructure.persistence.job_store import load_job
            job = load_job(work)
            engine._jobs[job.id] = job

        engine.cancel(job_id)
        console.print(f"[green]Cancelled job:[/green] {job_id}")
    except JobError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

@app.command("list-voices")
def list_voices(
    language: str | None = typer.Option(
        None, "--language", "-l", help="Filter by language code."
    ),
    work_dir: str | None = typer.Option(
        None, "--work-dir", "-w", help="Working directory."
    ),
) -> None:
    """List available voices and languages."""
    engine = _create_engine(work_dir)

    try:
        caps = engine.capabilities()
    except TTSBackendError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    table = Table(title="Available Voices")
    table.add_column("Voice ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Language", style="yellow")

    lang_map = {lang.code: lang.name for lang in caps.languages}

    for voice in caps.voices:
        if language and voice.language_code != language:
            continue
        lang_name = lang_map.get(voice.language_code, voice.language_code)
        table.add_row(voice.id, voice.name, lang_name)

    console.print(table)


@app.command()
def resume(
    job_id: str = typer.Argument(..., help="Job ID to resume."),
    keep_intermediates: bool = typer.Option(
        False,
        "--keep-intermediates",
        help="Keep intermediate WAV files after conversion.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-V", help="Enable verbose logging."
    ),
) -> None:
    """Resume a previously interrupted job."""
    _setup_logging(verbose)

    engine = _create_engine(None)

    try:
        job = engine.get_job(job_id)
    except JobError:
        # Try loading from default work directory
        from audiobook_engine.infrastructure.persistence.job_store import (
            load_job,
        )

        work = Path("work") / "jobs" / job_id
        if not work.exists():
            console.print(f"[red]Error:[/red] Job not found: {job_id}")
            raise typer.Exit(1) from None
        job = load_job(work)
        # Re-create engine with correct parser for source format
        if job.source_path is not None:
            engine = _create_engine(None, job.source_path)
        engine._jobs[job.id] = job
        if job.source_path is not None:
            engine._books[job.id] = engine.inspect(job.source_path)

    job.keep_intermediates = keep_intermediates

    if not job.can_resume():
        console.print(
            f"[red]Error:[/red] Job is in state "
            f"'{job.state.value}' and cannot be resumed."
        )
        raise typer.Exit(1)

    console.print(
        f"[bold]Resuming job:[/bold] {job.id} "
        f"({job.state.value})"
    )

    # Run engine in background thread
    run_error: Exception | None = None

    def _run_job() -> None:
        nonlocal run_error
        try:
            engine.resume(job_id)
        except Exception as exc:
            run_error = exc

    run_thread = threading.Thread(target=_run_job, daemon=True)
    run_thread.start()

    try:
        _display_progress(job.id, engine, run_thread)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Job saved.[/yellow]")
        console.print(
            f"[dim]Resume with: audiobook-engine resume {job.id}[/dim]"
        )
        raise typer.Exit(0) from None

    # Check for errors from the run thread
    if run_error is not None:
        console.print(f"\n[red]Error:[/red] {run_error}")
        raise typer.Exit(1) from None

    console.print("[bold green]Done![/bold green]")
