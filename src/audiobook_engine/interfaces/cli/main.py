"""CLI entry point for audiobook-engine.

Provides commands for converting ebooks to audiobooks, listing
available voices, and resuming interrupted jobs.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
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
from audiobook_engine.infrastructure.ebook.mobi_parser import MOBIParser
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
    elif ext in (".mobi", ".azw", ".azw3"):
        return MOBIParser()
    else:
        raise EbookError(
            f"Unsupported format: {ext}. "
            f"Supported: .epub, .txt, .mobi, .azw, .azw3"
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
    chapters: str | None = typer.Option(
        None,
        "--chapters",
        "-c",
        help="Comma-separated chapter indices to convert (e.g. '0,2,5' or '1-3').",
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
        chapter_indices = _parse_chapter_indices(chapters) if chapters else None
    except ValueError:
        console.print("[red]Error:[/red] Invalid chapter specification.")
        raise typer.Exit(1) from None

    if chapter_indices:
        console.print(f"[bold]Selected chapters:[/bold] {chapters}")

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
                chapter_indices=chapter_indices,
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
        langs = ", ".join(lang.code for lang in caps.languages)
        console.print(f"[bold]Languages:[/bold] {langs}")
        console.print(
            f"[bold]Total voices:[/bold] {len(caps.voices)}"
        )
    except TTSBackendError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None


def _parse_chapter_indices(spec: str) -> list[int]:
    """Parse a chapter index specification like '0,2,5' or '1-3'."""
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


@app.command()
def inspect(
    book: str = typer.Argument(..., help="Path to the ebook file."),
    chapters: str | None = typer.Option(
        None,
        "--chapters",
        "-c",
        help="Comma-separated chapter indices to analyze (e.g. '0,2,5' or '1-3').",
    ),
    max_chars: int = typer.Option(
        500,
        "--max-chars",
        help="Max characters per TTS chunk (default: 500).",
    ),
    preview: int | None = typer.Option(
        None,
        "--preview",
        "-p",
        help="Show TTS text chunks for a specific chapter index.",
    ),
    work_dir: str | None = typer.Option(
        None, "--work-dir", "-w", help="Working directory."
    ),
) -> None:
    """Analyze an ebook and show its structure.

    Runs parsing, chapter detection and chunking without generating
    audio. Shows per-chapter statistics and detection warnings.
    """
    source = Path(book)
    if not source.exists():
        console.print(f"[red]Error:[/red] File not found: {book}")
        raise typer.Exit(1)

    chapter_indices = _parse_chapter_indices(chapters) if chapters else None

    engine = _create_engine(work_dir, source)
    with console.status("[bold green]Analyzing ebook..."):
        try:
            result = engine.analyze(
                source, chapter_indices, max_chars, preview_index=preview
            )
        except EbookError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from None

    book_data = result.book
    chapter_analyses = result.chapter_analyses
    warnings = result.warnings

    console.print(f"[bold]Title:[/bold] {book_data.title}")
    console.print(f"[bold]Author:[/bold] {book_data.author}")
    console.print(f"[bold]Language:[/bold] {book_data.language}")
    has_cover = (
        book_data.cover_path is not None
        and getattr(book_data.cover_path, "exists", lambda: True)()
    )
    if has_cover and book_data.cover_path is not None:
        cover_name = getattr(
            book_data.cover_path, "name", str(book_data.cover_path)
        )
        console.print(f"[bold]Cover:[/bold] {cover_name}")
    else:
        console.print("[bold]Cover:[/bold] None")
    console.print(
        f"[bold]Chapters:[/bold] "
        f"{len(chapter_analyses)}"
        f"{f' / {len(book_data.chapters)} total' if chapters else ''}"
    )

    total_chars = sum(ca.chars for ca in chapter_analyses)
    total_words = sum(ca.words for ca in chapter_analyses)
    total_chunks = sum(ca.chunks for ca in chapter_analyses)
    console.print(f"[bold]Total chars:[/bold] {total_chars:,}")
    console.print(f"[bold]Total words:[/bold] {total_words:,}")
    console.print(f"[bold]Total chunks:[/bold] {total_chunks}")

    table = Table(title="Chapter Analysis")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Title", style="cyan")
    table.add_column("Source", style="blue")
    table.add_column("Chars", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Chunks", justify="right", style="green")

    for ca in chapter_analyses:
        table.add_row(
            str(ca.index),
            ca.title,
            ca.source_file or "—",
            f"{ca.chars:,}",
            f"{ca.words:,}",
            str(ca.chunks),
        )

    console.print(table)

    if preview is not None:
        for ca in chapter_analyses:
            if ca.index == preview:
                if not ca.text_chunks:
                    console.print(
                        f"\n[yellow]No text chunks for chapter {preview}[/yellow]"
                    )
                else:
                    console.print(
                        f"\n[bold cyan]Chapter {ca.index}:[/bold cyan] "
                        f"{ca.title} ({len(ca.text_chunks)} chunks)"
                    )
                    for i, chunk in enumerate(ca.text_chunks):
                        console.print(
                            Panel(
                                chunk,
                                title=f"Chunk {i + 1}/{len(ca.text_chunks)}",
                                border_style="dim",
                                padding=(0, 1),
                            )
                        )
                break
        else:
            console.print(f"\n[red]Chapter {preview} not found.[/red]")

    if warnings:
        console.print()
        for w in warnings:
            console.print(f"[yellow]Warning:[/yellow] {w}")


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
        total = job.total_segments or 1
        progress = job.completed_segments / total * 100
        console.print(
            f"[bold]Progress:[/bold] "
            f"{job.completed_segments}/{job.total_segments} "
            f"segments ({progress:.1f}%)"
        )
        console.print(
            f"[bold]Created:[/bold] "
            f"{job.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )
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


@app.command()
def reassemble(
    job_id: str = typer.Argument(..., help="Job ID to reassemble."),
    output: str | None = typer.Option(
        None, "--output", "-o", help="New output path (optional)."
    ),
    work_dir: str | None = typer.Option(
        None, "--work-dir", "-w", help="Working directory."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-V", help="Enable verbose logging."
    ),
) -> None:
    """Re-assemble a completed job without re-synthesizing.

    Useful for testing different output formats or re-generating
    output after assembly issues.
    """
    _setup_logging(verbose)

    engine = _create_engine(work_dir)

    try:
        # Load from disk if not in memory
        work = Path(work_dir or "work") / "jobs" / job_id
        if work.exists():
            from audiobook_engine.infrastructure.persistence.job_store import (
                load_job,
            )

            job = load_job(work)
            engine._jobs[job.id] = job
            if job.source_path is not None:
                engine._books[job.id] = engine.inspect(job.source_path)

        new_output = Path(output) if output else None
        job = engine.reassemble(job_id, new_output)
        console.print(
            f"[bold green]Done![/bold green] "
            f"Output: {job.output_path}"
        )
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
    job_id: str | None = typer.Argument(
        None, help="Job ID to resume (omitting lists all jobs)."
    ),
    work_dir: str | None = typer.Option(
        None, "--work-dir", "-w", help="Working directory."
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
    """Resume a previously interrupted job, or list all jobs."""
    _setup_logging(verbose)

    engine = _create_engine(work_dir)
    base_work = Path(work_dir) if work_dir else Path("work")
    jobs_dir = base_work / "jobs"

    if job_id is None:
        from audiobook_engine.infrastructure.persistence.job_store import (
            find_all_jobs,
        )

        all_jobs = find_all_jobs(jobs_dir)
        if not all_jobs:
            console.print("[yellow]No jobs found.[/yellow]")
            raise typer.Exit(0)

        table = Table(title="All Jobs")
        table.add_column("Job ID", style="cyan")
        table.add_column("Book Title", style="green")
        table.add_column("State", style="yellow")
        table.add_column("Progress", justify="right")
        table.add_column("Created", style="dim")

        for jid, job in all_jobs:
            title = job.book_title or "—"
            total = job.total_segments or 1
            pct = job.completed_segments / total * 100
            progress_str = (
                f"{job.completed_segments}/{job.total_segments} "
                f"({pct:.1f}%)"
            )
            created_str = job.created_at.strftime("%Y-%m-%d %H:%M:%S")

            state_style = "yellow"
            if job.state == JobState.COMPLETED:
                state_style = "bold green"
            elif job.state == JobState.FAILED:
                state_style = "bold red"
            elif job.state == JobState.CANCELLED:
                state_style = "dim"
            elif job.state == JobState.CREATED:
                state_style = "cyan"

            table.add_row(
                jid,
                title,
                f"[{state_style}]{job.state.value}[/{state_style}]",
                progress_str,
                created_str,
            )

        console.print(table)
        console.print(
            "\n[dim]Resume with: audiobook-engine resume <job-id>[/dim]"
        )
        return

    try:
        job = engine.get_job(job_id)
    except JobError:
        # Try loading from default/specified work directory
        from audiobook_engine.infrastructure.persistence.job_store import (
            load_job,
        )

        work = jobs_dir / job_id
        if not work.exists():
            console.print(f"[red]Error:[/red] Job not found: {job_id}")
            raise typer.Exit(1) from None
        job = load_job(work)
        # Re-create engine with correct parser for source format
        if job.source_path is not None:
            engine = _create_engine(work_dir, job.source_path)
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


@app.command()
def clean(
    work_dir: str | None = typer.Option(
        None,
        "--work-dir",
        "-w",
        help="Working directory to clean (default: 'work').",
    ),
    job_id: str | None = typer.Argument(
        None,
        help="Job ID to clean (removes only that job).",
    ),
) -> None:
    """Clean the working directory or a specific job."""
    work_path = Path(work_dir) if work_dir else Path("work")
    if not work_path.exists():
        console.print(
            f"[yellow]Working directory '{work_path}' does not exist.[/yellow]"
        )
        raise typer.Exit(0)

    if job_id:
        job_path = work_path / "jobs" / job_id
        if not job_path.exists():
            console.print(
                f"[yellow]Job '{job_id}' not found in '{work_path}'.[/yellow]"
            )
            raise typer.Exit(0)
        try:
            shutil.rmtree(job_path)
            console.print(f"[green]Cleaned job:[/green] {job_id}")
        except Exception as exc:
            console.print(f"[red]Error cleaning job '{job_id}':[/red] {exc}")
            raise typer.Exit(1) from None
    else:
        try:
            shutil.rmtree(work_path)
            console.print(f"[green]Cleaned working directory:[/green] {work_path}")
        except Exception as exc:
            console.print(f"[red]Error cleaning working directory:[/red] {exc}")
            raise typer.Exit(1) from None
