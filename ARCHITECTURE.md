# Architecture — Ebook Audiobook Engine

## Purpose

`ebook-audiobook-engine` is an independent, reusable local audiobook-conversion engine. It converts supported ebooks into audiobooks through pluggable TTS backends: Kokoro (default) and Phonon 40M.

It has **no dependency on Calibre** and supports two first-class modes:

1. **Standalone CLI** — conversion without running a service.
2. **Python API** — direct in-process use by other applications.

Both modes use the same domain/application pipeline.

## Architectural principles

- Local-first.
- Multilingual from the beginning.
- TTS backend abstraction.
- EPUB first; AZW3 later.
- Resumable jobs.
- Deterministic text extraction/chunking.
- No Calibre dependency.
- Dynamic language/voice capabilities.
- Thin CLI adapters.
- One conversion pipeline, not separate implementations per interface.

## High-level architecture

```text
                 CLI / Python API / HTTP API
                           |
                           v
                  Application services
                           |
                           v
                    Domain / Core
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
     Ebook adapters    TTS backends     Audio/output
       EPUB/AZW3         Kokoro            FFmpeg
                                          M4B
```

The HTTP server is an interface to the engine, not a second engine.

## Project structure

```text
ebook-audiobook-engine/
├── ARCHITECTURE.md
├── IMPLEMENTATION_PLAN.md
├── README.md
├── pyproject.toml
├── src/
│   └── audiobook_engine/
│       ├── domain/
│       ├── application/
│       ├── infrastructure/
│       │   ├── ebook/
│       │   ├── tts/
│       │   │   ├── kokoro/
│       │   │   └── phonon/
│       │   ├── audio/
│       │   ├── metadata/
│       │   └── persistence/
│       └── interfaces/
│           ├── cli/
│           └── api/
└── tests/
```

## Core layers

### Domain

Contains `Book`, `Chapter`, `TextSegment`, `Language`, `Voice`, `BackendCapabilities`, `AudiobookJob` and the `EbookParser`/`TTSBackend` contracts. It must not import Kokoro, ptts (Phonon), FFmpeg, HTTP frameworks or Calibre.

### Application

Coordinates inspect, capabilities, create job, run, resume, cancel, status and assembly. CLI and HTTP both invoke this layer.

### Infrastructure

Provides EPUB/AZW3 parsers, the Kokoro and Phonon 40M TTS adapters, persistence, audio handling, FFmpeg/M4B assembly and metadata.

### Interfaces

Only thin adapters for CLI and HTTP.

## Ebook abstraction

```python
class EbookParser(Protocol):
    def inspect(self, path: Path) -> Book: ...
```

EPUB is first. AZW3 must later produce the same `Book` model.

```text
EPUB ─┐
      ├──> Book -> Chapters -> Segments -> TTS -> M4B
AZW3 ─┘
```

## TTS abstraction

```python
class TTSBackend(Protocol):
    def capabilities(self) -> BackendCapabilities: ...
    def synthesize(
        self,
        text: str,
        language: str,
        voice: str,
        output_path: Path,
    ) -> None: ...
```

Language and voice combinations are discovered from backend capabilities; they are never hard-coded.

## Kokoro backend

Kokoro is the default concrete backend. Only its infrastructure adapter imports Kokoro-specific packages. It loads the model/tokenizer, exposes capabilities, synthesizes WAV and maps backend errors to engine errors.

## Phonon 40M backend

Phonon 40M (Gradium) is the second concrete backend, selected with `--backend phonon` (alias `phonom`). Only its infrastructure adapter imports the `ptts` runtime, shipped as the optional `[phonon]` extra; `domain/` never sees it.

- **One checkpoint per language** (`de`, `en`, `es`, `fr`, `pt`), cached in `<user cache>/ebook-audiobook-engine/phonon/phonon-40m-<lang>/` with `config.json`, `tokenizer.json`, a weight file (`model.q8.gguf` or `model.safetensors`) and optionally `default-voice.safetensors` / `voices/`.
- **First-use download** from `gradium/phonon-40m-<lang>`. Those repositories are gated, so a token (`HF_TOKEN` or `hf auth login`) is required. When the download cannot run or the local directory is incomplete, the adapter raises a `TTSBackendError` whose message carries the model URL, the required files and the exact absolute target directory.
- **Language resolution** happens in the CLI (explicit `--language`, then ebook metadata, then an interactive prompt; non-interactive sessions fail instead of blocking). The backend itself normalizes regional tags and rejects unsupported languages.
- **Prosody is unchanged**: the adapter validates `speed` in `[0.5, 2.0]`, maps it to Phonon's `duration_delta`, appends the same 80 ms acoustic tail and writes a WAV, so silence injection, chapter marks and `.m4b`/`.wav` assembly are backend-agnostic.
- **Resume** re-instantiates the backend recorded in `job.json`; historical records without the field are read as `kokoro` and never rewritten.

## Processing pipeline

```text
ebook
  -> reading order
  -> text extraction
  -> normalization
  -> sentence-aware chunking
  -> TextSegments
  -> TTS
  -> WAV intermediates
  -> M4B + chapters + metadata + cover
```

## Jobs and resumability

Suggested states:

```text
CREATED -> ANALYZING -> READY -> SYNTHESIZING -> ASSEMBLING -> COMPLETED
                                   |                         |
                                   +-> FAILED <--------------+
                                   +-> CANCELLED
```

A job persists source information, backend/language/voice, progress, intermediate files, errors and timestamps.

Example working directory:

```text
<workdir>/<job-id>/
├── job.json
├── segments/
├── chapters/
└── output/
```

Completed segments must be reusable after interruption.

## Standalone CLI

Example:

```bash
audiobook-engine convert book.epub   --language es   --voice <voice-id>   --output libro.m4b
```

This must work without starting the HTTP server.

## Python API

A documented public API should allow:

```python
from audiobook_engine import AudiobookEngine

engine = AudiobookEngine(...)
job = engine.create_job(...)
engine.run(job.id)
```

It must not require HTTP for in-process applications.

## Audio assembly

Intermediate synthesis uses WAV or another lossless format. FFmpeg is the initial implementation for M4B assembly, including ordering, chapters, metadata, cover and encoding.

## Configuration

Support backend, model path, work/output directories, default language/voice, FFmpeg path, concurrency and logging. Defaults must not assume Spanish.

## Testing

Unit tests cover domain, capabilities, normalization, chunking and state transitions. Integration tests cover EPUB, Kokoro, WAV, FFmpeg, metadata and persistence. An end-to-end test must perform:

```text
EPUB -> Book -> Chapters -> Segments -> TTS backend (Kokoro | Phonon 40M) -> WAV -> M4B
```

## Architectural decision

`ebook-audiobook-engine` is the independent product. It can be executed directly or imported as a Python library. Calibre is not part of the engine.
