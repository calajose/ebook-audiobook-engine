# Architecture — Ebook Audiobook Engine

## Purpose

`ebook-audiobook-engine` is an independent, reusable local audiobook-conversion engine. It converts supported ebooks into audiobooks through pluggable TTS backends; Kokoro is the first backend.

It has **no dependency on Calibre** and supports three first-class modes:

1. **Standalone CLI** — conversion without running a service.
2. **Python API** — direct in-process use by other applications.
3. **Local HTTP service** — for Calibre and other external clients.

All three modes use the same domain/application pipeline.

## Architectural principles

- Local-first.
- Multilingual from the beginning.
- TTS backend abstraction.
- EPUB first; AZW3 later.
- Resumable jobs.
- Deterministic text extraction/chunking.
- No Calibre dependency.
- HTTP service optional for standalone use.
- Dynamic language/voice capabilities.
- Thin CLI and HTTP adapters.
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
│       │   │   └── kokoro/
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

Contains `Book`, `Chapter`, `TextSegment`, `Language`, `Voice`, `BackendCapabilities`, `AudiobookJob` and the `EbookParser`/`TTSBackend` contracts. It must not import Kokoro, FFmpeg, HTTP frameworks or Calibre.

### Application

Coordinates inspect, capabilities, create job, run, resume, cancel, status and assembly. CLI and HTTP both invoke this layer.

### Infrastructure

Provides EPUB/AZW3 parsers, Kokoro, persistence, audio handling, FFmpeg/M4B assembly and metadata.

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

Kokoro is the first concrete backend. Only its infrastructure adapter imports Kokoro-specific packages. It loads the model/tokenizer, exposes capabilities, synthesizes WAV and maps backend errors to engine errors.

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

## HTTP service

Minimum conceptual endpoints:

```text
GET  /capabilities
GET  /languages
GET  /voices?language=...
POST /books/inspect
POST /jobs
GET  /jobs/{id}
POST /jobs/{id}/resume
POST /jobs/{id}/cancel
```

The service binds to localhost by default, validates requests, returns structured errors and exposes persisted job progress.

## External clients

Separate applications should use the HTTP contract:

```text
Calibre plugin ──HTTP──> engine service
Other app      ──HTTP──> engine service
CLI            ──direct─> application layer
Python app     ──direct─> Python API
```

## Audio assembly

Intermediate synthesis uses WAV or another lossless format. FFmpeg is the initial implementation for M4B assembly, including ordering, chapters, metadata, cover and encoding.

## Configuration

Support backend, model path, work/output directories, default language/voice, FFmpeg path, concurrency, logging and HTTP bind address/port. Defaults must not assume Spanish.

## Testing

Unit tests cover domain, capabilities, normalization, chunking and state transitions. Integration tests cover EPUB, Kokoro, WAV, FFmpeg, metadata and persistence. An end-to-end test must perform:

```text
EPUB -> Book -> Chapters -> Segments -> Kokoro -> WAV -> M4B
```

## Architectural decision

`ebook-audiobook-engine` is the independent product. It can be executed directly, imported as a Python library or run as a local service. Calibre is not part of the engine.
