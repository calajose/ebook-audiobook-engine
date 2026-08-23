# Implementation Plan — Ebook Audiobook Engine

## Objective

Build a standalone multilingual audiobook engine usable through CLI, Python API and local HTTP service. Kokoro is the first TTS backend and EPUB the first ebook format.

## Principles

- One conversion pipeline shared by CLI, Python and HTTP.
- No Calibre dependency.
- Standalone mode must not require a server.
- Dynamic language/voice discovery.
- Persistent, resumable jobs.
- Backend isolation.

## Phase 0 — Repository

Create:

```text
ebook-audiobook-engine/
├── ARCHITECTURE.md
├── IMPLEMENTATION_PLAN.md
├── README.md
├── pyproject.toml
├── src/
└── tests/
```

Configure Python packaging, tests, lint/type checks, logging and FFmpeg documentation.

**Acceptance:** package installs, tests run, CLI starts, no Calibre dependency.

## Phase 1 — Domain and contracts

Implement:

- `Book`, `Chapter`, `TextSegment`;
- `Language`, `Voice`, `BackendCapabilities`;
- `AudiobookJob` and states;
- `EbookParser`;
- `TTSBackend`.

**Acceptance:** domain tests pass and contains no Kokoro/FFmpeg/HTTP/Calibre imports.

## Phase 2 — EPUB

Implement EPUB validation, OPF metadata, spine order, XHTML extraction, chapter detection, language/title/author and cover.

Return the common `Book` model.

**Acceptance:** deterministic reading order and chapter structure.

## Phase 3 — Normalization/chunking

Implement whitespace cleanup, paragraph handling, removal of navigation noise and sentence-aware deterministic chunks.

**Acceptance:** repeated processing produces identical segments and no unintended prose loss.

## Phase 4 — Kokoro

Implement `KokoroBackend` using `kokoro-onnx` and its tokenizer.

Tasks:

- model loading;
- capability discovery;
- language/voice validation;
- WAV synthesis;
- backend error mapping.

**Acceptance:** supported language/voice produces WAV; unsupported combinations fail before synthesis.

## Phase 5 — Application services

Implement common use cases:

- inspect;
- capabilities;
- create job;
- run;
- status;
- resume;
- cancel;
- assemble.

**Acceptance:** complete conversion works in-process without CLI or HTTP.

## Phase 6 — Persistence/resume

Persist job state and intermediate audio:

```text
<workdir>/<job-id>/
├── job.json
├── segments/
├── chapters/
└── output/
```

**Acceptance:** interruption after segment N resumes at N+1 without regenerating completed segments.

## Phase 7 — M4B

Implement ordered WAV assembly, chapter markers, metadata, cover and M4B validation using FFmpeg.

**Acceptance:** playable M4B with chapters and available metadata/cover.

## Phase 8 — Standalone CLI

Provide commands equivalent to:

```text
capabilities
inspect
voices
create/convert
status
resume
cancel
```

The CLI invokes application services directly and does not start or require the HTTP service.

**Acceptance:** clean terminal conversion from EPUB to M4B.

## Phase 9 — Python API

Expose a small documented API around application services.

**Acceptance:** an external Python program can inspect and convert in-process.

## Phase 10 — HTTP service

Implement the local API:

```text
GET  /capabilities
GET  /languages
GET  /voices
POST /books/inspect
POST /jobs
GET  /jobs/{id}
POST /jobs/{id}/resume
POST /jobs/{id}/cancel
```

Use structured JSON responses/errors and localhost by default.

**Acceptance:** external client can perform a complete conversion through HTTP.

## Phase 11 — End-to-end hardening

Test multiple supported languages/voices where available, interruption/resume, missing metadata, invalid selections, missing model and missing FFmpeg.

## Phase 12 — AZW3

Add an AZW3 parser that maps to the same `Book` model. Reuse the entire downstream pipeline.

## Phase 13 — Release

Document installation, CLI, Python API and HTTP API. Add configuration, error codes, packaging and performance checks.

## Definition of Done

- [ ] EPUB converts successfully in standalone mode.
- [ ] CLI needs no server.
- [ ] Python API works in-process.
- [ ] HTTP service works independently.
- [ ] All three share the same application pipeline.
- [ ] Languages and voices are dynamic.
- [ ] Kokoro is isolated.
- [ ] Jobs are resumable.
- [ ] M4B contains chapters.
- [ ] Metadata/cover are handled.
- [ ] Tests pass.
- [ ] No Calibre dependency exists.
- [ ] AZW3 can reuse the downstream pipeline.

## Dependency order

```text
Domain/contracts
  -> EPUB
  -> normalization/chunking
  -> Kokoro
  -> application services
  -> jobs/resume
  -> M4B
  -> CLI
  -> Python API
  -> HTTP API
  -> E2E hardening
  -> AZW3
```

The standalone engine must be useful before Calibre integration begins.
