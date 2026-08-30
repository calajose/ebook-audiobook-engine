# AGENTS.md — Developer & AI Agent Guidelines

Welcome to `ebook-audiobook-engine`. This document serves as the single source of truth for AI agents (and human contributors) developing, extending, or maintaining this codebase.

---

## ⚠️ CRITICAL RULES

1. **NEVER commit or push without explicit user request.** Only run `git commit` and `git push` when the user explicitly asks for it (e.g. "commit", "push", "commit y push"). Do NOT proactively commit changes, even if they seem complete.
2. **Domain isolation:** `domain/` must NEVER import from `infrastructure/`, `application/`, `interfaces/`, or third-party libraries.

---

## 1. Project Overview & Architectural Vision

`ebook-audiobook-engine` is a local, offline-first ebook-to-audiobook conversion pipeline built with Clean Architecture / Hexagonal Architecture principles.

### Key Goals:
- **Zero Cloud Lock-in**: All synthesis runs locally (default engine: Kokoro TTS via ONNX runtime).
- **Audiobook Quality Prosody**: Natural pauses between paragraphs, scene breaks, and chapters, with speech rate control and dialogue tag preservation.
- **Resumable & Fault Tolerant**: Long conversions persist state to disk (`job.json`) and resume seamlessly without re-synthesizing completed segments.
- **Multi-Interface**: CLI, direct Python API, and upcoming local HTTP service (FastAPI).

---

## 2. Architecture & Layer Separation

Strict layering is enforced across the codebase. Never violate the dependency inversion rule.

```
src/audiobook_engine/
├── domain/                  # CORE: Pure Python dataclasses & protocols. ZERO external dependencies.
│   ├── models.py            # Book, Chapter, TextSegment, Voice, ProsodyConfig
│   ├── job.py               # AudiobookJob state machine & transition rules
│   ├── protocols.py         # EbookParser, TTSBackend contracts
│   └── exceptions.py        # Domain exception hierarchy (AudiobookEngineError base)
│
├── application/             # APPLICATION SERVICES: Orchestration layer.
│   └── engine.py            # AudiobookEngine (coordinates parsers, TTS, chunker, assembler)
│
├── infrastructure/          # ADAPTERS & IMPLEMENTATIONS: External libraries allowed.
│   ├── ebook/
│   │   ├── epub_parser.py   # ebooklib wrapper -> Book model
│   │   ├── txt_parser.py    # Plain text / paragraph parser -> Book model
│   │   ├── normalizer.py    # Noise removal & dialogue normalization
│   │   └── chunker.py       # Sentence & dialogue-aware deterministic text chunking
│   ├── tts/kokoro/
│   │   ├── backend.py       # kokoro-onnx TTS backend with speed & acoustic protection
│   │   └── models.py        # Model downloading & user cache management
│   ├── audio/
│   │   ├── silence.py       # PCM silence generation & pause injection
│   │   ├── wav_merger.py    # Bit-perfect WAV concatenation with pause support
│   │   ├── ffmpeg.py        # Safe subprocess wrapper for FFmpeg
│   │   └── m4b_assembler.py # M4B assembly with metadata, chapters, cover art
│   └── persistence/
│       └── job_store.py     # JSON serialization & crash recovery for jobs
│
└── interfaces/              # CLIENT ENTRY POINTS:
    ├── cli/main.py          # Typer + Rich CLI application
    └── api/                 # FastAPI HTTP service (Phase 9 / in development)
```

### Critical Rules:
1. **Domain Isolation**: `domain/` must NEVER import from `infrastructure/`, `application/`, `interfaces/`, or third-party libraries (e.g. `kokoro`, `ffmpeg`, `fastapi`, `requests`, `ebooklib`). This is automatically enforced by `test_no_forbidden_imports.py`.
2. **Pluggable Backends**: All TTS engines implement the `TTSBackend` protocol (`capabilities()` and `synthesize()`).
3. **Pluggable Parsers**: All ebook parsers implement the `EbookParser` protocol (`inspect() -> Book`).

---

## 3. Audio & Prosody Subsystem

### Pauses & Silence Injection:
- Pauses are not arbitrary sleeps; they are exact PCM zero-byte frames matching the audio format (sample rate, channels, sample width).
- **Paragraph pause** (default: `700ms`): Injected at the end of each paragraph segment WAV.
- **Chapter pause** (default: `2500ms`): Injected at the final segment of each chapter.
- **Scene break pause** (default: `1500ms`): Injected for section divider lines (`***`, `---`).
- **Chapter Markers**: `m4b_assembler.py` calculates durations by reading the generated WAV headers; because silence is appended directly to segment WAVs, FFmpeg timestamps match the audio down to the exact millisecond.

### Text Normalization & Dialogue Phrasing:
- In `normalizer.py`:
  - Spanish dialogue dashes (`- `, `– `) are converted to em-dash (`— `) so Kokoro’s phonemizer processes the dialogue onset accurately.
  - Guillemets (`« »`) and curly quotes (`“ ”`) are normalized to standard quotes.
  - Spacing after inverted punctuation (`¿ `, `¡ `) is cleaned.
- In `chunker.py`:
  - `split_sentences` inspects trailing tokens: if a question or quote is followed by a lowercase dialogue tag (e.g. `—preguntó`, `dijo él`), it does NOT split the tag away from the quote.
  - Sentences are grouped into chunks up to `max_chars` (default `500`).

### Kokoro TTS Synthesis & Tail Padding:
- In `kokoro/backend.py`:
  - Audio trimming in Kokoro can abruptly cut off the rising pitch of short questions (e.g. *«¿Por qué?»*).
  - Subtle acoustic padding (`~80ms`) is appended to synthesized arrays to guarantee smooth acoustic decay before WAV concatenation.
  - Supports speech speed multipliers (`0.5x` to `2.0x`).

---

## 4. Job State Machine & Persistence

```
[CREATED] ──► [ANALYZING] ──► [READY] ──► [SYNTHESIZING] ──► [ASSEMBLING] ──► [COMPLETED]
    │              │             │               │                 │
    ▼              ▼             ▼               ▼                 ▼
[CANCELLED]   [CANCELLED]   [CANCELLED]     [CANCELLED]         [FAILED]
                   │             │               │
                   ▼             ▼               ▼
                [FAILED]      [FAILED]        [FAILED]
```

- States eligible for resume: `SYNTHESIZING`, `ASSEMBLING`.
- Synthesis is idempotent: segment WAV paths are deterministic (`chXXXX_segYYYYYY_chunkZ.wav`). If a WAV file exists on disk, `_synthesize` skips re-generation.

---

## 5. Development & Verification Workflow

Before finalizing any changes, always run the full verification loop:

```bash
# 1. Run test suite (must pass 100%)
./.venv/bin/pytest

# 2. Linting & import order
./.venv/bin/ruff check .

# 3. Type checking (strict mode)
./.venv/bin/mypy src tests
```

### Adding New Features:
1. If adding a domain concept or configuration field, update `domain/models.py`, `domain/job.py`, and `infrastructure/persistence/job_store.py` (ensure backward-compatible JSON serialization).
2. If adding or modifying TTS backend methods, update `domain/protocols.py` first.
3. Write unit tests for all domain and infrastructure logic in `tests/`.
4. Update `CHANGELOG.md` and `README.md` when user-facing features or CLI flags change.
