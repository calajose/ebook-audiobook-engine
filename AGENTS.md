# AGENTS.md — Developer & AI Agent Guidelines

Instructions for future OpenCode sessions (and human contributors) working in
`ebook-audiobook-engine`.

## ⚠️ CRITICAL RULES

1. **NEVER commit or push without explicit user request.**
2. **Domain isolation:** `domain/` must never import `infrastructure/`, `application/`,
   `interfaces/`, or third-party libraries.
   - The third-party part is auto-checked by `tests/domain/test_no_forbidden_imports.py`
     (textual scan for `kokoro`, `ffmpeg`, `fastapi`, `uvicorn`, `requests`, `onnxruntime`,
     `openvino`, `httpx`, `starlette`, `calibre`).
   - Internal layering (`domain` → `audiobook_engine.infrastructure`) is **not** checked by
     any test — enforce it manually.

## Verification (run from repo root)

No CI, Makefile, or lockfile exists — the local venv is the source of truth.
The package is installed editable in `.venv/`:

```bash
./.venv/bin/pytest                 # full suite: 304 tests, ~2s, fully mocked
./.venv/bin/ruff check src tests   # scope matters — see below
./.venv/bin/mypy src tests
./.venv/bin/audiobook-engine --help  # CLI console script
```

- Single test: `./.venv/bin/pytest tests/domain/test_job.py::test_some_name`
- Tests need **no** network, FFmpeg, GPU, or Kokoro models (both are mocked).
- **Don't run `ruff check .`**: it also lints `scripts/experiments/` (scratch scripts) and
  `.specify/` (spec-kit tooling), which are not lint-clean.
- **Don't run `ruff format .` repo-wide**: ~30 of 65 files under `src`/`tests` are
  unformatted — it would create a huge unrelated diff. Format only files you touch.
- Known pre-existing baseline at v0.6.0 (not caused by your change): 1 ruff E501 in
  `mobi_parser.py`, 2 mypy errors in `job_store.py` and `test_kokoro_backend.py`.

## Architecture

```
src/audiobook_engine/
├── domain/              # Pure Python: models, JobState machine, protocols, exceptions
├── application/engine.py     # AudiobookEngine: analyze → synthesize → assemble
├── infrastructure/
│   ├── ebook/          # epub/mobi/txt parsers, normalizer.py, chunker.py
│   ├── tts/kokoro/     # backend.py (synthesis), models.py (auto-downloader)
│   ├── device.py       # CUDA / OpenVINO / CPU provider resolution
│   ├── audio/          # silence, wav_merger, ffmpeg, m4b_assembler
│   ├── metadata/       # (empty placeholder)
│   └── persistence/job_store.py  # job.json serialization
└── interfaces/
    ├── cli/main.py     # Typer CLI (console script: audiobook-engine)
    └── api/            # empty placeholder — HTTP service not started (plan Phase 10)
```

- Entrypoints: CLI `audiobook_engine.interfaces.cli.main:app`; Python API `AudiobookEngine`.
- One shared pipeline for all interfaces — never fork per-interface logic.
- Pipeline: parse → `normalize()` → `chunk_text()` → TTS → merge chunks → append pause → assemble.

## Prosody & audio — verified behavior

- **Only two pauses are actually applied** in `engine.py::_synthesize`: paragraph pause
  (default 700ms) after every segment, chapter pause (2500ms) on the chapter's last segment.
- **Gotcha:** `scene_break_pause_ms` (1500) and `chapter_title_pause_ms` (1200) exist as
  CLI/job fields and persist to `job.json`, but **no code consumes them yet**. Separator
  lines (`***`, `---`, `• • •`) are *removed as noise* by `remove_noise()` inside
  `normalize()` before synthesis. Don't claim they work; wire them up if asked to.
- Silence is exact PCM zero frames appended to the segment WAV (`append_silence_to_wav`),
  so `m4b_assembler.py` chapter marks (measured from WAV headers) match FFmpeg output exactly.
- Kokoro backend: speed clamped to `0.5–2.0`; ~80ms zero tail pad after synthesis to protect
  rising intonation on short questions. `normalize_dialogue()` converts `- `/`– ` to em-dash
  and guillemets/curly quotes to straight quotes before phonemization.
- Chunker: `DEFAULT_MAX_CHARS = 500`, sentence-boundary aware; lowercase dialogue tags
  (`—preguntó ella`) stay attached to the preceding quote.

## Job state machine & persistence

- States: `CREATED → ANALYZING → READY → SYNTHESIZING → ASSEMBLING → COMPLETED`;
  plus `FAILED` / `CANCELLED`. Transition table lives in `domain/job.py`.
- Resume-eligible: `SYNTHESIZING`, `ASSEMBLING` only (`_RESUMABLE_STATES`).
- Synthesis is idempotent: WAV paths are deterministic
  (`ch{chapter:04d}_seg{segment:06d}_chunk{i}.wav`); an existing file is skipped.
- Jobs live in `work/<job-id>/` (default `work/`, gitignored).

## Runtime quirks

- First real synthesis downloads ~354 MB of Kokoro ONNX models to
  `~/.cache/ebook-audiobook-engine/kokoro/` (tests never do this).
- `[cuda]` and `[openvino]` extras are mutually exclusive — install only one per env.
- FFmpeg is optional: required only for M4B output (WAV fallback otherwise).

## Docs & other instruction sources

- `ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md` (phase status), `CHANGELOG.md`.
- Update `CHANGELOG.md` and `README.md` when user-facing features or CLI flags change.
- Spec-kit workflow: `.specify/` + `.opencode/commands/speckit.*.md` — follow those files
  if the user invokes a speckit command.

## Constitution (current phase: document & protect, NOT improve)

`.specify/memory/constitution.md` (v1.0.0, ratified 2026-10-11) governs the current phase.
Non-negotiable principles:

- Current observable behavior is sacred — no behavior change without explicit written
  business decision, even if it looks like a bug.
- Every claim about system behavior needs evidence (file+lines, real data, or named business
  testimony); unverified claims are marked `SUPOSICIÓN` and turned into questions.
- Findings are recorded in the anomaly register (classification: severity/area/type), never
  fixed in this phase — fixes belong to the next module.
- Zero new dependencies, zero refactors (no moves/renames/cleanup, even behavior-preserving).
- All documentation in English.

## Specs regime

- Specifications (`specs/`) and the constitution (`.specify/memory/constitution.md`) live in
  a private repo of their own — NOT in this public repo.
- That repo is cloned separately at `specs-privado/`; here, `specs/` and `.specify/` are
  symlinks into it. All three paths are gitignored.
- To work with them: `cd specs-privado` and commit/push there; to sync, `git pull` there.
- The remote URL exists only in the local git config of `specs-privado` — never document it
  here.
