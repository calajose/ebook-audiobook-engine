# ebook-audiobook-engine

> **Version: v0.2.0** — Local audiobook conversion engine with configurable prosody, smart dialogue chunking, and pluggable TTS backends.

Developed in [OpenCode](https://opencode.ai) with Gemini and free AI models.

## Current State

| Feature | Status | Notes |
|---|---|---|
| **EPUB → M4B** | ✅ Funcional | Chapters, metadata, cover image embedding via FFmpeg |
| **TXT → M4B** | ✅ Funcional | Paragraph-aware parsing, auto fallback to WAV merger |
| **Kokoro TTS Backend** | ✅ Funcional | Multilingual ONNX local inference with auto model download |
| **Prosody & Pauses** | ✅ Funcional | Configurable paragraph (700ms), chapter (2500ms), and scene break (1500ms) pauses |
| **Dialogue & Interrogatives** | ✅ Mejorado | Natural phrasing: dialog tags kept with questions, em-dash normalization, acoustic tail padding |
| **Chunking** | ✅ Determinado | Max 500 chars, sentence-boundary aware, abbreviation & Spanish punctuation handling |
| **Assembly (WAV/M4B)** | ✅ Validado | Bit-perfect PCM pause insertion and FFmpeg M4B chapter metadata |
| **Tests** | 193 passing | 100% test pass rate with strict typing and linting |

---

## Prosody and Voice Control

The engine provides fine-grained control over narration rhythm and acoustic flow:

- **Paragraph Pauses**: Natural breathing intervals between paragraphs (default: `700ms`).
- **Chapter Pauses**: Extended silence between chapters (default: `2500ms`).
- **Scene Break Pauses**: Narrative pause for section transitions and dividers (default: `1500ms`).
- **Dialogue & Question Phrasing**:
  - Unbroken dialogue incises: Dialogue tags (e.g. *«—¿Por qué? —preguntó ella.»*) are kept in the same synthesis chunk to avoid abrupt cadence breaks.
  - Acoustic Tail Protection: Preserves rising pitch intonations and consonant decays on short interrogatives and exclamations without cutoff from silence trimming.
  - Dialogue Normalization: Automatically converts dialog dashes (`- `, `– `) to em-dash (`— `) and smart quotes (`« »`, `“ ”`) for optimal Kokoro phonemizer stress.
- **Narration Speed**: Configurable speech rate from `0.5x` to `2.0x` (default: `1.0x`; `0.95x` is recommended for Spanish narration clarity).

---

## How Kokoro TTS works

This project uses [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) as an external dependency — the TTS inference engine is **not** bundled in this repository.

| Component | Source | Location |
|---|---|---|
| `kokoro-onnx` (inference runtime) | PyPI | installed via `pip install` |
| Model files (~337 MB) | GitHub releases | `~/.cache/ebook-audiobook-engine/kokoro/` |
| Backend adapter | This repo | `src/.../tts/kokoro/` |

On first synthesis, the engine downloads two model files (~326 MB + ~28 MB) from [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases) and caches them locally. No internet connection is needed after the initial download.

---

## Overview

Converts ebooks (EPUB, TXT) into audiobooks (M4B/WAV) through a local pipeline with no external cloud service dependencies.

**Interfaces:**

- **Standalone CLI** — convert books with rich progress bars and prosody flags
- **Python API** — direct in-process use and programmatic pipeline integration
- **Local HTTP service** — for Calibre and external clients (in development)

---

## Requirements

- Python 3.13+
- FFmpeg (optional, for M4B output with chapters, metadata, and embedded cover art)

---

## Installation

```bash
pip install -e ".[all]"
```

Or with specific extras:

```bash
pip install -e ".[kokoro]"      # TTS backend
pip install -e ".[api]"         # HTTP service
pip install -e ".[dev]"         # Testing/linting
```

---

## Usage

### CLI

```bash
# List available voices
audiobook-engine list-voices
audiobook-engine list-voices -l es

# Convert EPUB with default prosody
audiobook-engine convert book.epub -v ef_dora -o book.m4b

# Convert with custom speed and pause durations
audiobook-engine convert book.epub \
  --voice ef_dora \
  --language es \
  --speed 0.95 \
  --paragraph-pause 800 \
  --chapter-pause 3000 \
  --output book.m4b

# Convert plain text ebook
audiobook-engine convert book.txt -v em_alex -o book.m4b -l es

# Resume interrupted job
audiobook-engine resume <job-id>

# Convert keeping intermediate WAV files (for debugging)
audiobook-engine convert book.epub -v ef_dora -o book.m4b --keep-intermediates
```

### CLI Options for `convert`

| Option | Flag | Default | Description |
|---|---|---|---|
| `--output` | `-o` | `output.m4b` | Path to output file (`.m4b` or `.wav`) |
| `--language` | `-l` | `en` | Target language code (`es`, `en-us`, `en-gb`, `ja`, etc.) |
| `--voice` | `-v` | *(Required)* | Voice identifier (e.g. `ef_dora`, `em_alex`, `af_sarah`) |
| `--speed` | `-s` | `1.0` | Narration speed multiplier (`0.5` to `2.0`) |
| `--paragraph-pause` | | `700` | Pause duration between paragraphs in milliseconds |
| `--chapter-pause` | | `2500` | Pause duration between chapters in milliseconds |
| `--scene-break-pause` | | `1500` | Pause duration for scene breaks in milliseconds |
| `--work-dir` | `-w` | `work/` | Temporary directory for job artifacts and cache |
| `--keep-intermediates` | | `False` | Keep intermediate WAV files after conversion (cleaned by default) |
| `--verbose` | `-V` | `False` | Enable debug logging |

---

### Python API

```python
from pathlib import Path
from audiobook_engine import AudiobookEngine
from audiobook_engine.infrastructure.ebook.epub_parser import EPUBParser
from audiobook_engine.infrastructure.tts.kokoro.backend import KokoroBackend

parser = EPUBParser()
tts = KokoroBackend()
engine = AudiobookEngine(parser, tts, work_dir=Path("work"))

book = engine.inspect(Path("book.epub"))
print(f"Title: {book.title}, Chapters: {len(book.chapters)}")

job = engine.create_job(
    source_path=Path("book.epub"),
    language="es",
    voice="ef_dora",
    output_path=Path("output.m4b"),
    speed=0.95,
    paragraph_pause_ms=800,
    chapter_pause_ms=3000,
)
engine.run(job.id)
```

---

## Architecture

```
src/audiobook_engine/
├── domain/              # Clean Architecture core (no external deps)
│   ├── models.py        # Book, Chapter, TextSegment, Voice, ProsodyConfig
│   ├── job.py           # AudiobookJob state machine & resume logic
│   ├── protocols.py     # EbookParser, TTSBackend contracts
│   └── exceptions.py    # Domain exception hierarchy
├── application/
│   └── engine.py        # AudiobookEngine orchestrator & synthesis pipeline
├── infrastructure/
│   ├── ebook/
│   │   ├── epub_parser.py
│   │   ├── txt_parser.py
│   │   ├── normalizer.py   # Dialogue dashes, quotes, whitespace cleaning
│   │   └── chunker.py      # Sentence & dialogue-aware text chunking
│   ├── tts/kokoro/         # Kokoro ONNX TTS backend adapter
│   │   ├── backend.py      # Synthesis with speed & tail protection
│   │   └── models.py       # Model cache & automated downloader
│   ├── audio/
│   │   ├── silence.py      # WAV silence generation & pause injection
│   │   ├── wav_merger.py   # Raw PCM WAV concatenator with silence support
│   │   ├── ffmpeg.py       # FFmpeg subprocess wrapper
│   │   └── m4b_assembler.py # M4B builder with accurate chapter marks
│   └── persistence/
│       └── job_store.py    # JSON job persistence for crash recovery
└── interfaces/
    └── cli/main.py         # Typer CLI with rich progress & prosody flags
```

---

## Pipeline

```
EPUB/TXT → Parse → Normalize → Chunk → TTS (Kokoro) → Silence Injection → WAV → M4B
                                        ↓
                                  Persistence (job.json)
                                        ↓
                              Cleanup intermediates (unless --keep-intermediates)
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy src tests

# Format
ruff format .
```

---

## License

MIT
