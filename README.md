# ebook-audiobook-engine

> **Status: Alpha** — functional pipeline, prosody improvements in progress

Local audiobook-conversion engine with pluggable TTS backends.

Developed in [OpenCode](https://opencode.ai) with Gemini and free AI models.

## Current State

| Format | Status |
|--------|--------|
| EPUB → M4B | ✅ Funcional |
| TXT → M4B | ✅ Funcional / Experimental |
| Kokoro TTS | ✅ Funcional |
| Chunking | Máximo 500 caracteres, respetando oraciones |
| Assembly (WAV/M4B) | ✅ Validado |
| Tests | 160 passing |

## Known Limitations

- Prosodia de frases muy cortas e interrogativas pendiente de mejora.

## How Kokoro TTS works

This project uses [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
as an external dependency — the TTS inference engine is **not** bundled in this
repository.

| Component | Source | Location |
|-----------|--------|----------|
| `kokoro-onnx` (inference runtime) | PyPI | installed via `pip install` |
| Model files (~337 MB) | GitHub releases | `~/.cache/ebook-audiobook-engine/kokoro/` |
| Backend adapter | This repo | `src/.../tts/kokoro/` |

On first synthesis, the engine downloads two model files (~310 MB + ~27 MB)
from the
[kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases)
and caches them locally. No internet connection is needed after the initial
download.

## Overview

Converts ebooks (EPUB, TXT) into audiobooks (M4B/WAV) through a local
pipeline with no external service dependencies.

**Modes:**

- **Standalone CLI** — convert without a running service
- **Python API** — direct in-process use
- **Local HTTP service** — for Calibre and other external clients (Phase 9)

## Requirements

- Python 3.10+
- FFmpeg (optional, for M4B output with chapters and metadata)

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

## Usage

### CLI

```bash
# List available voices
audiobook-engine list-voices
audiobook-engine list-voices -l es

# Convert EPUB to audiobook
audiobook-engine convert book.epub -v em_alex -o book.m4b

# Convert TXT to audiobook
audiobook-engine convert book.txt -v em_alex -o book.m4b -l es

# Resume interrupted job
audiobook-engine resume <job-id>
```

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
    voice="em_alex",
    output_path=Path("output.m4b"),
)
engine.run(job.id)
```

## Architecture

```
src/audiobook_engine/
├── domain/              # Business logic (no external deps)
│   ├── models.py        # Book, Chapter, TextSegment, Voice
│   ├── job.py           # AudiobookJob state machine
│   ├── protocols.py     # EbookParser, TTSBackend contracts
│   └── exceptions.py    # Domain exceptions
├── application/
│   └── engine.py        # AudiobookEngine orchestrator
├── infrastructure/
│   ├── ebook/
│   │   ├── epub_parser.py
│   │   ├── txt_parser.py
│   │   ├── normalizer.py
│   │   └── chunker.py
│   ├── tts/kokoro/      # Kokoro TTS backend
│   │   ├── backend.py
│   │   └── models.py
│   ├── audio/
│   │   ├── wav_merger.py
│   │   ├── ffmpeg.py
│   │   └── m4b_assembler.py
│   └── persistence/
│       └── job_store.py
└── interfaces/
    └── cli/main.py      # Typer CLI
```

## Pipeline

```
EPUB/TXT → Parse → Normalize → Chunk → TTS → WAV → M4B
                                       ↓
                                 Persistence (job.json)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src tests

# Type check
mypy src

# Format
ruff format src tests
```

## License

MIT
