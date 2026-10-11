# ebook-audiobook-engine

> **Version: v0.6.0** — Local audiobook conversion engine with configurable prosody, smart dialogue chunking, GPU acceleration via NVIDIA CUDA and Intel OpenVINO, and pluggable TTS backends.

Developed in [OpenCode](https://opencode.ai) through human-guided orchestration of Mimo V2.5, Gemini 3.5 Flash Lite, Gemini 3.7 Flash, Gemini 3.1 Pro, and minor assistance from auxiliary models.

## Current State

| Feature | Status | Notes |
|---|---|---|
| **EPUB → M4B** | ✅ Funcional | Chapters, metadata, cover image embedding via FFmpeg |
| **TXT → M4B** | ✅ Funcional | Paragraph-aware parsing, auto fallback to WAV merger |
| **MOBI/AZW/AZW3 → M4B** | ✅ Funcional | Kindle ebook support via KindleUnpack (DRM-free only) |
| **Kokoro TTS Backend** | ✅ Funcional | Multilingual ONNX local inference with auto model download |
| **GPU Acceleration** | ✅ Funcional | NVIDIA CUDA and Intel GPU via ONNX Runtime EPs, auto device detection, CPU fallback |
| **Prosody & Pauses** | ✅ Funcional | Configurable paragraph (700ms), chapter (2500ms), and scene break (1500ms) pauses |
| **Dialogue & Interrogatives** | ✅ Mejorado | Natural phrasing: dialog tags kept with questions, em-dash normalization, acoustic tail padding |
| **Chunking** | ✅ Determinado | Max 500 chars, sentence-boundary aware, abbreviation & Spanish punctuation handling |
| **Assembly (WAV/M4B)** | ✅ Validado | Bit-perfect PCM pause insertion, FFmpeg concat demuxer for large audiobooks |
| **Tests** | 304 passing | 100% test pass rate with strict typing and linting |

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

Converts ebooks (EPUB, TXT, MOBI, AZW, AZW3) into audiobooks (M4B/WAV) through a local pipeline with no external cloud service dependencies.

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
pip install -e ".[openvino]"    # Intel GPU acceleration (optional)
pip install -e ".[cuda]"        # NVIDIA GPU acceleration (optional)
pip install -e ".[api]"         # HTTP service
pip install -e ".[dev]"         # Testing/linting
```

For Intel GPU acceleration:

```bash
pip install -e ".[kokoro,openvino]"
```

For NVIDIA GPU acceleration:

```bash
pip install -e ".[kokoro,cuda]"
```

### Activating the virtual environment

If you created a virtual environment (or the project uses one via `uv`), activate it before running commands:

```bash
# bash / zsh
source .venv/bin/activate

# fish
source .venv/bin/activate.fish

# fish (alternative)
. .venv/bin/activate.fish

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

If you use [uv](https://docs.astral.sh/uv/), you can run commands directly without activating:

```bash
uv run audiobook-engine convert book.epub -v ef_dora -o book.m4b
uv run pytest
```

---

## Usage

### CLI

```bash
# List available voices
audiobook-engine list-voices
audiobook-engine list-voices -l es

# Check capabilities
audiobook-engine capabilities

# Inspect an ebook
audiobook-engine inspect book.epub

# Preview the exact TTS text for chapter 3
audiobook-engine inspect book.epub --preview 3

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

# Check job status
 audiobook-engine status <job-id>

# List unfinished jobs
audiobook-engine resume

# Resume interrupted job
audiobook-engine resume <job-id>

# Clean working directory
audiobook-engine clean

# Clean only a specific job
audiobook-engine clean <job-id>

# Re-assemble completed job (without re-synthesizing)
audiobook-engine reassemble <job-id>
audiobook-engine reassemble <job-id> -o new_output.m4b

# Cancel a job
audiobook-engine cancel <job-id>

# Convert plain text ebook
audiobook-engine convert book.txt -v em_alex -o book.m4b -l es

# Convert Kindle ebook (MOBI/AZW/AZW3)
audiobook-engine convert book.azw3 -v ef_dora -o book.m4b -l es

# Convert keeping intermediate WAV files (for debugging)
audiobook-engine convert book.epub -v ef_dora -o book.m4b --keep-intermediates

# Force CPU inference
audiobook-engine convert book.epub -v ef_dora -o book.m4b --device cpu

# Force NVIDIA CUDA GPU — errors if not available
audiobook-engine convert book.epub -v ef_dora -o book.m4b --device cuda

# Force OpenVINO GPU (Intel) — errors if not available
audiobook-engine convert book.epub -v ef_dora -o book.m4b --device openvino
```

### CLI Options for `convert`

| Option | Flag | Default | Description |
|---|---|---|---|
| `--output` | `-o` | `output.m4b` | Path to output file (`.m4b` or `.wav`) |
| `--language` | `-l` | `en` | Target language code (`es`, `en-us`, `en-gb`, `ja`, etc.) |
| `--voice` | `-v` | *(Required)* | Voice identifier (e.g. `ef_dora`, `em_alex`, `af_sarah`) |
| `--speed` | `-s` | `1.0` | Narration speed multiplier (`0.5` to `2.0`) |
| `--device` | `-d` | `auto` | Compute device: `auto`, `cpu`, `cuda`, `openvino` |
| `--paragraph-pause` | | `700` | Pause duration between paragraphs in milliseconds |
| `--chapter-pause` | | `2500` | Pause duration between chapters in milliseconds |
| `--scene-break-pause` | | `1500` | Pause duration for scene breaks in milliseconds |
| `--chapters` | `-c` | None | Comma-separated chapter indices to convert (e.g. '0,2,5' or '1-3') |
| `--work-dir` | `-w` | `work/` | Temporary directory for job artifacts and cache |
| `--keep-intermediates` | | `False` | Keep intermediate WAV files after conversion (cleaned by default) |
| `--verbose` | `-V` | `False` | Enable debug logging |

---

### GPU Acceleration (NVIDIA CUDA / Intel OpenVINO)

ebook-audiobook-engine supports GPU acceleration through ONNX Runtime execution providers for both NVIDIA CUDA and Intel OpenVINO. The `--device auto` mode automatically selects the best available device (CUDA > OpenVINO > CPU).

**NVIDIA CUDA:**

```bash
# Install with CUDA support
pip install -e ".[kokoro,cuda]"

# Auto-detect (prefers CUDA if available)
audiobook-engine convert book.epub -v af_sarah -o book.m4b

# Force CUDA — fails with clear error if not available
audiobook-engine convert book.epub -v af_sarah -o book.m4b --device cuda
```

Requirements:
- NVIDIA GPU with CUDA-capable drivers
- `onnxruntime-gpu` package (installed via `[cuda]` extra)

**Intel OpenVINO:**

```bash
# Install with OpenVINO support
pip install -e ".[kokoro,openvino]"

# Force OpenVINO — fails with clear error if not available
audiobook-engine convert book.epub -v af_sarah -o book.m4b --device openvino
```

Requirements:
- Intel CPU with integrated GPU (e.g. Iris Xe) or discrete Intel GPU
- `onnxruntime-openvino` package (installed via `[openvino]` extra)
- On Linux: Intel Compute Runtime and GPU drivers

> **Note:** `onnxruntime-gpu` and `onnxruntime-openvino` are mutually exclusive distributions. Only install one `[cuda]` or `[openvino]` extra per environment.

**Diagnostics:**

```bash
# Show detected hardware and providers
python scripts/detect_device.py

# Benchmark specific device
python scripts/benchmark_tts.py --device cuda
python scripts/benchmark_tts.py --device openvino
python scripts/benchmark_tts.py --all
```

---

### Python API

```python
from pathlib import Path
from audiobook_engine import AudiobookEngine
from audiobook_engine.infrastructure.ebook.epub_parser import EPUBParser
from audiobook_engine.infrastructure.ebook.mobi_parser import MOBIParser
from audiobook_engine.infrastructure.tts.kokoro.backend import KokoroBackend

# Parse EPUB
parser = EPUBParser()
tts = KokoroBackend(device="auto")  # or "cpu", "cuda", "openvino"
engine = AudiobookEngine(parser, tts, work_dir=Path("work"))

book = engine.inspect(Path("book.epub"))
print(f"Title: {book.title}, Chapters: {len(book.chapters)}")

# Or parse Kindle ebook (MOBI/AZW/AZW3)
mobi_parser = MOBIParser()
engine_mobi = AudiobookEngine(mobi_parser, tts, work_dir=Path("work"))

book = engine_mobi.inspect(Path("book.azw3"))
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
│   │   ├── mobi_parser.py   # MOBI/AZW/AZW3 (Kindle) format support
│   │   ├── txt_parser.py
│   │   ├── normalizer.py   # Dialogue dashes, quotes, whitespace cleaning
│   │   └── chunker.py      # Sentence & dialogue-aware text chunking
│   ├── tts/kokoro/         # Kokoro ONNX TTS backend adapter
│   │   ├── backend.py      # Synthesis with speed, tail protection & device selection
│   │   └── models.py       # Model cache & automated downloader
│   ├── device.py           # Hardware detection, CUDA/OpenVINO/CPU provider resolution
│   ├── audio/
│   │   ├── silence.py      # WAV silence generation & pause injection
│   │   ├── wav_merger.py   # Raw PCM WAV concatenator with silence support
│   │   ├── ffmpeg.py       # FFmpeg subprocess wrapper
│   │   └── m4b_assembler.py # M4B builder with accurate chapter marks
│   └── persistence/
│       └── job_store.py    # JSON job persistence for crash recovery
└── interfaces/
    └── cli/main.py         # Typer CLI with rich progress, prosody & device flags
```

---

## Pipeline

```
EPUB/TXT/MOBI/AZW/AZW3 → Parse → Normalize → Chunk → TTS (Kokoro) → Silence Injection → WAV → M4B
                                                  ↓
                                        Device Selection (CPU/GPU)
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

The code in this repository is licensed under the MIT License (see `LICENSE`).

The design specifications are maintained in a private repository; they are not
distributed with this project and are excluded from the scope of the MIT
license. All rights reserved on that content.
