# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] - 2026-08-30

### Added
- **CLI Commands**:
  - `audiobook-engine capabilities`: Shows supported languages and voice counts.
  - `audiobook-engine inspect`: Analyzes an ebook and shows its structure (title, author, language, chapters, segments).
  - `audiobook-engine status`: Shows the status and progress of a job.
  - `audiobook-engine cancel`: Cancels a running job.
  - `audiobook-engine reassemble`: Re-assembles a completed job without re-synthesizing. Useful for testing different output formats or re-generating output after assembly issues.

### Fixed
- **Resume Progress Reporting**:
  - Fixed issue where `audiobook-engine resume` did not display progress bars or segment counters.
  - Refactored `resume` command to run in a background thread and hook into the existing CLI `_display_progress` UI, matching behavior of the `convert` command.
- **M4B Assembly File Descriptor Limit**:
  - Replaced FFmpeg concat filter (N simultaneous inputs) with concat demuxer (sequential file processing).
  - Resolves "Too many open files" error when assembling audiobooks with 1000+ segments.
  - Temporary filelist is automatically cleaned up after assembly.
- **Resume Assembly Idempotency**:
  - Fixed bug where resuming a job in ASSEMBLING state would skip assembly and mark as COMPLETED.
  - Assembly phase now retries correctly when job state is ASSEMBLING.
- **M4B Assembly Relative Path Resolution**:
  - Fixed "No such file or directory" error in concat demuxer by using absolute paths in the filelist.
  - FFmpeg's concat demuxer resolves relative paths relative to the filelist location, not CWD.

---

## [0.2.0] - 2026-08-29

### Added
- **Automatic Intermediate File Cleanup**:
  - Intermediate WAV files (`segments/`, `chapters/`, `output/`) are now cleaned up automatically after successful M4B/WAV assembly.
  - New `--keep-intermediates` CLI flag for `convert` and `resume` commands to preserve intermediates for debugging.
  - `keep_intermediates` field added to `AudiobookJob` domain model and serialized in `job.json` (backward-compatible).
- **Kokoro Model Update**:
  - Upgraded model download URLs to `model-files-v1.1` release channel.
  - Model sizes updated: ONNX ~326 MB, voices ~28 MB.

### Changed
- Python minimum version requirement raised to 3.13+.
- `AudiobookEngine._cleanup_intermediates()` removes intermediate directories after assembly unless `keep_intermediates=True`.

---

## [0.2.0-alpha] - 2026-08-23

### Added
- **Configurable Prosody & Audio Pauses (`ProsodyConfig`)**:
  - Configurable paragraph pause duration (default: `700ms`).
  - Configurable chapter pause duration (default: `2500ms`).
  - Configurable scene break pause duration (default: `1500ms`).
  - Configurable chapter title pause duration (default: `1200ms`).
- **Audio Silence Infrastructure (`silence.py`)**:
  - `generate_silence_wav`: Generates silent PCM WAV audio matching sample rate, channels, and bit depth.
  - `append_silence_to_wav`: Appends exact duration of silence to audio segments without re-encoding.
  - `create_silence_matching`: Creates silence files tailored to reference WAV parameters.
  - `merge_wav_files(..., silence_between_ms=...)`: Merges WAV files with optional inter-file silence.
- **Dialogue & Short Phrasing Improvements**:
  - `normalize_dialogue` in `normalizer.py`: Normalizes dialogue dashes (`- `, `– ` to em-dash `— `), guillemets (`« »`), curly quotes, and spacing for inverted question/exclamation marks (`¿`, `¡`).
  - Dialogue tag preservation in `chunker.py`: Prevents breaking questions/exclamations away from trailing dialogue tags (e.g. *«—¿Adónde vas? —preguntó Juan.»*).
  - Spanish abbreviation support in sentence splitter (`Sr.`, `Sra.`, `D.`, `Dña.`, `pág.`, `art.`, etc.).
- **TTS Synthesis Speed & Acoustic Tail Protection**:
  - Added `speed` parameter support (`0.5x` to `2.0x`) in `KokoroBackend.synthesize` and domain `TTSBackend` protocol.
  - Added acoustic micro-padding to synthesized audio frames to prevent abrupt speech cutoff on short phrases and questions from silence trimming.
- **CLI Options**:
  - Added `--speed / -s` option to `convert` command.
  - Added `--paragraph-pause` option in milliseconds.
  - Added `--chapter-pause` option in milliseconds.
  - Added `--scene-break-pause` option in milliseconds.
- **Comprehensive Test Suite**:
  - Added test suites for silence generation, WAV merging with silence, dialogue normalization, sentence splitting with dialogue tags, speed controls, and CLI prosody flags (total test count expanded to 192).
- **Agent Guidelines**:
  - Added `AGENTS.md` for AI agents and developers to guide ongoing architecture, testing, and implementation.

### Changed
- `AudiobookJob` serialization and deserialization now persist prosody parameters with backward compatibility.
- `AudiobookEngine._synthesize` automatically injects paragraph pauses and chapter boundary pauses into segment WAVs.
- `m4b_assembler.py` metadata timestamps naturally include silence intervals, maintaining accurate chapter markers.

---

## [0.1.0] - 2026-08-23

### Added
- Initial project release with Clean Architecture (Domain, Application, Infrastructure, Interfaces).
- EPUB parser (`EPUBParser`) and Plain Text parser (`TXTParser`).
- Kokoro ONNX TTS backend (`KokoroBackend`) with automated model caching and voice detection.
- Text normalization (`normalizer.py`) and sentence-aware chunking (`chunker.py`).
- Audio pipeline with WAV merger and FFmpeg M4B assembler with chapter markers and cover art embedding.
- Job persistence (`job_store.py`) enabling crash recovery and resumption (`audiobook-engine resume <job-id>`).
- CLI interface using Typer and Rich with progress bars and voice listing.
