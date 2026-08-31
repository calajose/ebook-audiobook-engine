# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Chapter Selection (`convert` command)**:
  - Added `--chapters / -c` option to `convert` command, allowing conversion of specific chapters or ranges (e.g., `5-26,29`), skipping others during synthesis and assembly.
  - Automatically filters chapters during segment counting and synthesis.
- **Selective Job Cleanup (`clean` command)**:
  - Added optional `job_id` argument to `audiobook-engine clean` to remove only a specific job instead of the entire working directory.
  - When no `job_id` is provided, behaves as before (removes the entire work directory).

### Fixed
- **Section Separator Noise Removal**:
  - Fixed noise detection pattern to also filter out separators made of em-dashes (`—————`, U+2014) and en-dashes (`———`, U+2013), which some ebooks use as decorative section dividers. Previously only standard hyphens (`-`) were detected, causing TTS to attempt reading these separators.

---

## [0.5.2] - 2026-08-31

### Added
- **Job Listing (`resume` without ID)**:
  - Expanded `resume` command listing to show ALL jobs in the working directory, including completed, failed, and cancelled jobs, not just unfinished ones.
  - Added visual status differentiation in the job list (e.g., bold green for completed, bold red for failed).

### Fixed
- **TTS Text Preview** (`inspect --preview`):
  - Fixed an issue where the preview functionality was not displaying correctly.
- **MOBI HTML Parsing**:
  - Fixed an issue where HTML tags were leaking into the extracted text due to incorrect splitting of HTML content by NCX anchors.
  - Added robust start and end tag detection during chapter splitting to ensure HTML tags are extracted fully and subsequently stripped.
- **MOBI/AZW3 Cover Detection**:
  - Fixed cover extraction for MOBI v7 books that parse via the HTML path by adding OPF metadata parsing to correctly locate cover images.
  - Fixed cover extraction for AZW3/KF8 (EPUB path) by correcting the OPF namespace lookup in `ebooklib` metadata, allowing the parser to find the correct cover ID and manifest item.

---

## [0.5.1] - 2026-08-30

### Added
- **Unfinished Jobs List (`resume` without ID)**:
  - Running `audiobook-engine resume` without a job ID now lists all unfinished/resumable jobs in a Rich table displaying Job ID, Book Title, State, Progress percentage, and Created timestamp.
  - Added `--work-dir / -w` option support to `resume`.
- **Work Directory Cleanup (`clean` command)**:
  - Added `audiobook-engine clean` command to safely clear and remove temporary working directories and job artifacts, with optional `--work-dir / -w`.
- **MOBI V7 Cover Extraction**:
  - Added cover detection and extraction for MOBI v7 / AZW files by parsing `content.opf` metadata (`<meta name="cover">`), manifest items, and directory fallback search.

### Changed
- **CLI Inspect Output**:
  - Updated `audiobook-engine inspect` to display `Cover: None` instead of `Cover: Not detected.` when no cover image is found.

### Fixed
- **XHTML Cover Page Resolution**:
  - Enhanced EPUB and MOBI/AZW3 cover extraction to detect when cover metadata points to an XHTML/HTML container page instead of an image. Automatically parses the markup for `<img>` tags, extracts the underlying image asset, and ensures valid image embedding in final M4B audiobooks.
- **AZW3 / EPUB Cover Detection Fallback**:
  - Fixed issue where certain AZW3 books failed cover detection because their cover image items had non-standard IDs (e.g. `cover-image`) or names without explicit OPF `<meta name="cover">` tags.
  - Added robust case-insensitive substring search for `"cover"` in item IDs/filenames and automatic fallback to the first image item in the book.
- **M4B Narrator / Artist Metadata**:
  - Fixed bug where `job.backend` was not populated during job creation, causing the M4B `artist` metadata tag (`voice - backend`, e.g., `ef_dora - kokoro`) to be omitted.
  - Added robust backend name resolution and fallback to ensure narrator metadata is correctly embedded in generated audiobooks.

---

## [0.5.0] - 2026-08-30

### Added
- **Ebook Analysis Without Audio** (`inspect` command enhanced):
  - `audiobook-engine inspect` now runs the full parse→normalize→chunk pipeline without generating audio.
  - Per-chapter Rich table with: index, title, source XHTML file, characters, words, and chunk count.
  - `--chapters / -c` option to filter by chapter indices (e.g. `0,2,5` or `1-3`).
  - `--max-chars` option to override the default chunk size (500 chars).
  - Automatic warnings for dubious chapter detection: missing headings, single-segment chapters, noise-only content.
- **MOBI/AZW/AZW3 Format Support**:
  - New `MOBIParser` class supporting `.mobi`, `.azw`, and `.azw3` (Kindle) ebook formats.
  - Uses the `mobi` library (KindleUnpack) to unpack files to EPUB (KF8/AZW3) or HTML (MOBI v7/AZW).
  - AZW3/KF8 files: extracts to EPUB and parses with full metadata (title, author, language, cover).
  - MOBI v7/AZW files: extracts to HTML with chapter detection via NCX table-of-contents, heading tags, or single-chapter fallback.
  - MOBI v7/AZW files: reads `content.opf` metadata (language, author, title) instead of hardcoding defaults.
  - DRM-protected files: clear error message ("DRM-protected file not supported").
  - Automatic cleanup of temporary extraction directories.
  - Full `EbookParser` protocol compliance with deterministic output.
- **CLI Updates**:
  - `_get_parser()` factory now dispatches `.mobi`, `.azw`, `.azw3` extensions to `MOBIParser`.
  - Error message updated to list all supported formats.
- **New Dependency**:
  - Added `mobi>=0.4` to project dependencies (GPL-3.0, wraps KindleUnpack).

### Changed
- `pyproject.toml`: Added `mobi>=0.4` to `dependencies`.
- `infrastructure/ebook/__init__.py`: Now re-exports `MOBIParser`.
- `domain/models.py`: `Chapter` dataclass now includes optional `source_file: str | None` field (backward-compatible, defaults to `None`).
- `infrastructure/ebook/epub_parser.py`: Populates `Chapter.source_file` with the XHTML item filename from the EPUB spine.
- `infrastructure/ebook/epub_parser.py`: `_extract_heading()` now checks `<h4>`, `<h5>`, `<h6>` tags in addition to `<h1>`–`<h3>` (Calibre-generated EPUBs often use `<h4>` for chapter titles).
- `infrastructure/ebook/mobi_parser.py`: Populates `Chapter.source_file` with the XHTML item filename for AZW3/KF8 extractions.
- `infrastructure/ebook/mobi_parser.py`: HTML path (MOBI v7) now reads `content.opf` for metadata and `toc.ncx` for chapter splitting, with fallback to heading tags.
- `infrastructure/ebook/mobi_parser.py`: Same heading tag expansion as EPUB for the HTML extraction path.
- `application/engine.py`: New `ChapterAnalysis` and `BookAnalysis` dataclasses; `AudiobookEngine.analyze()` method orchestrates parsing, normalization and chunking without TTS.
- `interfaces/cli/main.py`: `inspect` command rewritten to use `analyze()` with Rich table output, chapter filtering, and warning display. Added `_parse_chapter_indices()` helper for range/spec parsing.

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
- **EPUB Cover Extraction**:
  - Fixed cover detection for EPUBs where ebooklib does not parse the `<meta name="cover">` tag into `book.metadata`.
  - Added direct fallback using `book.get_item_with_id("cover")` for EPUB2 files with non-standard metadata.
  - Verified with Harry Potter EPUB (cover.jpeg detected, 160KB).
- **M4B Metadata Tags (Audio Player Compatibility)**:
  - `artist` now contains the narrator (voice + engine, e.g. "ef_dora - kokoro") instead of the author.
  - `composer` now contains the book author (writer).
  - `album_artist` contains the author for player grouping (Cozy, Apple Books).
  - Tags updated in both `;FFMETADATA1` file and explicit FFmpeg `-metadata` flags.
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
- **M4B Chapter Markers & Metadata**:
  - Replaced broken `-metadata:s:a` chapter approach with proper `;FFMETADATA1` file using `[CHAPTER]` blocks.
  - Chapter markers now recognized by players (Cozy, Apple Books, Audiobookshelf).
  - Global metadata (title, author, language, genre) embedded via `;FFMETADATA1` and `-map_metadata`.
  - Cover image copied without re-encoding (`-c:v copy`).
  - FFmpeg arguments reordered: all inputs first, then output options.

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
