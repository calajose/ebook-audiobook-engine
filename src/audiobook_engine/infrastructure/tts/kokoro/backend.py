"""Kokoro TTS backend — first concrete TTS implementation.

Uses kokoro-onnx for ONNX-based inference. Models are downloaded
from GitHub releases on first use and cached locally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from audiobook_engine.domain.exceptions import TTSBackendError
from audiobook_engine.domain.models import (
    BackendCapabilities,
    Language,
    Voice,
)

if TYPE_CHECKING:
    from pathlib import Path

# Mapping from kokoro voice prefix to language code
_VOICE_PREFIX_TO_LANG: dict[str, str] = {
    "a": "en-us",
    "b": "en-gb",
    "j": "ja",
    "z": "zh",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}

# Language codes to human-readable names
_LANG_NAMES: dict[str, str] = {
    "en-us": "American English",
    "en-gb": "British English",
    "ja": "Japanese",
    "zh": "Mandarin Chinese",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "pt-br": "Brazilian Portuguese",
}


def _voice_to_language(voice_id: str) -> str | None:
    """Extract language code from a voice ID (e.g. 'af_sarah' -> 'en-us')."""
    prefix = voice_id.split("_")[0][:1] if "_" in voice_id else ""
    return _VOICE_PREFIX_TO_LANG.get(prefix)


class KokoroBackend:
    """Kokoro TTS backend using kokoro-onnx for inference.

    Models are downloaded from GitHub releases on first use and stored
    in the user cache directory. They are never part of the repository.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        voices_path: Path | None = None,
    ) -> None:
        self._model_path = model_path
        self._voices_path = voices_path
        self._kokoro: object | None = None

    def _ensure_loaded(self) -> object:
        """Lazy-load the kokoro model on first use."""
        if self._kokoro is not None:
            return self._kokoro

        try:
            from kokoro_onnx import Kokoro  # type: ignore[import-untyped]
        except ImportError as exc:
            raise TTSBackendError(
                "kokoro-onnx is not installed. "
                "Install with: pip install "
                "'ebook-audiobook-engine[kokoro]'"
            ) from exc

        model_path = self._model_path
        voices_path = self._voices_path

        if model_path is None or voices_path is None:
            from audiobook_engine.infrastructure.tts.kokoro.models import (
                get_model_path,
                get_voices_path,
            )

            if model_path is None:
                model_path = get_model_path()
            if voices_path is None:
                voices_path = get_voices_path()

        try:
            self._kokoro = Kokoro(
                str(model_path), str(voices_path)
            )
        except Exception as exc:
            raise TTSBackendError(
                f"Failed to load Kokoro model: {exc}"
            ) from exc

        return self._kokoro

    def capabilities(self) -> BackendCapabilities:
        """Return available languages and voices.

        On first call this loads the model (downloading if needed)
        to discover available voices.
        """
        kokoro = self._ensure_loaded()

        try:
            voice_names = kokoro.get_voices()  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise TTSBackendError(
                "Cannot retrieve voice list from kokoro-onnx"
            ) from exc

        voices: list[Voice] = []
        lang_codes: set[str] = set()

        for voice_id in voice_names:
            lang_code = _voice_to_language(voice_id)
            if lang_code:
                lang_codes.add(lang_code)
                voices.append(
                    Voice(
                        id=voice_id,
                        name=voice_id,
                        language_code=lang_code,
                    )
                )

        languages = tuple(
            Language(code=code, name=_LANG_NAMES.get(code, code))
            for code in sorted(lang_codes)
        )

        return BackendCapabilities(
            languages=languages,
            voices=tuple(voices),
        )

    def validate_selection(
        self, language: str, voice: str
    ) -> None:
        """Validate that a language/voice combination is available."""
        caps = self.capabilities()

        voice_lang = _voice_to_language(voice)
        if voice_lang is None:
            raise TTSBackendError(f"Unknown voice format: {voice}")

        available_voice_ids = {v.id for v in caps.voices}
        if voice not in available_voice_ids:
            raise TTSBackendError(
                f"Voice '{voice}' is not available. "
                f"Available voices: {sorted(available_voice_ids)}"
            )

        available_lang_codes = {
            lang.code for lang in caps.languages
        }
        if language not in available_lang_codes:
            raise TTSBackendError(
                f"Language '{language}' is not supported. "
                f"Supported languages: {sorted(available_lang_codes)}"
            )

    def synthesize(
        self,
        text: str,
        language: str,
        voice: str,
        output_path: Path,
    ) -> None:
        """Synthesize text to a WAV file.

        Downloads the model on first use if not cached.
        """
        if not text.strip():
            raise TTSBackendError("Cannot synthesize empty text")

        kokoro = self._ensure_loaded()
        self.validate_selection(language, voice)

        try:
            samples, sample_rate = kokoro.create(  # type: ignore[attr-defined]
                text,
                voice=voice,
                speed=1.0,
                lang=language,
            )
        except Exception as exc:
            raise TTSBackendError(
                f"Synthesis failed for voice='{voice}', "
                f"lang='{language}': {exc}"
            ) from exc

        try:
            import soundfile as sf  # type: ignore[import-untyped]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), samples, sample_rate)
        except ImportError as exc:
            raise TTSBackendError(
                "soundfile is not installed. "
                "Install with: pip install "
                "'ebook-audiobook-engine[kokoro]'"
            ) from exc
        except Exception as exc:
            raise TTSBackendError(
                f"Failed to write WAV file: {exc}"
            ) from exc
