"""Kokoro TTS backend — first concrete TTS implementation.

Uses kokoro-onnx for ONNX-based inference. Models are downloaded
from GitHub releases on first use and cached locally.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from audiobook_engine.domain.exceptions import TTSBackendError
from audiobook_engine.domain.models import (
    BackendCapabilities,
    Language,
    Voice,
)

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

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
    in the user cache directory.  They are never part of the repository.

    Parameters
    ----------
    model_path:
        Path to the ONNX model file.  ``None`` triggers auto-download.
    voices_path:
        Path to the voices binary.  ``None`` triggers auto-download.
    device:
        Compute device selection: ``"auto"`` (default), ``"cpu"``,
        ``"cuda"``, or ``"openvino"``.  In ``"auto"`` mode the best
        available device is chosen silently (CUDA > OpenVINO GPU > CPU).
        ``"cuda"`` forces NVIDIA GPU and raises an error if unavailable.
        ``"openvino"`` forces Intel GPU and raises an error if unavailable.
    """

    name: str = "kokoro"

    def __init__(
        self,
        model_path: Path | None = None,
        voices_path: Path | None = None,
        device: str = "auto",
    ) -> None:
        self._model_path = model_path
        self._voices_path = voices_path
        self._device = device
        self._kokoro: object | None = None
        self._active_device: str | None = None

    @property
    def active_device(self) -> str | None:
        """The device name that was actually used after loading.

        ``None`` before the first call to ``_ensure_loaded()``.
        """
        return self._active_device

    def _ensure_loaded(self) -> object:
        """Lazy-load the kokoro model on first use."""
        if self._kokoro is not None:
            return self._kokoro

        try:
            from kokoro_onnx import Kokoro
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
            self._kokoro, self._active_device = self._create_kokoro(
                Kokoro, str(model_path), str(voices_path)
            )
        except TTSBackendError:
            raise
        except Exception as exc:
            raise TTSBackendError(
                f"Failed to load Kokoro model: {exc}"
            ) from exc

        return self._kokoro

    def _create_kokoro_cpu(
        self,
        kokoro_cls: type[Any],
        model_path: str,
        voices_path: str,
    ) -> tuple[object, str]:
        """Create a Kokoro instance on CPU (fallback path).

        Creates an explicit CPU-only ORT session so kokoro-onnx does not
        try to auto-detect and use OpenVINO/CUDA internally.
        """
        import onnxruntime as ort

        log.info("Loading Kokoro on CPU (fallback)")
        so = ort.SessionOptions()
        session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        kokoro = kokoro_cls.from_session(session, voices_path)
        return (kokoro, "cpu")

    def _create_kokoro(
        self,
        kokoro_cls: type[Any],
        model_path: str,
        voices_path: str,
    ) -> tuple[object, str]:
        """Create a Kokoro instance, optionally with a specific EP.

        Returns ``(kokoro_instance, device_used)``.
        """
        from audiobook_engine.infrastructure.device import (
            get_provider_config,
            resolve_device,
            verify_session_provider,
        )

        resolved = resolve_device(self._device)

        # Plain CPU — use kokoro-onnx default path (no EP override)
        if resolved.provider_name is None:
            log.info("Loading Kokoro on CPU")
            return (kokoro_cls(model_path, voices_path), "cpu")

        # GPU / accelerated backend — create ORT session ourselves
        providers, disable_fallback = get_provider_config(self._device)

        try:
            import onnxruntime as ort

            so = ort.SessionOptions()
            if disable_fallback:
                # Prevent ORT from silently falling back to CPU when the
                # user explicitly requested a specific backend.
                session = ort.InferenceSession(
                    model_path,
                    sess_options=so,
                    providers=providers,
                    enable_fallback=0,
                )
            else:
                session = ort.InferenceSession(
                    model_path,
                    sess_options=so,
                    providers=providers,
                )
        except Exception as exc:
            # In auto mode, the selected EP may fail (e.g. OpenVINO
            # cannot handle Kokoro's dynamic-shape operators).  Fall
            # back to CPU instead of crashing.
            if not disable_fallback:
                log.warning(
                    "Device '%s' failed (%s), falling back to CPU",
                    resolved.display_name,
                    exc,
                )
                return self._create_kokoro_cpu(
                    kokoro_cls, model_path, voices_path
                )

            msg = str(exc)
            if "dynamic" in msg.lower() or "SequenceAt" in msg:
                hint = (
                    "The Kokoro ONNX model uses operators "
                    "(SplitToSequence, SequenceAt) that OpenVINO "
                    "cannot convert. This is a known model "
                    "compatibility limitation.\n"
                    "Use --device cpu instead."
                )
            else:
                hint = ""
            error_text = (
                f"Failed to create ONNX Runtime session with "
                f"{resolved.provider_name}: {exc}"
            )
            if hint:
                error_text += f"\n\n{hint}"
            raise TTSBackendError(error_text) from exc

        # Verify the EP is actually active (not just registered)
        error = verify_session_provider(
            session,
            resolved.provider_name,
            resolved.gpu_device_id,
        )
        if error:
            # In auto mode, EP verification failure also falls back to CPU
            if not disable_fallback:
                log.warning(
                    "Device '%s' EP verification failed: %s — "
                    "falling back to CPU",
                    resolved.display_name,
                    error,
                )
                return self._create_kokoro_cpu(
                    kokoro_cls, model_path, voices_path
                )
            raise TTSBackendError(error)

        log.info(
            "Loading Kokoro on %s (EP: %s, options: %s)",
            resolved.display_name,
            resolved.provider_name,
            resolved.provider_options,
        )

        kokoro = kokoro_cls.from_session(session, voices_path)
        return (kokoro, resolved.name)

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
        speed: float = 1.0,
    ) -> None:
        """Synthesize text to a WAV file.

        Downloads the model on first use if not cached.
        Applies speed control and subtle acoustic tail padding.
        """
        if not text.strip():
            raise TTSBackendError("Cannot synthesize empty text")

        if not (0.5 <= speed <= 2.0):
            raise TTSBackendError(
                f"Speed must be between 0.5 and 2.0, got {speed}"
            )

        kokoro = self._ensure_loaded()
        self.validate_selection(language, voice)

        try:
            samples, sample_rate = kokoro.create(  # type: ignore[attr-defined]
                text,
                voice=voice,
                speed=speed,
                lang=language,
            )
        except Exception as exc:
            raise TTSBackendError(
                f"Synthesis failed for voice='{voice}', "
                f"lang='{language}': {exc}"
            ) from exc

        try:
            import numpy as np
            import soundfile as sf  # type: ignore[import-untyped]

            # Acoustic tail protection for questions and short phrases
            # (prevents abrupt cutoff from aggressive silence trimming)
            pad_samples = int(sample_rate * 0.08)
            if pad_samples > 0 and len(samples) > 0:
                samples = np.pad(samples, (0, pad_samples), mode="constant")

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
