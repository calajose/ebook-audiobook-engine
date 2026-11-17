"""Phonon 40M TTS backend.

Wraps the official ``ptts`` runtime (Gradium's Rust inference stack with
Python bindings). Models are language-specific checkpoints downloaded on
first use and cached locally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audiobook_engine.domain.exceptions import TTSBackendError
from audiobook_engine.domain.models import (
    BackendCapabilities,
    Language,
    Voice,
)
from audiobook_engine.infrastructure.tts.phonon.models import (
    SUPPORTED_LANGUAGES,
    detect_quant,
    ensure_model_dir,
    list_cached_languages,
    normalize_phonon_language,
)

log = logging.getLogger(__name__)

#: Extra silence appended after each synthesized fragment, in seconds.
_ACOUSTIC_TAIL_SECONDS = 0.08

#: Devices the Phonon runtime accepts.
_SUPPORTED_DEVICES = ("auto", "cpu", "cuda")


class PhononBackend:
    """Phonon 40M TTS backend using the official ``ptts`` runtime.

    Parameters
    ----------
    language:
        Primary language code (``de``, ``en``, ``es``, ``fr``, ``pt``) used
        when listing capabilities. ``None`` means "every language already
        cached locally".
    model_dir:
        Explicit checkpoint directory. ``None`` resolves the per-language
        directory in the user cache, downloading it on first use.
    cache_dir:
        Override for the Phonon cache root (defaults to the user cache).
    device:
        Compute device: ``"auto"`` (default), ``"cpu"`` or ``"cuda"``.
    auto_download:
        When True, a missing checkpoint is downloaded on first use.
    """

    name: str = "phonon"

    def __init__(
        self,
        language: str | None = None,
        model_dir: Path | None = None,
        cache_dir: Path | None = None,
        device: str = "auto",
        auto_download: bool = True,
    ) -> None:
        self._language = normalize_phonon_language(language) if language else None
        self._model_dir = Path(model_dir) if model_dir is not None else None
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._device = device
        self._auto_download = auto_download
        self._tts: Any | None = None
        self._loaded_language: str | None = None
        self._active_device: str | None = None

    @property
    def active_device(self) -> str | None:
        """The device actually used after loading the model."""
        return self._active_device

    @property
    def language(self) -> str | None:
        """Primary language used when listing capabilities."""
        return self._language

    @language.setter
    def language(self, value: str | None) -> None:
        self._language = normalize_phonon_language(value) if value else None

    def _resolve_device(self) -> str | None:
        """Map the configured device to a ``ptts`` device argument."""
        if self._device == "openvino":
            raise TTSBackendError(
                "Phonon 40M does not support the OpenVINO device. "
                "Use --device auto, cpu or cuda."
            )
        if self._device not in _SUPPORTED_DEVICES:
            supported = ", ".join(_SUPPORTED_DEVICES)
            raise TTSBackendError(
                f"Unknown device '{self._device}' for Phonon 40M. "
                f"Supported devices: {supported}"
            )
        return None if self._device == "auto" else self._device

    def _ensure_loaded(self, language: str) -> Any:
        """Lazy-load the checkpoint for ``language`` on first use."""
        lang = normalize_phonon_language(language)
        if self._tts is not None and self._loaded_language == lang:
            return self._tts

        try:
            import ptts
        except ImportError as exc:
            raise TTSBackendError(
                "ptts is not installed. Install with: pip install "
                "'ebook-audiobook-engine[phonon]'"
            ) from exc

        model_dir = self._model_dir
        if model_dir is None:
            model_dir = ensure_model_dir(
                lang, self._cache_dir, auto_download=self._auto_download
            )
        elif not model_dir.is_dir():
            raise TTSBackendError(f"Phonon model directory not found: {model_dir}")

        device = self._resolve_device()
        options: dict[str, Any] = {"lang": lang}
        if device is not None:
            options["device"] = device
        quant = detect_quant(model_dir)
        if quant is not None:
            options["quant"] = quant

        try:
            tts = ptts.TTS(str(model_dir), **options)
        except Exception as exc:
            raise TTSBackendError(
                f"Failed to load the Phonon 40M model for '{lang}' "
                f"from {model_dir}: {exc}"
            ) from exc

        self._tts = tts
        self._loaded_language = lang
        raw_device = getattr(tts, "device", None)
        self._active_device = str(raw_device) if raw_device else None
        log.info(
            "Loaded Phonon 40M model for '%s' on %s",
            lang,
            self._active_device or "unknown",
        )
        return tts

    def _target_languages(self) -> list[str]:
        """Return the languages to report in ``capabilities()``."""
        if self._language is not None:
            return [self._language]
        return list_cached_languages(self._cache_dir)

    @staticmethod
    def _catalog_voice_names(language: str) -> list[str]:
        """Return voice names from ``ptts.catalog_voices()`` for ``language``.

        Falls back to an empty list when offline or when ``ptts`` does not
        expose a voice catalog.
        """
        try:
            import ptts
        except ImportError:
            return []

        catalog_fn = getattr(ptts, "catalog_voices", None)
        if not callable(catalog_fn):
            return []

        try:
            entries = catalog_fn()
        except Exception:
            return []

        names: list[str] = []
        for entry in entries or ():
            if not isinstance(entry, dict):
                continue
            raw_lang = entry.get("language")
            name = entry.get("name")
            if not name or not isinstance(raw_lang, str):
                continue
            norm_lang = raw_lang.strip().lower().replace("_", "-").split("-", 1)[0]
            if norm_lang == language:
                names.append(str(name))
        return names

    def _voices_for_language(self, language: str, tts: Any) -> list[str]:
        """Combine checkpoint voices and catalog voices for ``language``."""
        seen: dict[str, None] = {}
        for raw_name in getattr(tts, "voices", None) or []:
            seen[str(raw_name)] = None
        for cat_name in self._catalog_voice_names(language):
            seen[cat_name] = None
        if not seen:
            return ["default"]
        return list(seen.keys())

    def capabilities(self) -> BackendCapabilities:
        """Return the languages and voices available for Phonon 40M."""
        languages: list[Language] = []
        voices: list[Voice] = []

        for lang in self._target_languages():
            tts = self._ensure_loaded(lang)
            names = self._voices_for_language(lang, tts)
            languages.append(Language(code=lang, name=SUPPORTED_LANGUAGES[lang]))
            voices.extend(
                Voice(id=name, name=name, language_code=lang) for name in names
            )

        return BackendCapabilities(
            languages=tuple(languages),
            voices=tuple(voices),
        )

    def validate_selection(self, language: str, voice: str) -> None:
        """Validate that a language/voice combination is available."""
        lang = normalize_phonon_language(language)
        tts = self._ensure_loaded(lang)
        known = set(self._voices_for_language(lang, tts))
        if voice and voice != "default" and voice not in known:
            available = sorted(known) or ["default"]
            raise TTSBackendError(
                f"Voice '{voice}' is not available for Phonon 40M "
                f"language '{lang}'. Available voices: {available}"
            )

    @staticmethod
    def _apply_speed(tts: Any, speed: float) -> None:
        """Translate the speed multiplier into Phonon's ``duration_delta``."""
        delta = round((1.0 / speed) - 1.0, 4)
        try:
            if delta == 0.0:
                tts.set_conditions({})
            else:
                tts.set_conditions({"duration_delta": delta})
        except Exception as exc:  # pragma: no cover - runtime specific
            raise TTSBackendError(f"Failed to apply the speaking rate: {exc}") from exc

    def synthesize(
        self,
        text: str,
        language: str,
        voice: str,
        output_path: Path,
        speed: float = 1.0,
    ) -> None:
        """Synthesize text to a WAV file.

        Downloads the language checkpoint on first use if not cached and
        applies speed control plus a subtle acoustic tail padding.
        """
        if not text.strip():
            raise TTSBackendError("Cannot synthesize empty text")

        if not (0.5 <= speed <= 2.0):
            raise TTSBackendError(f"Speed must be between 0.5 and 2.0, got {speed}")

        lang = normalize_phonon_language(language)
        tts = self._ensure_loaded(lang)
        self.validate_selection(lang, voice)

        try:
            import numpy as np
            import soundfile as sf  # type: ignore[import-untyped]
        except ImportError as exc:
            raise TTSBackendError(
                "numpy and soundfile are required for Phonon synthesis. "
                "Install with: pip install "
                "'ebook-audiobook-engine[phonon]'"
            ) from exc

        self._apply_speed(tts, speed)

        selected_voice = None if voice in ("", "default") else voice
        try:
            samples = tts.synth(text, voice=selected_voice)
        except Exception as exc:
            raise TTSBackendError(
                f"Synthesis failed for voice='{voice}', lang='{lang}': {exc}"
            ) from exc

        try:
            sample_rate = int(tts.sample_rate)
            pad_samples = int(sample_rate * _ACOUSTIC_TAIL_SECONDS)
            if pad_samples > 0 and len(samples) > 0:
                samples = np.pad(samples, (0, pad_samples), mode="constant")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), samples, sample_rate)
        except TTSBackendError:
            raise
        except Exception as exc:
            raise TTSBackendError(f"Failed to write WAV file: {exc}") from exc
