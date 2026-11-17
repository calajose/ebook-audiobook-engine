"""Phonon 40M TTS backend integration.

Phonon 40M is Gradium's family of open-weight on-device text-to-speech
models. Each language ships its own checkpoint, downloaded on first use
from Hugging Face and cached locally.
"""

from __future__ import annotations

from audiobook_engine.infrastructure.tts.phonon.backend import PhononBackend
from audiobook_engine.infrastructure.tts.phonon.models import (
    SUPPORTED_LANGUAGES,
    normalize_phonon_language,
)

__all__ = [
    "SUPPORTED_LANGUAGES",
    "PhononBackend",
    "normalize_phonon_language",
]
