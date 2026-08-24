"""Tests for Kokoro TTS backend (mocked — no real model needed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audiobook_engine.domain.exceptions import TTSBackendError
from audiobook_engine.domain.models import BackendCapabilities
from audiobook_engine.infrastructure.tts.kokoro.backend import (
    KokoroBackend,
    _voice_to_language,
)
from audiobook_engine.infrastructure.tts.kokoro.models import (
    get_model_path,
    get_voices_path,
)


class TestVoiceToLanguage:
    def test_american_english(self) -> None:
        assert _voice_to_language("af_sarah") == "en-us"
        assert _voice_to_language("am_adam") == "en-us"

    def test_british_english(self) -> None:
        assert _voice_to_language("bf_alice") == "en-gb"
        assert _voice_to_language("bm_george") == "en-gb"

    def test_japanese(self) -> None:
        assert _voice_to_language("jf_alpha") == "ja"

    def test_spanish(self) -> None:
        assert _voice_to_language("ef_dora") == "es"
        assert _voice_to_language("em_alex") == "es"

    def test_unknown_voice(self) -> None:
        assert _voice_to_language("xx_unknown") is None


class TestKokoroBackendCapabilities:
    def test_capabilities_with_mock(self) -> None:
        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = [
            "af_sarah",
            "af_nicole",
            "am_adam",
            "bf_alice",
            "ef_dora",
        ]

        backend = KokoroBackend(
            model_path=Path("/fake/model.onnx"),
            voices_path=Path("/fake/voices.bin"),
        )
        backend._kokoro = mock_kokoro

        caps = backend.capabilities()

        assert isinstance(caps, BackendCapabilities)
        assert len(caps.languages) == 3  # en-us, en-gb, es
        assert len(caps.voices) == 5

        lang_codes = {lang.code for lang in caps.languages}
        assert "en-us" in lang_codes
        assert "en-gb" in lang_codes
        assert "es" in lang_codes


class TestKokoroBackendValidateSelection:
    @pytest.fixture()
    def backend_with_caps(self) -> KokoroBackend:
        backend = KokoroBackend(
            model_path=Path("/fake/model.onnx"),
            voices_path=Path("/fake/voices.bin"),
        )
        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = [
            "af_sarah",
            "ef_dora",
        ]
        backend._kokoro = mock_kokoro
        return backend

    def test_valid_selection(
        self, backend_with_caps: KokoroBackend
    ) -> None:
        backend_with_caps.validate_selection("en-us", "af_sarah")

    def test_invalid_voice(
        self, backend_with_caps: KokoroBackend
    ) -> None:
        with pytest.raises(TTSBackendError, match="not available"):
            backend_with_caps.validate_selection("en-us", "zz_fake")

    def test_invalid_language(
        self, backend_with_caps: KokoroBackend
    ) -> None:
        with pytest.raises(TTSBackendError, match="not supported"):
            backend_with_caps.validate_selection("xx", "af_sarah")


class TestKokoroBackendSynthesize:
    def test_synthesize_writes_wav(self, tmp_path: Path) -> None:
        import numpy as np

        mock_samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_sample_rate = 24000
        mock_kokoro = MagicMock()
        mock_kokoro.create.return_value = (
            mock_samples,
            mock_sample_rate,
        )
        mock_kokoro.get_voices.return_value = ["af_sarah"]

        backend = KokoroBackend(
            model_path=Path("/fake/model.onnx"),
            voices_path=Path("/fake/voices.bin"),
        )
        backend._kokoro = mock_kokoro

        output = tmp_path / "output.wav"

        mock_sf = MagicMock()
        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            backend.synthesize(
                "Hello world", "en-us", "af_sarah", output, speed=0.95
            )

        mock_kokoro.create.assert_called_once_with(
            "Hello world",
            voice="af_sarah",
            speed=0.95,
            lang="en-us",
        )
        mock_sf.write.assert_called_once()

    def test_synthesize_invalid_speed_raises(self) -> None:
        backend = KokoroBackend(
            model_path=Path("/fake/model.onnx"),
            voices_path=Path("/fake/voices.bin"),
        )
        with pytest.raises(TTSBackendError, match="Speed must be between"):
            backend.synthesize(
                "Hello", "en-us", "af_sarah", Path("/tmp/o.wav"), speed=3.0
            )

    def test_synthesize_empty_text_raises(self) -> None:
        backend = KokoroBackend(
            model_path=Path("/fake/model.onnx"),
            voices_path=Path("/fake/voices.bin"),
        )
        with pytest.raises(TTSBackendError, match="empty text"):
            backend.synthesize(
                "", "en-us", "af_sarah", Path("/tmp/o.wav")
            )


class TestModelDownload:
    def test_model_path_cache_hit(self, tmp_path: Path) -> None:
        model = tmp_path / "kokoro-v1.0.onnx"
        model.write_bytes(b"fake model data")
        result = get_model_path(tmp_path)
        assert result == model

    def test_voices_path_cache_hit(self, tmp_path: Path) -> None:
        voices = tmp_path / "voices-v1.0.bin"
        voices.write_bytes(b"fake voices data")
        result = get_voices_path(tmp_path)
        assert result == voices

    @patch("audiobook_engine.infrastructure.tts.kokoro.models._create_session")
    def test_download_ssl_error_raises_runtime_error(
        self, mock_create: MagicMock, tmp_path: Path
    ) -> None:
        import requests

        mock_session = MagicMock()
        mock_create.return_value = mock_session
        mock_session.get.side_effect = requests.exceptions.SSLError(
            "SSL error"
        )

        from audiobook_engine.infrastructure.tts.kokoro.models import (
            _download_file,
        )

        with pytest.raises(RuntimeError, match="SSL error"):
            _download_file(
                "https://example.com/model.onnx",
                tmp_path / "model.onnx",
                1000,
            )

    @patch("audiobook_engine.infrastructure.tts.kokoro.models._create_session")
    def test_download_connection_error_raises_runtime_error(
        self, mock_create: MagicMock, tmp_path: Path
    ) -> None:
        import requests

        mock_session = MagicMock()
        mock_create.return_value = mock_session
        mock_session.get.side_effect = requests.exceptions.ConnectionError(
            "Connection failed"
        )

        from audiobook_engine.infrastructure.tts.kokoro.models import (
            _download_file,
        )

        with pytest.raises(RuntimeError, match="Connection error"):
            _download_file(
                "https://example.com/model.onnx",
                tmp_path / "model.onnx",
                1000,
            )
