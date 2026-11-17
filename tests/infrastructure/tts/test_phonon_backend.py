"""Tests for the Phonon 40M TTS backend and its model cache."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import requests

from audiobook_engine.domain.exceptions import TTSBackendError
from audiobook_engine.infrastructure.tts.phonon import models
from audiobook_engine.infrastructure.tts.phonon.backend import PhononBackend
from audiobook_engine.infrastructure.tts.phonon.models import (
    detect_quant,
    ensure_model_dir,
    get_language_model_dir,
    get_phonon_cache_dir,
    list_cached_languages,
    normalize_phonon_language,
    try_normalize_phonon_language,
)

if TYPE_CHECKING:
    from pathlib import Path

SAMPLE_RATE = 24_000


def _make_checkpoint(root: Path, lang: str = "es") -> Path:
    """Create a complete fake checkpoint directory for ``lang``."""
    model_dir = root / f"phonon-40m-{lang}"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "model.q8.gguf").write_bytes(b"\x00" * 32)
    (model_dir / "default-voice.safetensors").write_bytes(b"\x00" * 8)
    return model_dir


class _FakeTTS:
    """Stand-in for ``ptts.TTS`` recording how it was constructed."""

    instances: list[_FakeTTS] = []

    def __init__(
        self,
        model: str,
        device: str | None = None,
        quant: str | None = None,
        *,
        lang: str,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.lang = lang
        self.device = device or "cpu"
        self.quant = quant
        self.sample_rate = SAMPLE_RATE
        self.voices = ["default", "Celia"]
        self.conditions: dict[str, Any] = {}
        self.last_synth: tuple[str, str | None] | None = None
        self.extra = kwargs
        _FakeTTS.instances.append(self)

    def set_conditions(self, conditions: dict[str, Any]) -> None:
        self.conditions = dict(conditions)

    def synth(self, text: str, voice: str | None = None) -> Any:
        self.last_synth = (text, voice)
        return np.zeros(SAMPLE_RATE // 10, dtype="float32")


@pytest.fixture
def fake_ptts(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTTS]:
    """Install a fake ``ptts`` module so no runtime is needed."""
    module = types.ModuleType("ptts")
    module.TTS = _FakeTTS  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ptts", module)
    _FakeTTS.instances = []
    return _FakeTTS


class TestLanguageNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("es", "es"),
            ("es-ES", "es"),
            ("es_es", "es"),
            ("en-US", "en"),
            ("en-gb", "en"),
            ("pt-BR", "pt"),
            ("FR", "fr"),
            ("de-DE", "de"),
        ],
    )
    def test_regional_tags_fold_to_base_language(self, raw: str, expected: str) -> None:
        assert normalize_phonon_language(raw) == expected

    @pytest.mark.parametrize("raw", ["ja", "zh", "xx", ""])
    def test_unsupported_language_raises(self, raw: str) -> None:
        with pytest.raises(TTSBackendError) as excinfo:
            normalize_phonon_language(raw)
        assert "de, en, es, fr, pt" in str(excinfo.value)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("es-ES", "es"), ("ja", None), (None, None), ("", None)],
    )
    def test_try_normalize_never_raises(
        self, raw: str | None, expected: str | None
    ) -> None:
        assert try_normalize_phonon_language(raw) == expected


class TestCachePaths:
    def test_cache_root_is_under_phonon(self, tmp_path: Path) -> None:
        assert get_phonon_cache_dir(tmp_path) == tmp_path

    def test_language_model_dir_shape(self, tmp_path: Path) -> None:
        assert get_language_model_dir("pt-BR", tmp_path) == tmp_path / "phonon-40m-pt"

    def test_detect_quant_prefers_q8(self, tmp_path: Path) -> None:
        model_dir = _make_checkpoint(tmp_path)
        assert detect_quant(model_dir) == "q8"

    def test_detect_quant_safetensors_returns_none(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "phonon-40m-en"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"\x00")
        assert detect_quant(model_dir) is None

    def test_list_cached_languages_skips_incomplete(self, tmp_path: Path) -> None:
        _make_checkpoint(tmp_path, "es")
        (tmp_path / "phonon-40m-en").mkdir()
        assert list_cached_languages(tmp_path) == ["es"]

    def test_list_cached_languages_empty_root(self, tmp_path: Path) -> None:
        assert list_cached_languages(tmp_path / "missing") == []


class _FakeResponse:
    def __init__(self, status_code: int = 200, body: bytes = b"payload") -> None:
        self.status_code = status_code
        self._body = body
        self.headers: dict[str, str] = {}

    def iter_content(self, chunk_size: int = 8192) -> Any:
        yield self._body


class _FakeSession:
    """Fake requests session keyed by URL."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(
        self,
        url: str,
        stream: bool = True,
        timeout: int = 30,
        headers: dict[str, str] | None = None,
    ) -> _FakeResponse:
        self.calls.append((url, headers or {}))
        result = self.responses.get(url)
        if isinstance(result, Exception):
            raise result
        if result is None:
            return _FakeResponse(status_code=404)
        return result  # type: ignore[no-any-return]


def _repo_url(lang: str) -> str:
    return f"https://huggingface.co/gradium/phonon-40m-{lang}"


class TestEnsureModelDir:
    def test_complete_cache_hit_needs_no_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = _make_checkpoint(tmp_path, "es")

        def _boom() -> None:
            raise AssertionError("network must not be used")

        monkeypatch.setattr(models, "_create_session", _boom)

        assert ensure_model_dir("es-ES", cache_dir=tmp_path) == expected

    def test_first_use_download_writes_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        urls = {
            f"{_repo_url('es')}/resolve/main/{name}": _FakeResponse()
            for name in (
                "config.json",
                "tokenizer.json",
                "model.q8.gguf",
                "default-voice.safetensors",
            )
        }
        session = _FakeSession(urls)
        monkeypatch.setattr(models, "_create_session", lambda: session)
        monkeypatch.setattr(models, "_resolve_hf_token", lambda: "hf_test")

        model_dir = ensure_model_dir("es", cache_dir=tmp_path)

        assert (model_dir / "config.json").is_file()
        assert (model_dir / "tokenizer.json").is_file()
        assert (model_dir / "model.q8.gguf").is_file()
        assert (model_dir / "default-voice.safetensors").is_file()
        assert all(
            headers.get("Authorization") == "Bearer hf_test"
            for _, headers in session.calls
        )

    def test_missing_optional_voice_is_tolerated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        urls = {
            f"{_repo_url('es')}/resolve/main/{name}": _FakeResponse()
            for name in ("config.json", "tokenizer.json", "model.q8.gguf")
        }
        monkeypatch.setattr(models, "_create_session", lambda: _FakeSession(urls))
        monkeypatch.setattr(models, "_resolve_hf_token", lambda: None)

        model_dir = ensure_model_dir("es", cache_dir=tmp_path)

        assert (model_dir / "model.q8.gguf").is_file()
        assert not (model_dir / "default-voice.safetensors").exists()

    def test_gated_repository_reports_url_and_exact_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = f"{_repo_url('fr')}/resolve/main/config.json"
        monkeypatch.setattr(
            models, "_create_session", lambda: _FakeSession({url: _FakeResponse(401)})
        )
        monkeypatch.setattr(models, "_resolve_hf_token", lambda: None)

        with pytest.raises(TTSBackendError) as excinfo:
            ensure_model_dir("fr", cache_dir=tmp_path)

        message = str(excinfo.value)
        assert _repo_url("fr") in message
        assert str((tmp_path / "phonon-40m-fr").resolve()) in message
        assert "HF_TOKEN" in message
        assert "config.json" in message
        assert not (tmp_path / "phonon-40m-fr" / "config.json").exists()

    def test_network_error_reports_instructions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = f"{_repo_url('de')}/resolve/main/config.json"
        monkeypatch.setattr(
            models,
            "_create_session",
            lambda: _FakeSession({url: requests.exceptions.ConnectionError("offline")}),
        )
        monkeypatch.setattr(models, "_resolve_hf_token", lambda: None)

        with pytest.raises(TTSBackendError) as excinfo:
            ensure_model_dir("de", cache_dir=tmp_path)

        assert _repo_url("de") in str(excinfo.value)
        assert "Connection error" in str(excinfo.value)

    def test_incomplete_directory_names_missing_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_dir = tmp_path / "phonon-40m-en"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        with pytest.raises(TTSBackendError) as excinfo:
            ensure_model_dir("en", cache_dir=tmp_path)

        message = str(excinfo.value)
        assert "tokenizer.json" in message
        assert "model.q8.gguf" in message
        assert str(model_dir.resolve()) in message

    def test_auto_download_disabled_reports_instructions(self, tmp_path: Path) -> None:
        with pytest.raises(TTSBackendError) as excinfo:
            ensure_model_dir("pt", cache_dir=tmp_path, auto_download=False)

        assert _repo_url("pt") in str(excinfo.value)


class TestPhononBackend:
    def test_name_is_phonon(self) -> None:
        assert PhononBackend().name == "phonon"

    def test_unsupported_language_raises(self) -> None:
        with pytest.raises(TTSBackendError):
            PhononBackend(language="ja")

    def test_capabilities_report_model_voices(
        self, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        backend = PhononBackend(language="es", model_dir=model_dir)

        caps = backend.capabilities()

        assert [lang.code for lang in caps.languages] == ["es"]
        assert caps.languages[0].name == "Spanish"
        assert [voice.id for voice in caps.voices] == ["default", "Celia"]
        assert all(v.language_code == "es" for v in caps.voices)
        assert backend.active_device == "cpu"

    def test_capabilities_use_cached_languages(
        self, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        _make_checkpoint(tmp_path, "fr")
        backend = PhononBackend(cache_dir=tmp_path)

        caps = backend.capabilities()

        assert [lang.code for lang in caps.languages] == ["fr"]

    def test_capabilities_merge_catalog_voices_for_language(
        self,
        tmp_path: Path,
        fake_ptts: type[_FakeTTS],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        monkeypatch.setattr(
            sys.modules["ptts"],
            "catalog_voices",
            lambda: [
                {"name": "Lucía", "language": "es-ES"},
                {"name": "Zoey", "language": "en-US"},
            ],
            raising=False,
        )
        backend = PhononBackend(language="es", model_dir=model_dir)

        caps = backend.capabilities()

        assert [v.id for v in caps.voices] == ["default", "Celia", "Lucía"]
        backend.validate_selection("es", "Lucía")

    def test_validate_selection_accepts_known_voice(
        self, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        backend = PhononBackend(language="es", model_dir=model_dir)

        backend.validate_selection("es", "Celia")
        backend.validate_selection("es", "default")

    def test_validate_selection_rejects_unknown_voice(
        self, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        backend = PhononBackend(language="es", model_dir=model_dir)

        with pytest.raises(TTSBackendError) as excinfo:
            backend.validate_selection("es", "af_heart")

        assert "not available" in str(excinfo.value)

    @pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
    def test_speed_bounds_are_inclusive(
        self, speed: float, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        backend = PhononBackend(language="es", model_dir=model_dir)

        backend.synthesize("hola", "es", "default", tmp_path / "out.wav", speed=speed)

        assert (tmp_path / "out.wav").is_file()

    @pytest.mark.parametrize("speed", [0.49, 2.01])
    def test_speed_out_of_range_raises(
        self, speed: float, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        backend = PhononBackend(language="es", model_dir=model_dir)

        with pytest.raises(TTSBackendError) as excinfo:
            backend.synthesize(
                "hola", "es", "default", tmp_path / "out.wav", speed=speed
            )

        assert "between 0.5 and 2.0" in str(excinfo.value)

    def test_empty_text_raises(self, tmp_path: Path, fake_ptts: type[_FakeTTS]) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        backend = PhononBackend(language="es", model_dir=model_dir)

        with pytest.raises(TTSBackendError):
            backend.synthesize("   ", "es", "default", tmp_path / "out.wav")

    def test_speed_maps_to_duration_delta(
        self, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        backend = PhononBackend(language="es", model_dir=model_dir)

        backend.synthesize("hola", "es", "default", tmp_path / "a.wav", speed=2.0)
        assert _FakeTTS.instances[-1].conditions == {"duration_delta": -0.5}

        backend.synthesize("hola", "es", "default", tmp_path / "b.wav", speed=0.5)
        assert _FakeTTS.instances[-1].conditions == {"duration_delta": 1.0}

        backend.synthesize("hola", "es", "default", tmp_path / "c.wav", speed=1.0)
        assert _FakeTTS.instances[-1].conditions == {}

    def test_synthesis_appends_acoustic_tail(
        self, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        model_dir = _make_checkpoint(tmp_path, "es")
        backend = PhononBackend(language="es", model_dir=model_dir)
        out = tmp_path / "out.wav"

        backend.synthesize("hola", "es", "default", out)

        data, sample_rate = sf.read(str(out))
        expected = (SAMPLE_RATE // 10) + int(sample_rate * 0.08)
        assert sample_rate == SAMPLE_RATE
        assert len(data) == expected

    def test_default_voice_is_passed_as_none(
        self, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        backend = PhononBackend(language="es", model_dir=model_dir)

        backend.synthesize("hola", "es", "default", tmp_path / "out.wav")

        assert _FakeTTS.instances[-1].last_synth == ("hola", None)

    def test_quant_is_detected_from_weights(
        self, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        PhononBackend(language="es", model_dir=model_dir).capabilities()

        assert _FakeTTS.instances[-1].quant == "q8"

    def test_openvino_device_is_rejected(
        self, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        backend = PhononBackend(language="es", model_dir=model_dir, device="openvino")

        with pytest.raises(TTSBackendError) as excinfo:
            backend.capabilities()

        assert "OpenVINO" in str(excinfo.value)

    def test_unknown_device_is_rejected(
        self, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        backend = PhononBackend(language="es", model_dir=model_dir, device="vulkan")

        with pytest.raises(TTSBackendError):
            backend.capabilities()

    def test_missing_model_dir_raises(
        self, tmp_path: Path, fake_ptts: type[_FakeTTS]
    ) -> None:
        backend = PhononBackend(language="es", model_dir=tmp_path / "does-not-exist")

        with pytest.raises(TTSBackendError):
            backend.capabilities()

    def test_missing_ptts_reports_install_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_dir = _make_checkpoint(tmp_path, "es")
        monkeypatch.setitem(sys.modules, "ptts", None)
        backend = PhononBackend(language="es", model_dir=model_dir)

        with pytest.raises(TTSBackendError) as excinfo:
            backend.capabilities()

        assert "ebook-audiobook-engine[phonon]" in str(excinfo.value)
