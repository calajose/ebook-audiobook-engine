"""Tests for domain protocols and exception hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING, is_protocol

import pytest

from audiobook_engine.domain.exceptions import (
    AudioAssemblyError,
    AudiobookEngineError,
    ConfigurationError,
    EbookError,
    JobError,
    TTSBackendError,
)
from audiobook_engine.domain.models import BackendCapabilities, Book
from audiobook_engine.domain.protocols import EbookParser, TTSBackend

if TYPE_CHECKING:
    from pathlib import Path


class TestExceptionHierarchy:
    def test_base_exception(self) -> None:
        assert issubclass(AudiobookEngineError, Exception)

    @pytest.mark.parametrize(
        "exc_class",
        [
            EbookError,
            TTSBackendError,
            AudioAssemblyError,
            ConfigurationError,
            JobError,
        ],
    )
    def test_subclass_of_base(self, exc_class: type[Exception]) -> None:
        assert issubclass(exc_class, AudiobookEngineError)


class TestEbookParserProtocol:
    def test_has_inspect_method(self) -> None:
        assert hasattr(EbookParser, "inspect")

    def test_is_protocol(self) -> None:
        assert is_protocol(EbookParser)

    def test_concrete_implementation_satisfies(self) -> None:
        class GoodParser:
            def inspect(self, path: Path) -> Book:
                return Book(title="", author="", language="en")

        parser = GoodParser()
        assert isinstance(parser, EbookParser)

    def test_missing_method_does_not_satisfy(self) -> None:
        class BadParser:
            pass

        parser = BadParser()
        assert not isinstance(parser, EbookParser)


class TestTTSBackendProtocol:
    def test_has_capabilities_and_synthesize(self) -> None:
        assert hasattr(TTSBackend, "capabilities")
        assert hasattr(TTSBackend, "synthesize")

    def test_is_protocol(self) -> None:
        assert is_protocol(TTSBackend)

    def test_concrete_implementation_satisfies(self) -> None:
        class GoodBackend:
            def capabilities(self) -> BackendCapabilities:
                return BackendCapabilities()

            def synthesize(
                self,
                text: str,
                language: str,
                voice: str,
                output_path: Path,
            ) -> None:
                pass

        backend = GoodBackend()
        assert isinstance(backend, TTSBackend)

    def test_missing_synthesize_does_not_satisfy(self) -> None:
        class IncompleteBackend:
            def capabilities(self) -> BackendCapabilities:
                return BackendCapabilities()

        backend = IncompleteBackend()
        assert not isinstance(backend, TTSBackend)
