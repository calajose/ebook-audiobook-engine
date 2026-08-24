"""Tests for text normalization."""

from __future__ import annotations

from audiobook_engine.infrastructure.ebook.normalizer import (
    normalize,
    normalize_dialogue,
    normalize_whitespace,
    remove_noise,
)


class TestNormalizeWhitespace:
    def test_collapse_spaces(self) -> None:
        assert normalize_whitespace("hello   world") == "hello world"

    def test_tabs_to_spaces(self) -> None:
        assert normalize_whitespace("hello\tworld") == "hello world"

    def test_collapse_newlines(self) -> None:
        assert normalize_whitespace("a\n\n\nb") == "a\nb"

    def test_strip_leading_trailing(self) -> None:
        assert normalize_whitespace("  hello  ") == "hello"

    def test_preserve_single_newline(self) -> None:
        assert normalize_whitespace("a\nb") == "a\nb"


class TestRemoveNoise:
    def test_page_number(self) -> None:
        text = "Some text.\nPage 42\nMore text."
        result = remove_noise(text)
        assert "Page 42" not in result
        assert "Some text." in result
        assert "More text." in result

    def test_standalone_chapter_number(self) -> None:
        text = "Chapter 3\n\nThe story continues."
        result = remove_noise(text)
        assert "Chapter 3" not in result
        assert "The story continues." in result

    def test_roman_numeral(self) -> None:
        text = "xvii\n\nIntroduction text."
        result = remove_noise(text)
        assert "xvii" not in result
        assert "Introduction text." in result

    def test_separator_line(self) -> None:
        text = "Before\n***\nAfter"
        result = remove_noise(text)
        assert "***" not in result
        assert "Before" in result
        assert "After" in result

    def test_copyright_line(self) -> None:
        text = "Copyright © 2024\nAll rights reserved."
        result = remove_noise(text)
        assert "Copyright" not in result
        assert "All rights reserved." not in result

    def test_contents_heading(self) -> None:
        text = "Contents\nChapter 1\nChapter 2"
        result = remove_noise(text)
        assert "Contents" not in result

    def test_preserves_real_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        assert remove_noise(text) == text


class TestNormalizeDialogue:
    def test_guillemets_to_quotes(self) -> None:
        assert normalize_dialogue("«Hola mundo»") == '"Hola mundo"'

    def test_curly_quotes_to_straight(self) -> None:
        assert normalize_dialogue("“Texto”") == '"Texto"'

    def test_dialogue_dash_start_of_line(self) -> None:
        assert normalize_dialogue("- Hola, dijo él.") == "— Hola, dijo él."
        assert normalize_dialogue("– Buenos días.") == "— Buenos días."

    def test_inverted_punctuation_spacing(self) -> None:
        assert normalize_dialogue("¿ Cómo estás ?") == "¿Cómo estás ?"
        assert normalize_dialogue("¡ Qué sorpresa !") == "¡Qué sorpresa !"

    def test_ellipsis_standardization(self) -> None:
        assert normalize_dialogue("Bueno . . . sí.") == "Bueno ... sí."
        assert normalize_dialogue("Espera....") == "Espera..."


class TestNormalize:
    def test_full_pipeline(self) -> None:
        text = "  Hello   world  \n\n\nPage 5\n\n  Goodbye  "
        result = normalize(text)
        assert "Hello world" in result
        assert "Page 5" not in result
        assert "Goodbye" in result

    def test_empty_input(self) -> None:
        assert normalize("") == ""

    def test_only_noise(self) -> None:
        text = "Page 1\n***\nPage 2"
        result = normalize(text)
        assert result == ""

    def test_preserves_paragraph_structure(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        result = normalize(text)
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_deterministic(self) -> None:
        text = "  Test   input  \n\nPage 3\n\n  Output  "
        assert normalize(text) == normalize(text)
