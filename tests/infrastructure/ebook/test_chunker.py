"""Tests for sentence-aware text chunking."""

from __future__ import annotations

from audiobook_engine.infrastructure.ebook.chunker import (
    chunk_text,
    split_sentences,
)


class TestSplitSentences:
    def test_simple_sentences(self) -> None:
        text = "Hello world. How are you? I am fine!"
        result = split_sentences(text)
        assert result == ["Hello world.", "How are you?", "I am fine!"]

    def test_abbreviation_not_split(self) -> None:
        text = "Dr. Smith went to Washington. He was happy."
        result = split_sentences(text)
        assert len(result) == 2
        assert result[0] == "Dr. Smith went to Washington."

    def test_single_sentence(self) -> None:
        text = "Just one sentence here."
        result = split_sentences(text)
        assert len(result) == 1

    def test_empty_text(self) -> None:
        assert split_sentences("") == []

    def test_whitespace_only(self) -> None:
        assert split_sentences("   \n  ") == []

    def test_ellipsis_handling(self) -> None:
        text = "Well... I suppose so. Yes."
        result = split_sentences(text)
        assert any("Well..." in s for s in result)

    def test_dialogue_question_with_tag(self) -> None:
        text = "—¿Adónde vas? —preguntó Juan. Ya es tarde."
        result = split_sentences(text)
        assert len(result) == 2
        assert result[0] == "—¿Adónde vas? —preguntó Juan."
        assert result[1] == "Ya es tarde."

    def test_spanish_abbreviations(self) -> None:
        text = "El Sr. González y la Sra. Martínez hablaron con D. Pedro."
        result = split_sentences(text)
        assert len(result) == 1



class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        text = "Short text."
        result = chunk_text(text, max_chars=500)
        assert result == ["Short text."]

    def test_respects_sentence_boundaries(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_text(text, max_chars=30)
        # Each sentence should be in its own chunk
        assert len(result) >= 2
        for chunk in result:
            # No chunk should contain partial sentences
            assert any(
                chunk.endswith(s.strip())
                for s in ["First sentence.", "Second sentence.", "Third sentence."]
            )

    def test_groups_sentences_when_possible(self) -> None:
        text = "A short sentence. Another short one."
        result = chunk_text(text, max_chars=100)
        assert len(result) == 1
        assert result[0] == text

    def test_empty_text(self) -> None:
        assert chunk_text("") == []

    def test_whitespace_only(self) -> None:
        assert chunk_text("   \n  ") == []

    def test_long_sentence_becomes_own_chunk(self) -> None:
        text = "A" * 200 + "."
        result = chunk_text(text, max_chars=100)
        assert len(result) == 1
        assert result[0] == text

    def test_deterministic(self) -> None:
        text = "First part. Second part. Third part. Fourth part."
        r1 = chunk_text(text, max_chars=40)
        r2 = chunk_text(text, max_chars=40)
        assert r1 == r2

    def test_all_text_preserved(self) -> None:
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a second sentence to test with. "
            "And a third one for good measure!"
        )
        result = chunk_text(text, max_chars=50)
        combined = " ".join(result)
        # All original words should be present
        for word in text.split():
            assert word in combined
