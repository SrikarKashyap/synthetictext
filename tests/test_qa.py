"""Tests for RAG Q&A generation."""
from __future__ import annotations

import pandas as pd
import pytest

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider
from synthetictext.qa import RAGQAGenerator

from tests.conftest import MockLLMProvider


class CapturingLLMProvider(MockLLMProvider):
    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(responses)
        self.response_formats = []
        self.max_tokens_values = []

    def generate(
        self,
        prompt: str,
        *,
        model=None,
        temperature=0.9,
        max_tokens=250,
        system_prompt=None,
        response_format=None,
    ) -> str:
        self.response_formats.append(response_format)
        self.max_tokens_values.append(max_tokens)
        return super().generate(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            response_format=response_format,
        )


class LegacyLLMProvider(BaseLLMProvider):
    def generate(
        self,
        prompt: str,
        *,
        model=None,
        temperature=0.9,
        max_tokens=250,
        system_prompt=None,
    ) -> str:
        return '{"qa_pairs": [{"question": "Q?", "answer": "A."}]}'


class TestRAGQAGenerator:
    def test_generates_qa_pairs(self) -> None:
        llm = MockLLMProvider([
            (
                '{"qa_pairs": ['
                '{"question": "What does the chunk describe?", "answer": "A test feature."},'
                '{"question": "Why is it useful?", "answer": "It helps evaluate RAG."}'
                "]}"
            )
        ])
        gen = RAGQAGenerator(llm_provider=llm)

        df = gen.generate(
            chunk="This chunk describes a test feature. It helps evaluate RAG.",
            num_samples=2,
            language="English",
        )

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["id", "question", "answer", "source", "metadata"]
        assert len(df) == 2
        assert df.loc[0, "question"] == "What does the chunk describe?"
        assert df.loc[0, "metadata"]["language"] == "English"
        assert "chunk_hash" in df.loc[0, "metadata"]

    def test_requests_json_response_format(self) -> None:
        llm = CapturingLLMProvider([
            '{"qa_pairs": [{"question": "Q?", "answer": "A."}]}'
        ])
        gen = RAGQAGenerator(llm_provider=llm)

        gen.generate(chunk="A chunk with one answer.", num_samples=1)

        assert llm.response_formats == [{"type": "json_object"}]
        assert llm.max_tokens_values == [4000]

    def test_falls_back_for_legacy_provider_without_response_format(self) -> None:
        gen = RAGQAGenerator(llm_provider=LegacyLLMProvider())  # type: ignore[arg-type]

        df = gen.generate(chunk="A chunk with one answer.", num_samples=1)

        assert len(df) == 1
        assert df.loc[0, "question"] == "Q?"

    def test_limits_extra_pairs(self) -> None:
        llm = MockLLMProvider([
            (
                '{"qa_pairs": ['
                '{"question": "Q1?", "answer": "A1"},'
                '{"question": "Q2?", "answer": "A2"}'
                "]}"
            )
        ])
        gen = RAGQAGenerator(llm_provider=llm)

        df = gen.generate(chunk="Q1 and Q2 are answerable.", num_samples=1)

        assert len(df) == 1
        assert df.loc[0, "question"] == "Q1?"

    def test_skips_malformed_pairs(self) -> None:
        llm = MockLLMProvider([
            (
                '{"qa_pairs": ['
                '{"question": "Valid?", "answer": "Yes."},'
                '{"question": "", "answer": "Missing question."},'
                '{"question": "Missing answer?"}'
                "]}"
            )
        ])
        gen = RAGQAGenerator(llm_provider=llm)

        df = gen.generate(chunk="Valid information is present.", num_samples=3)

        assert len(df) == 1
        assert df.loc[0, "answer"] == "Yes."

    def test_invalid_json_returns_empty_dataframe(self) -> None:
        llm = MockLLMProvider(["not json"])
        gen = RAGQAGenerator(llm_provider=llm)

        df = gen.generate(chunk="A chunk with content.", num_samples=2)

        assert df.empty
        assert list(df.columns) == ["id", "question", "answer", "source", "metadata"]

    def test_requires_non_empty_chunk(self) -> None:
        gen = RAGQAGenerator(llm_provider=MockLLMProvider())

        with pytest.raises(ValueError, match="non-empty"):
            gen.generate(chunk="  ", num_samples=1)

    def test_requires_positive_num_samples(self) -> None:
        gen = RAGQAGenerator(llm_provider=MockLLMProvider())

        with pytest.raises(ValueError, match="greater than zero"):
            gen.generate(chunk="Some content.", num_samples=0)


class TestRAGQAPrompt:
    def test_render_rag_qa_without_task(self) -> None:
        renderer = PromptRenderer()

        prompt = renderer.render_rag_qa(
            chunk="The library now supports RAG Q&A.",
            num_samples=3,
            language="English",
        )

        assert "The library now supports RAG Q&A." in prompt
        assert "3 question-answer pairs" in prompt
        assert "English" in prompt
