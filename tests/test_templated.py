"""Tests for templated style-transfer generation."""
from __future__ import annotations

import pandas as pd
import pytest

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider
from synthetictext.templated import TemplatedGenerator

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
        return '{"samples": [{"text": "A garden update in the same style."}]}'


class TestTemplatedGenerator:
    def test_generates_samples(self) -> None:
        llm = MockLLMProvider([
            (
                '{"samples": ['
                '{"text": "A garden update in the same crisp style."},'
                '{"text": "Another urban gardening note with matching rhythm."}'
                "]}"
            )
        ])
        gen = TemplatedGenerator(llm_provider=llm)

        df = gen.generate(
            text="A compact EV market note with crisp phrasing.",
            source_topic="electric vehicles",
            target_topic="urban gardening",
            style="compact market note",
            num_samples=2,
            language="English",
        )

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["id", "text", "source", "metadata"]
        assert len(df) == 2
        assert df.loc[0, "text"] == "A garden update in the same crisp style."
        assert df.loc[0, "metadata"]["source_topic"] == "electric vehicles"
        assert df.loc[0, "metadata"]["target_topic"] == "urban gardening"
        assert df.loc[0, "metadata"]["style"] == "compact market note"
        assert "source_text_hash" in df.loc[0, "metadata"]

    def test_generate_one_returns_first_text(self) -> None:
        llm = MockLLMProvider([
            '{"samples": [{"text": "A single templated gardening note."}]}'
        ])
        gen = TemplatedGenerator(llm_provider=llm)

        text = gen.generate_one(
            text="Original EV note.",
            source_topic="electric vehicles",
            target_topic="urban gardening",
            style="brief note",
        )

        assert text == "A single templated gardening note."

    def test_requests_json_response_format(self) -> None:
        llm = CapturingLLMProvider([
            '{"samples": [{"text": "A garden update in the same style."}]}'
        ])
        gen = TemplatedGenerator(llm_provider=llm)

        gen.generate(
            text="Original EV note.",
            source_topic="electric vehicles",
            target_topic="urban gardening",
            style="brief note",
            num_samples=1,
        )

        assert llm.response_formats == [{"type": "json_object"}]
        assert llm.max_tokens_values == [4000]

    def test_parses_top_level_text_response(self) -> None:
        llm = MockLLMProvider(['{"text": "A top-level templated response."}'])
        gen = TemplatedGenerator(llm_provider=llm)

        df = gen.generate(
            text="Original EV note.",
            source_topic="electric vehicles",
            target_topic="urban gardening",
            style="brief note",
            num_samples=1,
        )

        assert len(df) == 1
        assert df.loc[0, "text"] == "A top-level templated response."

    def test_parses_texts_array_response(self) -> None:
        llm = MockLLMProvider(['{"texts": ["First templated text.", "Second templated text."]}'])
        gen = TemplatedGenerator(llm_provider=llm)

        df = gen.generate(
            text="Original EV note.",
            source_topic="electric vehicles",
            target_topic="urban gardening",
            style="brief note",
            num_samples=2,
        )

        assert len(df) == 2
        assert list(df["text"]) == ["First templated text.", "Second templated text."]

    def test_falls_back_for_legacy_provider_without_response_format(self) -> None:
        gen = TemplatedGenerator(llm_provider=LegacyLLMProvider())  # type: ignore[arg-type]

        df = gen.generate(
            text="Original EV note.",
            source_topic="electric vehicles",
            target_topic="urban gardening",
            style="brief note",
            num_samples=1,
        )

        assert len(df) == 1
        assert df.loc[0, "text"] == "A garden update in the same style."

    def test_skips_malformed_samples(self) -> None:
        llm = MockLLMProvider([
            (
                '{"samples": ['
                '{"text": "Valid templated text."},'
                '{"text": ""},'
                '{"not_text": "Missing text."}'
                "]}"
            )
        ])
        gen = TemplatedGenerator(llm_provider=llm)

        df = gen.generate(
            text="Original EV note.",
            source_topic="electric vehicles",
            target_topic="urban gardening",
            style="brief note",
            num_samples=3,
        )

        assert len(df) == 1
        assert df.loc[0, "text"] == "Valid templated text."

    def test_invalid_json_returns_empty_dataframe(self) -> None:
        gen = TemplatedGenerator(llm_provider=MockLLMProvider(["not json"]))

        df = gen.generate(
            text="Original EV note.",
            source_topic="electric vehicles",
            target_topic="urban gardening",
            style="brief note",
            num_samples=2,
        )

        assert df.empty
        assert list(df.columns) == ["id", "text", "source", "metadata"]

    def test_requires_non_empty_inputs(self) -> None:
        gen = TemplatedGenerator(llm_provider=MockLLMProvider())

        with pytest.raises(ValueError, match="non-empty text"):
            gen.generate(
                text=" ",
                source_topic="electric vehicles",
                target_topic="urban gardening",
                style="brief note",
            )

    def test_requires_positive_num_samples(self) -> None:
        gen = TemplatedGenerator(llm_provider=MockLLMProvider())

        with pytest.raises(ValueError, match="greater than zero"):
            gen.generate(
                text="Original EV note.",
                source_topic="electric vehicles",
                target_topic="urban gardening",
                style="brief note",
                num_samples=0,
            )


class TestTemplatedPrompt:
    def test_render_templated_generation_without_task(self) -> None:
        renderer = PromptRenderer()

        prompt = renderer.render_templated_generation(
            text="Original EV note.",
            source_topic="electric vehicles",
            target_topic="urban gardening",
            style="brief note",
            num_samples=3,
            language="English",
        )

        assert "Original EV note." in prompt
        assert "electric vehicles" in prompt
        assert "urban gardening" in prompt
        assert "brief note" in prompt
        assert "3 sample" in prompt
        assert "English" in prompt
