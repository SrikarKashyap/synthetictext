"""Tests for generators (using mock LLM provider)."""
from __future__ import annotations

import pandas as pd
import pytest

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.task import LanguageConfig, TaskSpec

from tests.conftest import MockLLMProvider


class TestDirectGenerator:
    def test_generates_samples(self, binary_task: TaskSpec) -> None:
        from synthetictext.generators.direct import DirectGenerator

        llm = MockLLMProvider(["A positive review about great quality."])
        renderer = PromptRenderer(binary_task)
        gen = DirectGenerator(binary_task, llm, renderer)

        df = gen.generate(language="English", num_samples=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "text" in df.columns
        assert "label" in df.columns
        assert "id" in df.columns
        assert set(df["label"].unique()).issubset({0, 1})

    def test_balanced_generation(self, binary_task: TaskSpec) -> None:
        from synthetictext.generators.direct import DirectGenerator

        llm = MockLLMProvider(["Generated sample text here."])
        renderer = PromptRenderer(binary_task)
        gen = DirectGenerator(binary_task, llm, renderer)

        df = gen.generate(language="English", num_samples=20)
        counts = df["label"].value_counts()
        assert abs(counts.get(0, 0) - counts.get(1, 0)) <= 2


class TestParaphraseGenerator:
    def test_generates_from_original(
        self, binary_task: TaskSpec, sample_df: pd.DataFrame
    ) -> None:
        from synthetictext.generators.paraphrase import ParaphraseGenerator

        llm = MockLLMProvider(["This is a paraphrased version of the text."])
        renderer = PromptRenderer(binary_task)
        gen = ParaphraseGenerator(binary_task, llm, renderer)

        df = gen.generate(language="English", num_samples=5, original_df=sample_df)
        assert len(df) > 0
        assert "source" in df.columns

    def test_requires_original_df(self, binary_task: TaskSpec) -> None:
        from synthetictext.generators.paraphrase import ParaphraseGenerator

        llm = MockLLMProvider()
        renderer = PromptRenderer(binary_task)
        gen = ParaphraseGenerator(binary_task, llm, renderer)

        with pytest.raises(ValueError, match="original_df"):
            gen.generate(language="English", num_samples=5)


class TestContrastiveGenerator:
    def test_generates_pairs(self, binary_task: TaskSpec) -> None:
        from synthetictext.generators.contrastive import ContrastiveGenerator

        llm = MockLLMProvider(
            ["negative: Bad product terrible.\npositive: Great product amazing."]
        )
        renderer = PromptRenderer(binary_task)
        gen = ContrastiveGenerator(binary_task, llm, renderer)

        df = gen.generate(language="English", num_samples=6)
        assert len(df) > 0

    def test_rejects_non_binary(self, multiclass_task: TaskSpec) -> None:
        from synthetictext.generators.contrastive import ContrastiveGenerator

        llm = MockLLMProvider()
        renderer = PromptRenderer(multiclass_task)

        with pytest.raises(ValueError, match="binary"):
            ContrastiveGenerator(multiclass_task, llm, renderer)
