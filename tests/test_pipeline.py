"""Tests for the main SyntheticDataGenerator pipeline."""
from __future__ import annotations

import pandas as pd

from synthetictext.pipeline import SyntheticDataGenerator
from synthetictext.task import TaskSpec

from tests.conftest import MockLLMProvider


class TestSyntheticDataGenerator:
    def test_single_language_direct(self, binary_task: TaskSpec) -> None:
        llm = MockLLMProvider(["A well-written sample text for testing the generation."])
        gen = SyntheticDataGenerator(task=binary_task, llm_provider=llm)

        df = gen.generate(
            language="English",
            num_samples=10,
            strategies=["direct"],
            apply_filters=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "text" in df.columns
        assert "label" in df.columns

    def test_multiple_strategies(
        self, binary_task: TaskSpec, sample_df: pd.DataFrame
    ) -> None:
        llm = MockLLMProvider([
            "Generated text sample for classification.",
            "negative: Bad stuff.\npositive: Good stuff.",
        ])
        gen = SyntheticDataGenerator(task=binary_task, llm_provider=llm)

        df = gen.generate(
            language="English",
            num_samples=20,
            strategies=["direct", "contrastive"],
            strategy_weights=[0.7, 0.3],
            apply_filters=False,
        )
        assert len(df) > 0

    def test_weight_normalization(self) -> None:
        weights = SyntheticDataGenerator._normalize_weights(
            ["a", "b", "c"], [2, 3, 5]
        )
        assert abs(sum(weights) - 1.0) < 1e-9
        assert abs(weights[0] - 0.2) < 1e-9

    def test_allocation(self) -> None:
        alloc = SyntheticDataGenerator._allocate_samples(100, [0.5, 0.3, 0.2])
        assert sum(alloc) == 100
        assert alloc[0] == 50
        assert alloc[1] == 30
        assert alloc[2] == 20
