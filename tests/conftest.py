"""Shared fixtures for synthetictext tests."""
from __future__ import annotations

import pandas as pd
import pytest

from synthetictext.task import LanguageConfig, TaskSpec
from synthetictext.providers.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock provider for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._call_count = 0

    def generate(
        self,
        prompt: str,
        *,
        model=None,
        temperature=0.9,
        max_tokens=250,
        system_prompt=None,
    ) -> str:
        if self._responses:
            resp = self._responses[self._call_count % len(self._responses)]
        else:
            resp = "This is a mock generated text for testing purposes."
        self._call_count += 1
        return resp


@pytest.fixture
def binary_task() -> TaskSpec:
    return TaskSpec(
        name="Test Sentiment",
        labels={0: "negative", 1: "positive"},
        description="Classify text as positive or negative.",
        label_descriptions={
            0: "Text expressing dissatisfaction.",
            1: "Text expressing satisfaction.",
        },
        topics={"general": ["product quality", "customer service"]},
        text_domain="review",
    )


@pytest.fixture
def multiclass_task() -> TaskSpec:
    return TaskSpec(
        name="News Category",
        labels={0: "politics", 1: "sports", 2: "technology"},
        description="Classify news headlines by category.",
    )


@pytest.fixture
def lang_config() -> LanguageConfig:
    return LanguageConfig(
        languages={"en": "English", "de": "German"},
        iso_map={"en": "en", "de": "de"},
    )


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [f"test_{i}" for i in range(10)],
        "text": [
            "Great product, highly recommend!",
            "Terrible experience, never again.",
            "Average quality, nothing special.",
            "Absolutely love this item!",
            "Worst purchase I've ever made.",
            "It works as expected.",
            "Outstanding customer service!",
            "Broken on arrival, very disappointed.",
            "Good value for the price.",
            "Not worth the money at all.",
        ],
        "label": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    })
