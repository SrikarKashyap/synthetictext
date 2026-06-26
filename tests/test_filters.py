"""Tests for quality filters."""
from __future__ import annotations

import pandas as pd
import pytest

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider
from synthetictext.task import TaskSpec


def _make_df(texts: list[str], labels: list[int] | None = None) -> pd.DataFrame:
    n = len(texts)
    if labels is None:
        labels = [0] * n
    return pd.DataFrame({
        "id": [f"s_{i}" for i in range(n)],
        "text": texts,
        "label": labels,
    })


class TestBasicFilter:
    def test_removes_empty(self) -> None:
        from synthetictext.filters.basic import BasicFilter

        df = _make_df(["hello world test", "", None, "another good text"])  # type: ignore[list-item]
        f = BasicFilter(min_length=5)
        result = f.filter(df)
        assert len(result) == 2

    def test_length_bounds(self) -> None:
        from synthetictext.filters.basic import BasicFilter

        df = _make_df(["short", "a" * 2000, "This is a normal length text."])
        f = BasicFilter(min_length=10, max_length=100)
        result = f.filter(df)
        assert len(result) == 1
        assert result.iloc[0]["text"] == "This is a normal length text."


class TestLeakageFilter:
    def test_detects_leakage(self) -> None:
        from synthetictext.filters.leakage import LeakageFilter

        df = _make_df([
            "This is classified as positive",
            "A normal text without issues",
            "label: 1 this is positive",
        ])
        f = LeakageFilter()
        result = f.filter(df)
        assert len(result) == 1
        assert "normal" in result.iloc[0]["text"]


class TestMarkerFilter:
    def test_filters_missing_markers(self) -> None:
        from synthetictext.filters.markers import MarkerFilter

        task = TaskSpec(
            name="T",
            labels={0: "good", 1: "bad"},
            description="d",
            marker_keywords={1: ["terrible", "awful", "horrible"]},
        )
        df = _make_df(
            ["terrible product", "nice day", "just ok"],
            [1, 1, 0],
        )
        f = MarkerFilter(task, min_markers=1)
        result = f.filter(df)
        assert len(result) == 2
        label_1_rows = result[result["label"] == 1]
        assert len(label_1_rows) == 1
        assert "terrible" in label_1_rows.iloc[0]["text"]


class CapturingJudgeProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.response_formats = []

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
        return (
            '{"realistic": true, "label_correct": true, '
            '"clear": true, "grammatical": true}'
        )


class TestLLMJudgeFilter:
    def test_requests_json_response_format(self, binary_task: TaskSpec) -> None:
        from synthetictext.filters.llm_judge import LLMJudgeFilter

        provider = CapturingJudgeProvider()
        judge = LLMJudgeFilter(
            provider=provider,
            renderer=PromptRenderer(binary_task),
            language="English",
        )

        result = judge.filter(_make_df(["good text here"], [1]))

        assert len(result) == 1
        assert provider.response_formats == [{"type": "json_object"}]


class TestFilterPipeline:
    def test_default_pipeline(self, binary_task: TaskSpec) -> None:
        from synthetictext.filters.pipeline import FilterPipeline

        pipeline = FilterPipeline.default(binary_task, use_embeddings=False)
        df = _make_df(
            ["good text here", "", "this is classified as positive", "another good one"],
            [0, 0, 1, 1],
        )
        result = pipeline.run(df)
        assert len(result) <= len(df)
        assert len(result) > 0

    def test_composability(self) -> None:
        from synthetictext.filters.basic import BasicFilter
        from synthetictext.filters.pipeline import FilterPipeline

        pipe = FilterPipeline()
        pipe.add(BasicFilter(min_length=5))
        df = _make_df(["hi", "hello world this is fine"])
        result = pipe.run(df)
        assert len(result) == 1
