"""Tests for TaskSpec and LanguageConfig."""
from __future__ import annotations

import json

import pytest

from synthetictext.task import LanguageConfig, TaskSpec


class TestTaskSpec:
    def test_basic_creation(self, binary_task: TaskSpec) -> None:
        assert binary_task.name == "Test Sentiment"
        assert binary_task.num_classes == 2
        assert binary_task.is_binary is True
        assert binary_task.label_indices == [0, 1]

    def test_requires_two_labels(self) -> None:
        with pytest.raises(ValueError, match="at least 2 labels"):
            TaskSpec(name="Bad", labels={0: "only"}, description="x")

    def test_label_accessors(self, binary_task: TaskSpec) -> None:
        assert binary_task.label_name(0) == "negative"
        assert binary_task.label_name(1) == "positive"
        assert "dissatisfaction" in binary_task.label_description(0)

    def test_label_description_fallback(self) -> None:
        task = TaskSpec(name="T", labels={0: "a", 1: "b"}, description="d")
        assert task.label_description(0) == "a"

    def test_random_topic(self, binary_task: TaskSpec) -> None:
        topic = binary_task.random_topic()
        assert topic is not None
        assert topic in ["product quality", "customer service"]

    def test_random_topic_none_when_no_topics(self) -> None:
        task = TaskSpec(name="T", labels={0: "a", 1: "b"}, description="d")
        assert task.random_topic() is None

    def test_multiclass(self, multiclass_task: TaskSpec) -> None:
        assert multiclass_task.num_classes == 3
        assert multiclass_task.is_binary is False

    def test_round_trip_serialization(self, binary_task: TaskSpec) -> None:
        d = binary_task.to_dict()
        s = json.dumps(d)
        restored = TaskSpec.from_dict(json.loads(s))
        assert restored.name == binary_task.name
        assert restored.labels == binary_task.labels
        assert restored.description == binary_task.description
        assert restored.text_domain == binary_task.text_domain

    def test_custom_prompts(self) -> None:
        task = TaskSpec(
            name="T",
            labels={0: "a", 1: "b"},
            description="d",
            custom_prompts={"direct": "Custom prompt {language}"},
        )
        assert task.get_prompt("direct") == "Custom prompt {language}"
        assert task.get_prompt("paraphrase") is None


class TestLanguageConfig:
    def test_single_factory(self) -> None:
        lc = LanguageConfig.single("fr", "French")
        assert lc.language_name("fr") == "French"
        assert lc.all_language_codes() == ["fr"]

    def test_iso_code_identity(self) -> None:
        lc = LanguageConfig(languages={"en": "English"})
        assert lc.iso_code("en") == "en"

    def test_iso_code_mapped(self, lang_config: LanguageConfig) -> None:
        assert lang_config.iso_code("de") == "de"

    def test_related_language(self) -> None:
        lc = LanguageConfig(
            languages={"ne": "Nepali", "hi": "Hindi"},
            related_languages={"ne": "hi"},
        )
        assert lc.get_related_language("ne") == "hi"
        assert lc.get_related_language("hi") is None
