"""Tests for prompt rendering."""
from __future__ import annotations

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.task import TaskSpec


class TestPromptRenderer:
    def test_render_direct(self, binary_task: TaskSpec) -> None:
        renderer = PromptRenderer(binary_task)
        prompt = renderer.render_direct(label_index=1, language="English", topic="laptops")
        assert "English" in prompt
        assert "positive" in prompt
        assert "laptops" in prompt
        assert binary_task.description in prompt

    def test_render_paraphrase(self, binary_task: TaskSpec) -> None:
        renderer = PromptRenderer(binary_task)
        prompt = renderer.render_paraphrase(
            text="Great product!", label_index=1, language="English"
        )
        assert "Great product!" in prompt
        assert "positive" in prompt

    def test_render_contrastive(self, binary_task: TaskSpec) -> None:
        renderer = PromptRenderer(binary_task)
        prompt = renderer.render_contrastive(label_index=1, language="English", topic="phones")
        assert "positive" in prompt
        assert "negative" in prompt
        assert "phones" in prompt

    def test_render_judge(self, binary_task: TaskSpec) -> None:
        renderer = PromptRenderer(binary_task)
        prompt = renderer.render_judge(
            text="Sample text", label_index=0, language="English"
        )
        assert "Sample text" in prompt
        assert "negative" in prompt

    def test_custom_prompt_override(self) -> None:
        task = TaskSpec(
            name="T",
            labels={0: "a", 1: "b"},
            description="d",
            custom_prompts={"direct": "Custom: generate {label_name} in {language}"},
        )
        renderer = PromptRenderer(task)
        prompt = renderer.render_direct(label_index=1, language="German")
        assert prompt == "Custom: generate b in German"
