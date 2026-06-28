"""Prompt rendering engine that merges TaskSpec fields into templates."""
from __future__ import annotations

from typing import Any, Dict, Optional

from synthetictext.prompts.defaults import DEFAULT_PROMPTS
from synthetictext.task import TaskSpec


class PromptRenderer:
    """Render prompt templates using values from a :class:`TaskSpec`.

    If the ``TaskSpec`` carries a ``custom_prompts`` entry for a given
    strategy, that template is used instead of the library default.
    """

    def __init__(self, task: TaskSpec | None = None) -> None:
        self.task = task

    def _base_vars(self, language: str) -> Dict[str, Any]:
        if self.task is None:
            return {
                "language": language,
                "task_name": "RAG Q&A",
                "task_description": "Generate grounded question-answer pairs.",
                "text_domain": "source chunk",
                "word_min": 0,
                "word_max": 0,
            }
        return {
            "language": language,
            "task_name": self.task.name,
            "task_description": self.task.description,
            "text_domain": self.task.text_domain,
            "word_min": self.task.word_count_range[0],
            "word_max": self.task.word_count_range[1],
        }

    def _label_vars(self, label_index: int) -> Dict[str, str]:
        if self.task is None:
            raise ValueError("PromptRenderer requires a TaskSpec for label-based prompts.")
        v: Dict[str, str] = {
            "label_name": self.task.label_name(label_index),
            "label_description": self.task.label_description(label_index),
        }
        if self.task.is_binary:
            other = [i for i in self.task.label_indices if i != label_index][0]
            v["other_label_name"] = self.task.label_name(other)
            v["other_label_description"] = self.task.label_description(other)
        return v

    def _get_template(self, strategy: str) -> str:
        custom = self.task.get_prompt(strategy) if self.task is not None else None
        if custom is not None:
            return custom
        if strategy not in DEFAULT_PROMPTS:
            raise ValueError(
                f"No default prompt for strategy '{strategy}'. "
                f"Available: {list(DEFAULT_PROMPTS.keys())}"
            )
        return DEFAULT_PROMPTS[strategy]

    def render_direct(
        self,
        label_index: int,
        language: str,
        topic: Optional[str] = None,
    ) -> str:
        template = self._get_template("direct")
        variables = {
            **self._base_vars(language),
            **self._label_vars(label_index),
            "topic": topic or "any relevant topic",
        }
        return template.format(**variables)

    def render_paraphrase(
        self,
        text: str,
        label_index: int,
        language: str,
    ) -> str:
        template = self._get_template("paraphrase")
        variables = {
            **self._base_vars(language),
            **self._label_vars(label_index),
            "text": text,
        }
        return template.format(**variables)

    def render_contrastive(
        self,
        label_index: int,
        language: str,
        topic: Optional[str] = None,
    ) -> str:
        template = self._get_template("contrastive")
        variables = {
            **self._base_vars(language),
            **self._label_vars(label_index),
            "topic": topic or "any relevant topic",
        }
        return template.format(**variables)

    def render_judge(
        self,
        text: str,
        label_index: int,
        language: str,
    ) -> str:
        template = self._get_template("judge")
        variables = {
            **self._base_vars(language),
            **self._label_vars(label_index),
            "text": text,
        }
        return template.format(**variables)

    def render_rag_qa(
        self,
        chunk: str,
        num_samples: int,
        language: str,
    ) -> str:
        template = self._get_template("rag_qa")
        variables = {
            **self._base_vars(language),
            "chunk": chunk,
            "num_samples": num_samples,
        }
        return template.format(**variables)

    def render_templated_generation(
        self,
        text: str,
        source_topic: str,
        target_topic: str,
        style: str,
        num_samples: int,
        language: str,
    ) -> str:
        template = self._get_template("templated_generation")
        variables = {
            **self._base_vars(language),
            "text": text,
            "source_topic": source_topic,
            "target_topic": target_topic,
            "style": style,
            "num_samples": num_samples,
        }
        return template.format(**variables)
