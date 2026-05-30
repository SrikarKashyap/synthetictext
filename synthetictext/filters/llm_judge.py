"""LLM-as-judge quality filter."""

from __future__ import annotations

import random
from typing import Optional

import pandas as pd

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider
from synthetictext.utils import logger, parse_json_response

from .base import BaseFilter


class LLMJudgeFilter(BaseFilter):
    """Uses an LLM to evaluate samples on realistic, label_correct, clear, grammatical.

    Supports sampling for cost control: only a random subset of rows is judged,
    and the rest pass through. Samples where the LLM judgment fails to parse
    are kept (benefit of the doubt).
    """

    CRITERIA = ("realistic", "label_correct", "clear", "grammatical")

    def __init__(
        self,
        provider: BaseLLMProvider,
        renderer: PromptRenderer,
        language: str = "en",
        sample_rate: float = 1.0,
        min_passing_criteria: int = 4,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> None:
        self._provider = provider
        self._renderer = renderer
        self._language = language
        self._sample_rate = max(0.0, min(1.0, sample_rate))
        self._min_passing = min_passing_criteria
        self._model = model
        self._temperature = temperature

    @property
    def name(self) -> str:
        return "LLMJudgeFilter"

    def _judge_sample(self, text: str, label_index: int) -> bool:
        """Return True if the sample passes LLM judgment."""
        try:
            prompt = self._renderer.render_judge(text, label_index, self._language)
            response = self._provider.generate(
                prompt, model=self._model, temperature=self._temperature, max_tokens=250
            )
            result = parse_json_response(response)
            passing = sum(
                1 for criterion in self.CRITERIA if result.get(criterion) is True
            )
            return passing >= self._min_passing
        except Exception:
            return True

    def filter(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        initial_count = len(df)

        if self._sample_rate >= 1.0:
            indices_to_judge = list(df.index)
        else:
            k = max(1, int(len(df) * self._sample_rate))
            indices_to_judge = sorted(random.sample(list(df.index), k))

        keep_mask = pd.Series(True, index=df.index)

        for idx in indices_to_judge:
            row = df.loc[idx]
            passes = self._judge_sample(str(row["text"]), int(row["label"]))
            if not passes:
                keep_mask[idx] = False

        out = df[keep_mask].copy().reset_index(drop=True)
        removed = initial_count - len(out)
        logger.info(
            "%s: removed %d / %d samples (judged %d)",
            self.name,
            removed,
            initial_count,
            len(indices_to_judge),
        )
        return out
