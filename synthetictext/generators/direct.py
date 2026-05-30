from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider
from synthetictext.task import TaskSpec
from synthetictext.utils import clean_generated_text, logger

from .base import BaseGenerator


class DirectGenerator(BaseGenerator):
    def __init__(
        self,
        task: TaskSpec,
        llm: BaseLLMProvider,
        renderer: PromptRenderer,
        *,
        temperature: float = 0.9,
    ) -> None:
        super().__init__(task, llm, renderer)
        self.temperature = temperature

    def generate(
        self,
        language: str,
        num_samples: int,
        *,
        imbalance_ratio: dict[int, float] | None = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        distribution = self._compute_label_distribution(num_samples, imbalance_ratio)
        records: list[dict[str, Any]] = []
        sample_idx = 0

        total = sum(distribution.values())
        pbar = tqdm(total=total, desc="DirectGenerator", disable=not show_progress)

        for label_idx, count in distribution.items():
            for _ in range(count):
                try:
                    topic = self.task.random_topic()
                    prompt = self.renderer.render_direct(
                        label_idx, language, topic=topic
                    )
                    raw = self.llm.generate(
                        prompt, temperature=self.temperature
                    )
                    text = clean_generated_text(raw)
                    if text:
                        record = self._make_record(
                            text=text,
                            label=label_idx,
                            source="direct",
                            prefix="direct",
                            index=sample_idx,
                            language=language,
                            topic=topic,
                            label_name=self.task.label_name(label_idx),
                        )
                        records.append(record)
                        sample_idx += 1
                except Exception as e:
                    logger.warning(
                        "DirectGenerator: failed to generate sample %d: %s",
                        sample_idx, e,
                    )
                finally:
                    pbar.update(1)

        pbar.close()
        return self._build_dataframe(records)
