from __future__ import annotations

import re
from typing import Any

import pandas as pd
from tqdm import tqdm

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider
from synthetictext.task import TaskSpec
from synthetictext.utils import clean_generated_text, logger

from .base import BaseGenerator


class ContrastiveGenerator(BaseGenerator):
    TEMPERATURE = 0.8

    def __init__(
        self,
        task: TaskSpec,
        llm: BaseLLMProvider,
        renderer: PromptRenderer,
    ) -> None:
        if not task.is_binary:
            raise ValueError(
                "ContrastiveGenerator only supports binary classification tasks"
            )
        super().__init__(task, llm, renderer)

    def generate(
        self,
        language: str,
        num_samples: int,
        *,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        # Each generation call produces a pair, so we need ceil(num_samples / 2) calls
        num_pairs = (num_samples + 1) // 2
        records: list[dict[str, Any]] = []
        sample_idx = 0

        pbar = tqdm(
            total=num_pairs, desc="ContrastiveGenerator", disable=not show_progress
        )

        labels = self.task.label_indices
        for pair_idx in range(num_pairs):
            try:
                topic = self.task.random_topic()
                prompt = self.renderer.render_contrastive(
                    labels[0], language, topic=topic
                )
                raw = self.llm.generate(
                    prompt, temperature=self.TEMPERATURE
                )
                pair = self._parse_pair(raw)
                if pair is None:
                    logger.warning(
                        "ContrastiveGenerator: could not parse pair %d", pair_idx
                    )
                    continue

                text_a, text_b = pair
                for label_idx, text in zip(labels, [text_a, text_b]):
                    if len(records) >= num_samples:
                        break
                    text = clean_generated_text(text)
                    if text:
                        record = self._make_record(
                            text=text,
                            label=label_idx,
                            source="contrastive",
                            prefix="contrast",
                            index=sample_idx,
                            language=language,
                            pair_id=pair_idx,
                            topic=topic,
                            label_name=self.task.label_name(label_idx),
                        )
                        records.append(record)
                        sample_idx += 1
            except Exception as e:
                logger.warning(
                    "ContrastiveGenerator: failed at pair %d: %s", pair_idx, e
                )
            finally:
                pbar.update(1)

        pbar.close()
        return self._build_dataframe(records)

    def _parse_pair(self, raw: str) -> tuple[str, str] | None:
        """Parse LLM response into two contrastive samples.

        Supports formats:
          - Numbered: "1. ..." / "2. ..."
          - Labeled: "Label A: ..." / "Label B: ..."
          - Separator: text separated by "---" or "***"
        """
        separator_match = re.split(r"\n\s*(?:---+|\*\*\*+)\s*\n", raw)
        if len(separator_match) == 2:
            return separator_match[0].strip(), separator_match[1].strip()

        numbered = re.findall(r"(?:^|\n)\s*[12][.)]\s*(.+?)(?=\n\s*[12][.)]|\Z)", raw, re.DOTALL)
        if len(numbered) >= 2:
            return numbered[0].strip(), numbered[1].strip()

        label_names = [self.task.label_name(i) for i in self.task.label_indices]
        pattern = "|".join(re.escape(name) for name in label_names)
        labeled = re.split(rf"(?:^|\n)\s*(?:{pattern})\s*[:]\s*", raw)
        labeled = [s.strip() for s in labeled if s.strip()]
        if len(labeled) >= 2:
            return labeled[0], labeled[1]

        lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        if len(lines) >= 2:
            return lines[0], lines[1]

        return None
