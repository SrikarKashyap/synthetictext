from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider
from synthetictext.task import TaskSpec
from synthetictext.utils import clean_generated_text, logger

from .base import BaseGenerator


class ParaphraseGenerator(BaseGenerator):
    TEMPERATURE = 0.7

    def __init__(
        self,
        task: TaskSpec,
        llm: BaseLLMProvider,
        renderer: PromptRenderer,
        *,
        original_df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(task, llm, renderer)
        self.original_df = original_df

    def generate(
        self,
        language: str,
        num_samples: int,
        *,
        original_df: pd.DataFrame | None = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        source_df = original_df if original_df is not None else self.original_df
        if source_df is None or source_df.empty:
            raise ValueError(
                "ParaphraseGenerator requires original_df with samples to paraphrase"
            )

        sampled = source_df.sample(n=num_samples, replace=True).reset_index(drop=True)
        records: list[dict[str, Any]] = []

        pbar = tqdm(
            total=num_samples, desc="ParaphraseGenerator", disable=not show_progress
        )

        for idx, row in sampled.iterrows():
            try:
                original_text = row["text"]
                label_idx = int(row["label"])

                prompt = self.renderer.render_paraphrase(
                    original_text, label_idx, language
                )
                raw = self.llm.generate(
                    prompt, temperature=self.TEMPERATURE
                )
                text = clean_generated_text(raw)
                if text:
                    original_id = row.get("id", f"unknown_{idx}")
                    record = self._make_record(
                        text=text,
                        label=label_idx,
                        source="paraphrase",
                        prefix="para",
                        index=idx,
                        language=language,
                        original_id=original_id,
                        label_name=self.task.label_name(label_idx),
                    )
                    records.append(record)
            except Exception as e:
                logger.warning(
                    "ParaphraseGenerator: failed at sample %d: %s", idx, e
                )
            finally:
                pbar.update(1)

        pbar.close()
        return self._build_dataframe(records)
