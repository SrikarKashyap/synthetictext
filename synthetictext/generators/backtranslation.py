from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider, BaseTranslationProvider
from synthetictext.task import TaskSpec
from synthetictext.utils import clean_generated_text, logger

from .base import BaseGenerator


class BacktranslationGenerator(BaseGenerator):
    """Generates augmented samples by round-tripping through a pivot language.

    Does not use the LLM for generation — only the translation provider.
    The llm/renderer params are accepted for interface compatibility but unused.
    """

    def __init__(
        self,
        task: TaskSpec,
        llm: BaseLLMProvider,
        renderer: PromptRenderer,
        *,
        translator: BaseTranslationProvider,
        original_df: pd.DataFrame | None = None,
        pivot_languages: list[str] | None = None,
    ) -> None:
        super().__init__(task, llm, renderer)
        self.translator = translator
        self.original_df = original_df
        self.pivot_languages = pivot_languages or ["en"]

    def generate(
        self,
        language: str,
        num_samples: int,
        *,
        original_df: pd.DataFrame | None = None,
        pivot_languages: list[str] | None = None,
        class_balanced: bool = True,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        source_df = original_df if original_df is not None else self.original_df
        if source_df is None or source_df.empty:
            raise ValueError(
                "BacktranslationGenerator requires original_df with samples"
            )

        pivots = pivot_languages or self.pivot_languages
        # Filter out pivot == source language
        pivots = [p for p in pivots if p != language]
        if not pivots:
            pivots = ["en"]

        if class_balanced:
            sampled = self._balanced_sample(source_df, num_samples)
        else:
            sampled = source_df.sample(n=num_samples, replace=True).reset_index(drop=True)

        records: list[dict[str, Any]] = []
        sample_idx = 0

        pbar = tqdm(
            total=len(sampled), desc="BacktranslationGenerator", disable=not show_progress
        )

        for idx, row in sampled.iterrows():
            try:
                original_text = row["text"]
                label_idx = int(row["label"])
                pivot = pivots[idx % len(pivots)]

                backtranslated = self.translator.backtranslate(
                    original_text, source_lang=language, pivot_lang=pivot
                )
                text = clean_generated_text(backtranslated)
                if text and text != original_text:
                    original_id = row.get("id", f"unknown_{idx}")
                    record = self._make_record(
                        text=text,
                        label=label_idx,
                        source="backtranslation",
                        prefix="bt",
                        index=sample_idx,
                        language=language,
                        pivot_language=pivot,
                        original_id=original_id,
                    )
                    records.append(record)
                    sample_idx += 1
            except Exception as e:
                logger.warning(
                    "BacktranslationGenerator: failed at sample %d: %s", idx, e
                )
            finally:
                pbar.update(1)

        pbar.close()
        return self._build_dataframe(records)

    def _balanced_sample(
        self, df: pd.DataFrame, num_samples: int
    ) -> pd.DataFrame:
        labels = df["label"].unique()
        per_label = num_samples // len(labels)
        remainder = num_samples % len(labels)

        parts = []
        for i, label in enumerate(sorted(labels)):
            n = per_label + (1 if i < remainder else 0)
            subset = df[df["label"] == label]
            parts.append(subset.sample(n=n, replace=True))

        return pd.concat(parts, ignore_index=True)
