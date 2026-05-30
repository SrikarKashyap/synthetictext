from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider, BaseTranslationProvider
from synthetictext.task import LanguageConfig, TaskSpec
from synthetictext.utils import clean_generated_text, logger

from .base import BaseGenerator


class PivotGenerator(BaseGenerator):
    """Generates in a high-resource source language, then translates to target.

    Designed for low-resource target languages. Optionally performs cross-lingual
    transfer by also generating from a related high-resource language.
    """

    def __init__(
        self,
        task: TaskSpec,
        llm: BaseLLMProvider,
        renderer: PromptRenderer,
        *,
        translator: BaseTranslationProvider,
        lang_config: LanguageConfig | None = None,
        source_language: str = "en",
        temperature: float = 0.9,
    ) -> None:
        super().__init__(task, llm, renderer)
        self.translator = translator
        self.lang_config = lang_config
        self.source_language = source_language
        self.temperature = temperature

    def generate(
        self,
        language: str,
        num_samples: int,
        *,
        source_language: str | None = None,
        cross_lingual: bool = False,
        cross_lingual_ratio: float = 0.3,
        imbalance_ratio: dict[int, float] | None = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        src_lang = source_language or self.source_language
        distribution = self._compute_label_distribution(num_samples, imbalance_ratio)

        # Split between primary source and cross-lingual related language
        related_lang: str | None = None
        cross_lingual_count = 0
        if cross_lingual and self.lang_config:
            related_lang = self.lang_config.get_related_language(language)
            if related_lang and related_lang != language:
                cross_lingual_count = round(num_samples * cross_lingual_ratio)

        records: list[dict[str, Any]] = []
        sample_idx = 0

        total = sum(distribution.values())
        pbar = tqdm(total=total, desc="PivotGenerator", disable=not show_progress)

        for label_idx, count in distribution.items():
            for i in range(count):
                try:
                    use_related = (
                        related_lang is not None
                        and sample_idx < cross_lingual_count
                    )
                    gen_lang = related_lang if use_related else src_lang

                    topic = self.task.random_topic()
                    prompt = self.renderer.render_direct(
                        label_idx, gen_lang, topic=topic
                    )
                    raw = self.llm.generate(
                        prompt, temperature=self.temperature, **kwargs
                    )
                    generated_text = clean_generated_text(raw)
                    if not generated_text:
                        continue

                    translated = self.translator.translate(
                        generated_text,
                        source_lang=gen_lang,
                        target_lang=language,
                    )
                    text = clean_generated_text(translated)
                    if text:
                        record = self._make_record(
                            text=text,
                            label=label_idx,
                            source="pivot",
                            prefix="pivot",
                            index=sample_idx,
                            language=language,
                            source_language=gen_lang,
                            topic=topic,
                            cross_lingual=use_related,
                            label_name=self.task.label_name(label_idx),
                        )
                        records.append(record)
                        sample_idx += 1
                except Exception as e:
                    logger.warning(
                        "PivotGenerator: failed at sample %d: %s", sample_idx, e
                    )
                finally:
                    pbar.update(1)

        pbar.close()
        return self._build_dataframe(records)
