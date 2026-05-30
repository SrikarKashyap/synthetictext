"""Main orchestrator for synthetic text data generation."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider, BaseTranslationProvider
from synthetictext.task import LanguageConfig, TaskSpec
from synthetictext.utils import logger, save_data

# Lazy imports for generators/filters to avoid circular imports at module level.


def _resolve_llm_provider(
    provider: BaseLLMProvider | str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseLLMProvider:
    """Accept a provider instance or a shorthand string like ``"openai"``."""
    if isinstance(provider, BaseLLMProvider):
        return provider
    if provider == "openai":
        from synthetictext.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=api_key, default_model=model or "gpt-4o-mini")
    raise ValueError(f"Unknown LLM provider shorthand: '{provider}'")


class SyntheticDataGenerator:
    """High-level orchestrator for multi-strategy, multilingual synthetic data generation.

    Args:
        task: The classification task specification.
        llm_provider: An :class:`BaseLLMProvider` instance or a shorthand
            string (``"openai"``).
        llm_model: Default model name forwarded to the provider.
        api_key: Optional API key forwarded when *llm_provider* is a string.
        translation_provider: Optional :class:`BaseTranslationProvider` for
            backtranslation / pivot strategies.
        lang_config: Optional :class:`LanguageConfig` for multilingual runs.
    """

    def __init__(
        self,
        task: TaskSpec,
        llm_provider: BaseLLMProvider | str = "openai",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        translation_provider: Optional[BaseTranslationProvider] = None,
        lang_config: Optional[LanguageConfig] = None,
    ) -> None:
        self.task = task
        self.llm = _resolve_llm_provider(llm_provider, llm_model, api_key)
        self.translation = translation_provider
        self.lang_config = lang_config or LanguageConfig.single()
        self.renderer = PromptRenderer(task)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        language: str,
        num_samples: int = 1000,
        strategies: Sequence[str] = ("direct",),
        strategy_weights: Optional[Sequence[float]] = None,
        original_df: Optional[pd.DataFrame] = None,
        apply_filters: bool = True,
        filter_pipeline: Optional[Any] = None,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data for a single language.

        Args:
            language: Language code (e.g. ``"en"``, ``"de"``).
            num_samples: Total number of samples to generate across all
                strategies.
            strategies: List of strategy names to use. Supported:
                ``"direct"``, ``"paraphrase"``, ``"contrastive"``,
                ``"backtranslation"``, ``"pivot"``.
            strategy_weights: Relative weights controlling how *num_samples*
                is split across strategies.  Defaults to equal weighting.
            original_df: Existing training data, required for ``"paraphrase"``
                and ``"backtranslation"`` strategies.  Must have columns
                ``text`` and ``label``.
            apply_filters: Whether to run the quality-filter pipeline.
            filter_pipeline: A :class:`FilterPipeline` instance, or ``None``
                to use the default pipeline.
            save_path: Optional file path to save results.

        Returns:
            Filtered DataFrame with columns ``id``, ``text``, ``label``,
            ``source``, and metadata columns.
        """
        weights = self._normalize_weights(strategies, strategy_weights)
        allocations = self._allocate_samples(num_samples, weights)

        all_dfs: List[pd.DataFrame] = []

        for strategy, n in zip(strategies, allocations):
            if n <= 0:
                continue
            logger.info("Generating %d samples via '%s' for %s", n, strategy, language)
            gen = self._build_generator(strategy, original_df=original_df)
            df = gen.generate(language=language, num_samples=n, original_df=original_df)
            all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame(columns=["id", "text", "label", "source"])

        combined = pd.concat(all_dfs, ignore_index=True)

        if apply_filters:
            combined = self._run_filters(combined, original_df, language, filter_pipeline)

        if save_path:
            save_data(combined, save_path)

        return combined

    def generate_all(
        self,
        num_samples: int = 1000,
        strategies: Sequence[str] = ("direct",),
        strategy_weights: Optional[Sequence[float]] = None,
        original_data_dir: Optional[str] = None,
        label_column: str = "label",
        apply_filters: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for every language in the :class:`LanguageConfig`.

        Returns:
            Mapping of language code to generated DataFrame.
        """
        from pathlib import Path

        results: Dict[str, pd.DataFrame] = {}

        for lang in tqdm(self.lang_config.all_language_codes(), desc="Languages"):
            original_df = None
            if original_data_dir:
                path = Path(original_data_dir) / f"{lang}.csv"
                if path.exists():
                    from synthetictext.utils import load_data

                    original_df = load_data(path, label_column=label_column)

            try:
                df = self.generate(
                    language=lang,
                    num_samples=num_samples,
                    strategies=strategies,
                    strategy_weights=strategy_weights,
                    original_df=original_df,
                    apply_filters=apply_filters,
                )
                results[lang] = df
                logger.info("Completed %s: %d samples", lang, len(df))
            except Exception as e:
                logger.error("Failed for %s: %s", lang, e)
                results[lang] = pd.DataFrame()

            if output_dir:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_data(df, str(Path(output_dir) / f"{lang}_synthetic_{ts}.csv"))

        self._print_summary(results)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_generator(
        self, strategy: str, original_df: Optional[pd.DataFrame] = None
    ) -> Any:
        from synthetictext.generators.backtranslation import BacktranslationGenerator
        from synthetictext.generators.contrastive import ContrastiveGenerator
        from synthetictext.generators.direct import DirectGenerator
        from synthetictext.generators.paraphrase import ParaphraseGenerator
        from synthetictext.generators.pivot import PivotGenerator

        if strategy == "direct":
            return DirectGenerator(self.task, self.llm, self.renderer)
        elif strategy == "paraphrase":
            return ParaphraseGenerator(self.task, self.llm, self.renderer)
        elif strategy == "contrastive":
            return ContrastiveGenerator(self.task, self.llm, self.renderer)
        elif strategy == "backtranslation":
            if self.translation is None:
                raise ValueError(
                    "BacktranslationGenerator requires a translation_provider."
                )
            return BacktranslationGenerator(
                self.task, self.llm, self.renderer, translator=self.translation
            )
        elif strategy == "pivot":
            if self.translation is None:
                raise ValueError("PivotGenerator requires a translation_provider.")
            return PivotGenerator(
                self.task,
                self.llm,
                self.renderer,
                translator=self.translation,
                lang_config=self.lang_config,
            )
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'")

    def _run_filters(
        self,
        df: pd.DataFrame,
        original_df: Optional[pd.DataFrame],
        language: str,
        pipeline: Optional[Any],
    ) -> pd.DataFrame:
        if pipeline is None:
            from synthetictext.filters.pipeline import FilterPipeline

            pipeline = FilterPipeline.default(self.task)
        return pipeline.run(df, original_df=original_df, language=language)

    @staticmethod
    def _normalize_weights(
        strategies: Sequence[str],
        weights: Optional[Sequence[float]],
    ) -> List[float]:
        n = len(strategies)
        if weights is None:
            return [1.0 / n] * n
        if len(weights) != n:
            raise ValueError(
                f"strategy_weights length ({len(weights)}) must match "
                f"strategies length ({n})."
            )
        total = sum(weights)
        return [w / total for w in weights]

    @staticmethod
    def _allocate_samples(
        total: int, weights: List[float]
    ) -> List[int]:
        allocations = [int(total * w) for w in weights]
        remainder = total - sum(allocations)
        for i in range(remainder):
            allocations[i % len(allocations)] += 1
        return allocations

    @staticmethod
    def _print_summary(results: Dict[str, pd.DataFrame]) -> None:
        total = 0
        for lang, df in sorted(results.items()):
            n = len(df)
            total += n
            status = "ok" if n > 0 else "FAIL"
            print(f"  [{status}] {lang}: {n} samples")
        print(f"  Total: {total} samples across {len(results)} languages")
