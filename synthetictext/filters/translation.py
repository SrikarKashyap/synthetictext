"""Round-trip translation quality filter."""

from __future__ import annotations

import pandas as pd

from synthetictext.providers.base import BaseTranslationProvider
from synthetictext.utils import logger

from .base import BaseFilter


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


class TranslationQualityFilter(BaseFilter):
    """Filters pivot/backtranslation samples by round-trip consistency.

    Expects the DataFrame to have an ``english_original`` column containing
    the original English text before translation. Back-translates the current
    text to English and checks word-overlap Jaccard similarity against the
    original.

    Requires a configured ``BaseTranslationProvider``.
    """

    def __init__(
        self,
        provider: BaseTranslationProvider,
        source_lang: str = "en",
        threshold: float = 0.70,
    ) -> None:
        self._provider = provider
        self._source_lang = source_lang
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "TranslationQualityFilter"

    def _check_row(self, text: str, english_original: str) -> bool:
        """Return True if the sample passes the quality check."""
        try:
            back_translated = self._provider.translate(
                text, source_lang=self._source_lang, target_lang="en"
            )
            similarity = _jaccard_similarity(back_translated, english_original)
            return similarity >= self._threshold
        except Exception:
            return True

    def filter(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        if "english_original" not in df.columns:
            logger.warning(
                "%s: 'english_original' column not found, skipping filter.", self.name
            )
            return df.copy()

        initial_count = len(df)
        mask = df.apply(
            lambda row: self._check_row(
                str(row["text"]), str(row["english_original"])
            ),
            axis=1,
        )
        out = df[mask].copy().reset_index(drop=True)

        removed = initial_count - len(out)
        logger.info("%s: removed %d / %d samples", self.name, removed, initial_count)
        return out
