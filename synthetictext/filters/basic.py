"""Basic quality filter: empty text, invalid labels, and length bounds."""

from __future__ import annotations

import pandas as pd

from synthetictext.utils import logger

from .base import BaseFilter


class BasicFilter(BaseFilter):
    """Removes rows with empty/null text, invalid labels, or out-of-range length."""

    def __init__(
        self,
        valid_labels: list[int] | None = None,
        min_length: int = 10,
        max_length: int = 1000,
    ) -> None:
        self._valid_labels = valid_labels
        self._min_length = min_length
        self._max_length = max_length

    @property
    def name(self) -> str:
        return "BasicFilter"

    def filter(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        initial_count = len(df)
        out = df.copy()

        out = out[out["text"].notna() & (out["text"].astype(str).str.strip() != "")]

        if self._valid_labels is not None:
            out = out[out["label"].isin(self._valid_labels)]

        lengths = out["text"].astype(str).str.len()
        out = out[(lengths >= self._min_length) & (lengths <= self._max_length)]

        removed = initial_count - len(out)
        logger.info("%s: removed %d / %d samples", self.name, removed, initial_count)
        return out.reset_index(drop=True)
