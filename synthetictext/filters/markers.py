"""Marker keyword presence filter."""

from __future__ import annotations

import pandas as pd

from synthetictext.task import TaskSpec
from synthetictext.utils import logger

from .base import BaseFilter


class MarkerFilter(BaseFilter):
    """Checks that samples for target label classes contain required marker keywords.

    For each label that has marker keywords defined in the TaskSpec, this filter
    verifies that the text contains at least ``min_markers`` of those keywords.
    Samples belonging to labels without marker keywords pass through unchanged.
    """

    def __init__(self, task: TaskSpec, min_markers: int = 1) -> None:
        self._task = task
        self._min_markers = min_markers
        self._keywords_map: dict[int, list[str]] = task.marker_keywords or {}

    @property
    def name(self) -> str:
        return "MarkerFilter"

    def _passes(self, text: str, label: int) -> bool:
        keywords = self._keywords_map.get(label)
        if not keywords:
            return True
        text_lower = text.lower()
        matched = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matched >= self._min_markers

    def filter(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        initial_count = len(df)
        mask = df.apply(
            lambda row: self._passes(str(row["text"]), int(row["label"])), axis=1
        )
        out = df[mask].copy().reset_index(drop=True)

        removed = initial_count - len(out)
        logger.info("%s: removed %d / %d samples", self.name, removed, initial_count)
        return out
