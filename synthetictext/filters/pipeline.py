"""Composable filter pipeline."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from synthetictext.task import TaskSpec
from synthetictext.utils import logger

from .base import BaseFilter
from .basic import BasicFilter
from .deduplication import EmbeddingDeduplicator
from .leakage import LeakageFilter
from .markers import MarkerFilter


class FilterPipeline:
    """Composes multiple filters into a sequential pipeline.

    Usage::

        pipeline = FilterPipeline()
        pipeline.add(BasicFilter(valid_labels=[0, 1]))
        pipeline.add(LeakageFilter(task_labels={0: "negative", 1: "positive"}))
        result = pipeline.run(df)
    """

    def __init__(self) -> None:
        self._filters: list[BaseFilter] = []

    def add(self, f: BaseFilter) -> "FilterPipeline":
        """Append a filter to the pipeline. Returns self for chaining."""
        self._filters.append(f)
        return self

    def run(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run all filters in sequence, returning the final filtered DataFrame."""
        logger.info("FilterPipeline: starting with %d samples", len(df))
        current = df.copy()

        for f in self._filters:
            before = len(current)
            current = f.filter(current, **kwargs)
            after = len(current)
            logger.info(
                "FilterPipeline [%s]: %d -> %d (-%d)",
                f.name,
                before,
                after,
                before - after,
            )

        logger.info("FilterPipeline: finished with %d samples", len(current))
        return current

    @classmethod
    def default(
        cls,
        task: TaskSpec,
        *,
        language: str = "en",
        min_length: int = 10,
        max_length: int = 1000,
        use_embeddings: bool = True,
        dedup_threshold: float = 0.90,
        dedup_model: Optional[str] = None,
    ) -> "FilterPipeline":
        """Create a sensible default pipeline: basic -> leakage -> dedup -> markers.

        Parameters
        ----------
        task : TaskSpec
            Task specification providing labels, marker keywords, etc.
        language : str
            Language code for leakage patterns.
        min_length, max_length : int
            Character length bounds for BasicFilter.
        use_embeddings : bool
            If *True*, include embedding-based deduplication (requires
            ``synthetictext[embeddings]``). Set to *False* for lightweight
            filtering without the sentence-transformers dependency.
        dedup_threshold : float
            Cosine similarity threshold for deduplication.
        dedup_model : str, optional
            Override the default sentence-transformers model.
        """
        pipeline = cls()

        pipeline.add(
            BasicFilter(
                valid_labels=list(task.labels.keys()),
                min_length=min_length,
                max_length=max_length,
            )
        )

        pipeline.add(
            LeakageFilter(task_labels=task.labels, language=language)
        )

        if use_embeddings:
            dedup_kwargs: dict = {"threshold": dedup_threshold}
            if dedup_model:
                dedup_kwargs["model_name"] = dedup_model
            pipeline.add(EmbeddingDeduplicator(**dedup_kwargs))

        if task.marker_keywords:
            pipeline.add(MarkerFilter(task))

        return pipeline
