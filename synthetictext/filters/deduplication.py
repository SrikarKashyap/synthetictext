"""Embedding-based near-duplicate removal."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from synthetictext.utils import logger

from .base import BaseFilter


class EmbeddingDeduplicator(BaseFilter):
    """Removes near-duplicate texts using cosine similarity on sentence embeddings.

    Supports two modes:
    - Intra-synthetic dedup (default): removes duplicates within the given DataFrame.
    - Cross-dedup: if ``original_df`` is passed via kwargs, also removes synthetic
      samples that are too similar to the original training data.

    Requires ``pip install synthetictext[embeddings]`` (sentence-transformers, sklearn).
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        threshold: float = 0.90,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._model: Any = None

    @property
    def name(self) -> str:
        return "EmbeddingDeduplicator"

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for EmbeddingDeduplicator. "
                    "Install with: pip install synthetictext[embeddings]"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def _intra_dedup(self, embeddings: np.ndarray) -> np.ndarray:
        """Return boolean mask of rows to keep (first occurrence wins)."""
        from sklearn.metrics.pairwise import cosine_similarity

        n = len(embeddings)
        keep = np.ones(n, dtype=bool)
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0.0)

        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if keep[j] and sim_matrix[i, j] >= self._threshold:
                    keep[j] = False
        return keep

    def _cross_dedup(
        self, synth_embeddings: np.ndarray, orig_embeddings: np.ndarray
    ) -> np.ndarray:
        """Return boolean mask of synthetic rows to keep."""
        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(synth_embeddings, orig_embeddings)
        max_sim = sim_matrix.max(axis=1)
        return max_sim < self._threshold

    def filter(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        initial_count = len(df)
        texts = df["text"].astype(str).tolist()
        embeddings = self._encode(texts)

        keep_mask = self._intra_dedup(embeddings)

        original_df: pd.DataFrame | None = kwargs.get("original_df")
        if original_df is not None and not original_df.empty:
            orig_texts = original_df["text"].astype(str).tolist()
            orig_embeddings = self._encode(orig_texts)
            cross_mask = self._cross_dedup(embeddings, orig_embeddings)
            keep_mask = keep_mask & cross_mask

        out = df[keep_mask].copy().reset_index(drop=True)
        removed = initial_count - len(out)
        logger.info("%s: removed %d / %d samples", self.name, removed, initial_count)
        return out
