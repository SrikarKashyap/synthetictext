from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider
from synthetictext.task import TaskSpec
from synthetictext.utils import generate_sample_id


class BaseGenerator(ABC):
    def __init__(
        self,
        task: TaskSpec,
        llm: BaseLLMProvider,
        renderer: PromptRenderer,
    ) -> None:
        self.task = task
        self.llm = llm
        self.renderer = renderer

    @abstractmethod
    def generate(
        self, language: str, num_samples: int, **kwargs: Any
    ) -> pd.DataFrame:
        ...

    def _build_dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=["id", "text", "label", "source", "metadata"])
        df = pd.DataFrame(records)
        return df[["id", "text", "label", "source", "metadata"]]

    def _base_metadata(self, language: str, **extra: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "language": language,
        }
        meta.update(extra)
        return meta

    def _make_record(
        self,
        text: str,
        label: int,
        source: str,
        prefix: str,
        index: int,
        language: str,
        **extra_meta: Any,
    ) -> dict[str, Any]:
        return {
            "id": generate_sample_id(prefix, index, source),
            "text": text,
            "label": label,
            "source": source,
            "metadata": self._base_metadata(language, **extra_meta),
        }

    def _compute_label_distribution(
        self, num_samples: int, imbalance_ratio: dict[int, float] | None = None
    ) -> dict[int, int]:
        labels = self.task.label_indices
        if imbalance_ratio:
            total_weight = sum(imbalance_ratio.get(lbl, 1.0) for lbl in labels)
            dist = {}
            remaining = num_samples
            for i, lbl in enumerate(labels):
                if i == len(labels) - 1:
                    dist[lbl] = remaining
                else:
                    count = round(num_samples * imbalance_ratio.get(lbl, 1.0) / total_weight)
                    dist[lbl] = count
                    remaining -= count
            return dist

        per_label = num_samples // len(labels)
        remainder = num_samples % len(labels)
        dist = {}
        for i, lbl in enumerate(labels):
            dist[lbl] = per_label + (1 if i < remainder else 0)
        return dist
