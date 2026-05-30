"""Label-leakage detection filter."""

from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

from synthetictext.utils import logger

from .base import BaseFilter

DEFAULT_PATTERNS: Dict[str, List[str]] = {
    "generic": [
        r"label\s*[:=]",
        r"class\s*[:=]",
        r"category\s*[:=]",
        r"classification\s*[:=]",
        r"sentiment\s*[:=]",
    ],
    "en": [
        r"this\s+(is|was)\s+(a|an)\s+(positive|negative|neutral)\s+(example|sample|text)",
        r"the\s+label\s+(is|for this)",
        r"classified\s+as\s*[:=]?",
        r"belongs?\s+to\s+(class|category)",
    ],
}


class LeakageFilter(BaseFilter):
    """Removes rows where generated text leaks its own label or class information."""

    def __init__(
        self,
        task_labels: dict[int, str] | None = None,
        patterns: Dict[str, List[str]] | None = None,
        language: str = "en",
    ) -> None:
        self._task_labels = task_labels or {}
        self._patterns = patterns or DEFAULT_PATTERNS
        self._language = language
        self._compiled = self._compile_patterns()

    @property
    def name(self) -> str:
        return "LeakageFilter"

    def _compile_patterns(self) -> list[re.Pattern]:
        raw: list[str] = []
        raw.extend(self._patterns.get("generic", []))
        raw.extend(self._patterns.get(self._language, []))

        for label_name in self._task_labels.values():
            escaped = re.escape(label_name.lower())
            raw.append(
                rf"(?:label|class|category|classified\s+as)\s*[:=]?\s*{escaped}"
            )

        return [re.compile(p, re.IGNORECASE) for p in raw]

    def _is_leaky(self, text: str) -> bool:
        text_lower = text.lower()
        return any(pat.search(text_lower) for pat in self._compiled)

    def filter(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        initial_count = len(df)
        mask = ~df["text"].astype(str).apply(self._is_leaky)
        out = df[mask].copy().reset_index(drop=True)

        removed = initial_count - len(out)
        logger.info("%s: removed %d / %d samples", self.name, removed, initial_count)
        return out
