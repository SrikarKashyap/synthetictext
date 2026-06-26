"""Shared utility functions for text cleaning, ID generation, and data I/O."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import pandas as pd

logger = logging.getLogger("synthetictext")

F = TypeVar("F", bound=Callable[..., Any])


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0) -> Callable[[F], F]:
    """Decorator for retrying API calls with exponential backoff."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.2fs...",
                        attempt + 1,
                        max_retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
            raise last_exception  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_PREFIXES_TO_STRIP = [
    "Here is",
    "Here's",
    "Social media post:",
    "Post:",
    "Text:",
    "Output:",
    "Result:",
    "Translation:",
    "Rewritten:",
    "Version:",
]


def clean_generated_text(text: str) -> str:
    """Remove common LLM artifacts from generated text."""
    if not text:
        return ""
    cleaned = text.strip()
    for prefix in _PREFIXES_TO_STRIP:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
            if cleaned.startswith(":") or cleaned.startswith("-"):
                cleaned = cleaned[1:].strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from an LLM response, tolerating markdown fences."""
    if not text:
        return {}
    decoder = json.JSONDecoder()

    def _parse_first_object(candidate: str) -> Dict[str, Any]:
        candidate = candidate.strip()
        for idx, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return {}

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        parsed = _parse_first_object(m.group(1))
        if parsed:
            return parsed
    parsed = _parse_first_object(text)
    if parsed:
        return parsed
    logger.warning("Failed to parse JSON from: %s...", text[:100])
    return {}


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def generate_sample_id(prefix: str, index: int, source: str) -> str:
    """Generate a unique sample ID incorporating a content hash."""
    unique_str = f"{prefix}_{source}_{index}_{time.time()}"
    hash_str = hashlib.md5(unique_str.encode()).hexdigest()[:12]
    return f"{prefix}_syn_{hash_str}"


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["id", "text"]


def load_data(
    path: Path | str,
    label_column: str = "label",
) -> pd.DataFrame:
    """Load a CSV dataset, validating required columns.

    The returned DataFrame always has columns ``id``, ``text``, and
    ``label`` (renamed from *label_column* if necessary).
    """
    df = pd.read_csv(path)
    if "id" not in df.columns:
        df["id"] = range(len(df))
    if "text" not in df.columns:
        raise ValueError(f"CSV at {path} is missing a 'text' column.")
    if label_column not in df.columns:
        raise ValueError(f"CSV at {path} is missing the label column '{label_column}'.")
    if label_column != "label":
        df = df.rename(columns={label_column: "label"})
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def save_data(
    df: pd.DataFrame,
    path: Path | str,
) -> str:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d samples to %s", len(df), path)
    return str(path)


def get_class_distribution(df: pd.DataFrame, label_col: str = "label") -> Dict[int, int]:
    return df[label_col].value_counts().to_dict()  # type: ignore[no-any-return]


def sample_balanced(
    df: pd.DataFrame,
    n_per_class: int,
    label_col: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample up to *n_per_class* rows per label value."""
    parts: List[pd.DataFrame] = []
    for label_val in sorted(df[label_col].unique()):
        sub = df[df[label_col] == label_val]
        n = min(n_per_class, len(sub))
        parts.append(sub.sample(n=n, random_state=random_state))
    return pd.concat(parts, ignore_index=True)
