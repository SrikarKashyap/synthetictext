"""Templated style-transfer text generation."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from synthetictext.pipeline import _resolve_llm_provider
from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider
from synthetictext.utils import clean_generated_text, generate_sample_id, logger, parse_json_response


class TemplatedGenerator:
    """Generate new text on a target topic while preserving a source style."""

    TEMPERATURE = 0.7

    def __init__(
        self,
        llm_provider: BaseLLMProvider | str = "openai",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        renderer: Optional[PromptRenderer] = None,
    ) -> None:
        self.llm = _resolve_llm_provider(llm_provider, llm_model, api_key)
        self.renderer = renderer or PromptRenderer()
        self.last_response: str = ""

    def generate_one(
        self,
        text: str,
        source_topic: str,
        target_topic: str,
        style: str,
        language: str = "English",
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a single templated text sample."""
        df = self.generate(
            text=text,
            source_topic=source_topic,
            target_topic=target_topic,
            style=style,
            num_samples=1,
            language=language,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if df.empty:
            return ""
        return str(df.loc[0, "text"])

    def generate(
        self,
        text: str,
        source_topic: str,
        target_topic: str,
        style: str,
        num_samples: int = 1,
        language: str = "English",
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate templated text samples as a DataFrame."""
        cleaned_text, source_topic, target_topic, style = self._validate_inputs(
            text, source_topic, target_topic, style, num_samples
        )
        prompt = self.renderer.render_templated_generation(
            text=cleaned_text,
            source_topic=source_topic,
            target_topic=target_topic,
            style=style,
            num_samples=num_samples,
            language=language,
        )
        generation_kwargs = {
            "model": model,
            "temperature": self.TEMPERATURE if temperature is None else temperature,
            "max_tokens": max_tokens or max(4000, num_samples * 800),
            "response_format": {"type": "json_object"},
        }
        try:
            raw = self.llm.generate(prompt, **generation_kwargs)
        except TypeError as exc:
            if "response_format" not in str(exc):
                raise
            generation_kwargs.pop("response_format")
            raw = self.llm.generate(prompt, **generation_kwargs)
        self.last_response = raw

        samples = self._parse_samples(raw, limit=num_samples)
        records = [
            self._make_record(
                text=sample,
                index=idx,
                source_text=cleaned_text,
                source_topic=source_topic,
                target_topic=target_topic,
                style=style,
                language=language,
            )
            for idx, sample in enumerate(samples)
        ]
        return self._build_dataframe(records)

    @staticmethod
    def _validate_inputs(
        text: str,
        source_topic: str,
        target_topic: str,
        style: str,
        num_samples: int,
    ) -> tuple[str, str, str, str]:
        values = {
            "text": text,
            "source_topic": source_topic,
            "target_topic": target_topic,
            "style": style,
        }
        cleaned: dict[str, str] = {}
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"TemplatedGenerator requires a non-empty {name}.")
            cleaned[name] = value.strip()
        if num_samples <= 0:
            raise ValueError("num_samples must be greater than zero.")
        return (
            cleaned["text"],
            cleaned["source_topic"],
            cleaned["target_topic"],
            cleaned["style"],
        )

    @staticmethod
    def _parse_samples(raw: str, limit: int) -> list[str]:
        parsed = parse_json_response(raw)
        samples = TemplatedGenerator._candidate_samples(parsed)
        if not samples:
            logger.warning(
                "TemplatedGenerator: expected JSON with 'samples', 'texts', or 'text'."
            )
            return []

        texts: list[str] = []
        for item in samples:
            sample_text = item.get("text") if isinstance(item, dict) else item
            if not isinstance(sample_text, str):
                continue
            sample_text = clean_generated_text(sample_text)
            if sample_text:
                texts.append(sample_text)
            if len(texts) >= limit:
                break
        return texts

    @staticmethod
    def _candidate_samples(parsed: dict[str, Any]) -> list[Any]:
        samples = parsed.get("samples")
        if isinstance(samples, list):
            return samples
        if isinstance(samples, dict):
            return [samples]

        for key in ("texts", "generated_texts", "outputs"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value

        for key in ("text", "generated_text", "output"):
            value = parsed.get(key)
            if isinstance(value, str):
                return [value]

        sample = parsed.get("sample")
        if isinstance(sample, dict):
            return [sample]
        if isinstance(sample, str):
            return [sample]

        return []

    @staticmethod
    def _make_record(
        text: str,
        index: int,
        source_text: str,
        source_topic: str,
        target_topic: str,
        style: str,
        language: str,
    ) -> dict[str, Any]:
        source_text_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        return {
            "id": generate_sample_id("templated", index, "templated_generation"),
            "text": text,
            "source": "templated_generation",
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "language": language,
                "source_topic": source_topic,
                "target_topic": target_topic,
                "style": style,
                "source_text_hash": source_text_hash,
                "source_text_preview": source_text[:200],
            },
        }

    @staticmethod
    def _build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
        columns = ["id", "text", "source", "metadata"]
        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records)[columns]
