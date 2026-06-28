"""RAG question-answer sample generation."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from synthetictext.pipeline import _resolve_llm_provider
from synthetictext.prompts.templates import PromptRenderer
from synthetictext.providers.base import BaseLLMProvider
from synthetictext.utils import generate_sample_id, logger, parse_json_response


class RAGQAGenerator:
    """Generate grounded question-answer pairs from a source text chunk."""

    TEMPERATURE = 0.2

    def __init__(
        self,
        llm_provider: BaseLLMProvider | str = "openai",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        renderer: Optional[PromptRenderer] = None,
    ) -> None:
        self.llm = _resolve_llm_provider(llm_provider, llm_model, api_key)
        self.renderer = renderer or PromptRenderer()

    def generate(
        self,
        chunk: str,
        num_samples: int,
        language: str = "English",
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate RAG evaluation Q&A pairs from a single text chunk.

        Returns a DataFrame with columns ``id``, ``question``, ``answer``,
        ``source``, and ``metadata``.
        """
        cleaned_chunk = self._validate_inputs(chunk, num_samples)
        prompt = self.renderer.render_rag_qa(cleaned_chunk, num_samples, language)
        generation_kwargs = {
            "model": model,
            "temperature": self.TEMPERATURE if temperature is None else temperature,
            "max_tokens": max_tokens or max(4000, num_samples * 500),
            "response_format": {"type": "json_object"},
        }
        try:
            raw = self.llm.generate(prompt, **generation_kwargs)
        except TypeError as exc:
            if "response_format" not in str(exc):
                raise
            generation_kwargs.pop("response_format")
            raw = self.llm.generate(prompt, **generation_kwargs)
        pairs = self._parse_pairs(raw, limit=num_samples)
        records = [
            self._make_record(pair["question"], pair["answer"], idx, cleaned_chunk, language)
            for idx, pair in enumerate(pairs)
        ]
        return self._build_dataframe(records)

    @staticmethod
    def _validate_inputs(chunk: str, num_samples: int) -> str:
        if not isinstance(chunk, str) or not chunk.strip():
            raise ValueError("RAGQAGenerator requires a non-empty text chunk.")
        if num_samples <= 0:
            raise ValueError("num_samples must be greater than zero.")
        return chunk.strip()

    @staticmethod
    def _parse_pairs(raw: str, limit: int) -> list[dict[str, str]]:
        parsed = parse_json_response(raw)
        qa_pairs = parsed.get("qa_pairs", [])
        if not isinstance(qa_pairs, list):
            logger.warning("RAGQAGenerator: expected 'qa_pairs' to be a list.")
            return []

        pairs: list[dict[str, str]] = []
        for item in qa_pairs:
            if not isinstance(item, dict):
                continue
            question = item.get("question")
            answer = item.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                continue
            question = question.strip()
            answer = answer.strip()
            if question and answer:
                pairs.append({"question": question, "answer": answer})
            if len(pairs) >= limit:
                break
        return pairs

    @staticmethod
    def _make_record(
        question: str,
        answer: str,
        index: int,
        chunk: str,
        language: str,
    ) -> dict[str, Any]:
        chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        return {
            "id": generate_sample_id("ragqa", index, "rag_qa"),
            "question": question,
            "answer": answer,
            "source": "rag_qa",
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "language": language,
                "chunk_hash": chunk_hash,
                "chunk_preview": chunk[:200],
            },
        }

    @staticmethod
    def _build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
        columns = ["id", "question", "answer", "source", "metadata"]
        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records)[columns]
