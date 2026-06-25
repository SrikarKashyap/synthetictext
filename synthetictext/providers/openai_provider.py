"""OpenAI LLM provider implementation."""
from __future__ import annotations

from typing import Optional

from synthetictext.providers.base import BaseLLMProvider
from synthetictext.utils import retry_with_backoff


class OpenAIProvider(BaseLLMProvider):
    """LLM provider backed by the OpenAI chat completions API.

    Requires ``pip install synthetictext[openai]``.

    Args:
        api_key: OpenAI API key. If *None*, falls back to the
            ``OPENAI_API_KEY`` environment variable.
        default_model: Model name used when callers don't specify one.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for OpenAIProvider. "
                "Install it with: pip install synthetictext[openai]"
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self.default_model = default_model

    @retry_with_backoff(max_retries=3)
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.9,
        max_tokens: int = 250,
        system_prompt: Optional[str] = None,
        response_format: Optional[dict[str, str]] = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
