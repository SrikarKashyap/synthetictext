"""Tests for OpenAI provider compatibility behavior."""
from __future__ import annotations

from types import SimpleNamespace

from synthetictext.providers.openai_provider import OpenAIProvider


class MaxTokensFakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "max_tokens" in kwargs:
            raise Exception(
                "Unsupported parameter: 'max_tokens' is not supported with this model. "
                "Use 'max_completion_tokens' instead."
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Generated response")
                )
            ]
        )


def test_openai_provider_falls_back_to_max_completion_tokens() -> None:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    completions = MaxTokensFakeCompletions()
    provider._client = SimpleNamespace(  # type: ignore[attr-defined]
        chat=SimpleNamespace(completions=completions)
    )
    provider.default_model = "gpt-5.5"

    result = provider.generate("Prompt", max_tokens=123)

    assert result == "Generated response"
    assert completions.calls[0]["max_tokens"] == 123
    assert "max_completion_tokens" not in completions.calls[0]
    assert completions.calls[1]["max_completion_tokens"] == 123
    assert "max_tokens" not in completions.calls[1]


class TemperatureFakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "temperature" in kwargs and kwargs["temperature"] != 1:
            raise Exception(
                "Unsupported value: 'temperature' does not support 0.7 with this model. "
                "Only the default (1) value is supported."
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Generated response")
                )
            ]
        )


def test_openai_provider_drops_unsupported_temperature() -> None:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    completions = TemperatureFakeCompletions()
    provider._client = SimpleNamespace(  # type: ignore[attr-defined]
        chat=SimpleNamespace(completions=completions)
    )
    provider.default_model = "gpt-5.5"

    result = provider.generate("Prompt", temperature=0.7)

    assert result == "Generated response"
    assert completions.calls[0]["temperature"] == 0.7
    assert "temperature" not in completions.calls[1]
