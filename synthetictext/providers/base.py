"""Abstract base classes for LLM and translation providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """Interface for text-generation backends (OpenAI, Anthropic, local, etc.)."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.9,
        max_tokens: int = 250,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text from a prompt.

        Returns:
            The generated text string.
        """
        ...


class BaseTranslationProvider(ABC):
    """Interface for machine translation backends."""

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate *text* from *source_lang* to *target_lang*.

        Language codes should be ISO 639-1 (e.g., ``"en"``, ``"de"``).
        """
        ...

    def backtranslate(
        self,
        text: str,
        source_lang: str,
        pivot_lang: str = "en",
    ) -> str:
        """Round-trip translate: source -> pivot -> source."""
        pivot_text = self.translate(text, source_lang, pivot_lang)
        return self.translate(pivot_text, pivot_lang, source_lang)
