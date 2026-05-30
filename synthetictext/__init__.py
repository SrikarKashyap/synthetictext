"""synthetictext -- LLM-powered synthetic text data generation for classification tasks.

Quick start::

    from synthetictext import TaskSpec, SyntheticDataGenerator

    task = TaskSpec(
        name="Sentiment Analysis",
        labels={0: "negative", 1: "positive"},
        description="Classify product reviews as positive or negative.",
    )

    generator = SyntheticDataGenerator(task=task, llm_provider="openai")
    df = generator.generate(language="English", num_samples=100)
"""

__version__ = "0.1.0"

from synthetictext.pipeline import SyntheticDataGenerator
from synthetictext.providers.base import BaseLLMProvider, BaseTranslationProvider
from synthetictext.task import LanguageConfig, TaskSpec

__all__ = [
    "TaskSpec",
    "LanguageConfig",
    "SyntheticDataGenerator",
    "BaseLLMProvider",
    "BaseTranslationProvider",
    "__version__",
]
