"""Task specification for synthetic text classification data generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TaskSpec:
    """
    Defines a text classification task for synthetic data generation.

    This is the central abstraction: instead of hardcoding prompts for a specific
    task (e.g., polarization detection), users describe their task and the library
    generates appropriate prompts and data.

    Args:
        name: Human-readable task name (e.g., "Toxicity Detection").
        labels: Mapping from label index to short label name.
            Example: ``{0: "non-toxic", 1: "toxic"}``
        description: One-sentence description of the classification task.
        label_descriptions: Optional per-label description of what qualifies.
            Example: ``{0: "A post that discusses topics respectfully.", ...}``
        topics: Optional mapping of topic categories to specific topic strings.
            Used to steer diverse generation across subject matter.
        text_domain: Domain of the text being generated (default ``"social media post"``).
        language_name: Default language for monolingual generation (default ``"English"``).
        word_count_range: ``(min, max)`` word count range for generated samples.
        marker_keywords: Optional per-label keyword lists used by the marker filter.
            Example: ``{1: ["slur", "threat", "insult"]}``
        custom_prompts: Optional dict overriding default prompt templates.
            Keys are strategy names (``"direct"``, ``"paraphrase"``, ``"contrastive"``,
            ``"judge"``). Values are format-string templates.
    """

    name: str
    labels: Dict[int, str]
    description: str
    label_descriptions: Optional[Dict[int, str]] = None
    topics: Optional[Dict[str, List[str]]] = None
    text_domain: str = "social media post"
    language_name: str = "English"
    word_count_range: tuple[int, int] = (20, 80)
    marker_keywords: Optional[Dict[int, List[str]]] = None
    custom_prompts: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if len(self.labels) < 2:
            raise ValueError("TaskSpec requires at least 2 labels.")
        if self.label_descriptions is None:
            self.label_descriptions = {}

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def is_binary(self) -> bool:
        return self.num_classes == 2

    @property
    def label_indices(self) -> List[int]:
        return sorted(self.labels.keys())

    def label_name(self, index: int) -> str:
        return self.labels[index]

    def label_description(self, index: int) -> str:
        if self.label_descriptions and index in self.label_descriptions:
            return self.label_descriptions[index]
        return self.labels[index]

    def random_topic(self) -> Optional[str]:
        """Return a random topic string, or None if no topics are configured."""
        if not self.topics:
            return None
        import random

        category = random.choice(list(self.topics.keys()))
        return random.choice(self.topics[category])

    def get_prompt(self, strategy: str) -> Optional[str]:
        """Return a user-supplied custom prompt for a strategy, or None."""
        if self.custom_prompts and strategy in self.custom_prompts:
            return self.custom_prompts[strategy]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "labels": self.labels,
            "description": self.description,
            "label_descriptions": self.label_descriptions,
            "topics": self.topics,
            "text_domain": self.text_domain,
            "language_name": self.language_name,
            "word_count_range": list(self.word_count_range),
            "marker_keywords": self.marker_keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSpec":
        labels = {int(k): v for k, v in data["labels"].items()}
        label_descs = None
        if data.get("label_descriptions"):
            label_descs = {int(k): v for k, v in data["label_descriptions"].items()}
        marker_kw = None
        if data.get("marker_keywords"):
            marker_kw = {int(k): v for k, v in data["marker_keywords"].items()}
        wc = tuple(data["word_count_range"]) if "word_count_range" in data else (20, 80)
        return cls(
            name=data["name"],
            labels=labels,
            description=data["description"],
            label_descriptions=label_descs,
            topics=data.get("topics"),
            text_domain=data.get("text_domain", "social media post"),
            language_name=data.get("language_name", "English"),
            word_count_range=wc,  # type: ignore[arg-type]
            marker_keywords=marker_kw,
            custom_prompts=data.get("custom_prompts"),
        )


@dataclass
class LanguageConfig:
    """
    Configuration for multilingual synthetic data generation.

    Args:
        languages: Mapping of language codes to full names.
            Example: ``{"en": "English", "de": "German"}``
        iso_map: Optional mapping of user language codes to ISO 639-1 codes
            used by translation APIs. Defaults to identity mapping.
        tiers: Optional grouping of languages by resource level.
            Example: ``{1: ["en", "de"], 2: ["ar", "hi"], 3: ["am", "sw"]}``
        related_languages: Optional mapping for cross-lingual transfer.
            Keys are target language codes, values are source language codes.
            Example: ``{"ne": "hi"}`` (Nepali <-- Hindi)
        imbalanced: Optional per-language class imbalance info.
            Example: ``{"am": {"minority_class": 0, "ratio": 3.0}}``
    """

    languages: Dict[str, str]
    iso_map: Optional[Dict[str, str]] = None
    tiers: Optional[Dict[int, List[str]]] = None
    related_languages: Optional[Dict[str, str]] = None
    imbalanced: Optional[Dict[str, Dict[str, Any]]] = None

    def iso_code(self, lang_code: str) -> str:
        """Get the ISO 639-1 code for translation APIs."""
        if self.iso_map and lang_code in self.iso_map:
            return self.iso_map[lang_code]
        return lang_code

    def language_name(self, lang_code: str) -> str:
        return self.languages.get(lang_code, lang_code)

    def tier_languages(self, tier: int) -> List[str]:
        if self.tiers and tier in self.tiers:
            return self.tiers[tier]
        return []

    def all_language_codes(self) -> List[str]:
        return list(self.languages.keys())

    def get_related_language(self, lang_code: str) -> Optional[str]:
        if self.related_languages and lang_code in self.related_languages:
            return self.related_languages[lang_code]
        return None

    def get_imbalance_info(self, lang_code: str) -> Optional[Dict[str, Any]]:
        if self.imbalanced and lang_code in self.imbalanced:
            return self.imbalanced[lang_code]
        return None

    @classmethod
    def single(cls, code: str = "en", name: str = "English") -> "LanguageConfig":
        """Create a minimal single-language config."""
        return cls(languages={code: name})
