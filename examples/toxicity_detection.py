#!/usr/bin/env python3
"""
Toxicity Detection -- Simple Example
=====================================

Generates synthetic training data for a binary toxicity classifier.

Usage:
    python examples/toxicity_detection.py
"""
from __future__ import annotations

import logging

from synthetictext import SyntheticDataGenerator, TaskSpec

task = TaskSpec(
    name="Toxicity Detection",
    labels={0: "non-toxic", 1: "toxic"},
    description="Classify social media comments as toxic or non-toxic.",
    label_descriptions={
        0: (
            "A comment that expresses opinions, asks questions, or discusses "
            "topics without attacking, insulting, or threatening anyone."
        ),
        1: (
            "A comment containing insults, slurs, threats, harassment, "
            "or dehumanizing language targeting individuals or groups."
        ),
    },
    text_domain="social media comment",
    word_count_range=(10, 60),
    topics={
        "political": ["elections", "government policy", "political debate"],
        "social": ["gender issues", "immigration", "inequality"],
        "tech": ["AI regulation", "social media moderation"],
    },
    marker_keywords={
        1: [
            "idiot", "stupid", "moron", "trash", "scum",
            "kill", "die", "hate", "disgusting", "pathetic",
        ],
    },
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    generator = SyntheticDataGenerator(
        task=task,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )

    df = generator.generate(
        language="English",
        num_samples=100,
        strategies=["direct", "contrastive"],
        strategy_weights=[0.6, 0.4],
    )

    print(f"\nGenerated {len(df)} samples")
    print(f"Distribution: {df['label'].value_counts().to_dict()}")
    print(f"\nSample outputs:")
    for _, row in df.head(5).iterrows():
        label_name = task.label_name(int(row["label"]))
        print(f"  [{label_name}] {row['text'][:80]}...")


if __name__ == "__main__":
    main()
