#!/usr/bin/env python3
"""
SemEval-2026 Task 9: Multilingual Polarization Detection
=========================================================

This example recreates the synthetic data generation pipeline from the
PSK system paper (Pulipaka, SemEval-2026) using the synthetictext library.

Usage:
    python examples/polarization_detection.py --language eng --num-samples 1000
    python examples/polarization_detection.py --all --num-samples 500
"""
from __future__ import annotations

import argparse
import logging

from synthetictext import LanguageConfig, SyntheticDataGenerator, TaskSpec

POLARIZATION_TASK = TaskSpec(
    name="Multilingual Polarization Detection",
    labels={0: "non-polarized", 1: "polarized"},
    description=(
        "Classify social media text as polarized or non-polarized. "
        "Polarization encompasses stereotyping, vilification, dehumanization, "
        "deindividuation, or intolerance that incites division, groupism, "
        "hatred, or conflict between groups."
    ),
    label_descriptions={
        0: (
            "A post that discusses political, social, or cultural topics "
            "in a neutral or balanced way, without vilifying or dehumanizing "
            "anyone. May express opinions but without attacking an outgroup."
        ),
        1: (
            "A post that targets a specific group (ethnic, religious, "
            "political, social) using language that vilifies, stereotypes, "
            "or dehumanizes the outgroup. Shows blind support for an ingroup "
            "while attacking the outgroup with an us-vs-them mentality."
        ),
    },
    text_domain="social media post",
    word_count_range=(20, 80),
    topics={
        "political": [
            "government corruption",
            "election integrity",
            "political opponents",
            "party politics",
            "political protests",
            "democracy vs authoritarianism",
        ],
        "ethnic": [
            "immigration policy",
            "refugees and asylum",
            "ethnic minorities",
            "racial tensions",
            "indigenous rights",
            "multiculturalism",
        ],
        "religious": [
            "religious extremism",
            "secularism",
            "interfaith relations",
            "religious minorities",
            "religious nationalism",
            "religious education",
        ],
        "social": [
            "class inequality",
            "elite vs common people",
            "gender issues",
            "generational conflict",
            "urban vs rural divide",
            "education policy",
        ],
        "international": [
            "foreign intervention",
            "territorial disputes",
            "colonial history",
            "international organizations",
            "economic sanctions",
            "trade relations",
        ],
    },
    marker_keywords={
        1: [
            "animals", "vermin", "cockroaches", "parasites", "plague",
            "savages", "barbarians", "subhuman",
            "evil", "corrupt", "traitor", "enemy", "destroy", "eliminate",
            "scum", "filth", "garbage", "disgusting",
            "they want", "they always", "those people", "their kind",
            "our people", "our country", "unlike us",
            "agenda", "plot", "scheme", "puppet", "brainwashed", "sheeple",
            "fight back", "rise up", "take back", "send them back",
        ],
    },
)

SEMEVAL_LANGUAGES = LanguageConfig(
    languages={
        "eng": "English", "deu": "German", "spa": "Spanish", "ita": "Italian",
        "rus": "Russian", "zho": "Chinese", "pol": "Polish", "tur": "Turkish",
        "arb": "Arabic", "hin": "Hindi", "fas": "Persian", "urd": "Urdu",
        "ben": "Bengali", "amh": "Amharic", "swa": "Swahili", "hau": "Hausa",
        "nep": "Nepali", "pan": "Punjabi", "tel": "Telugu", "ori": "Oriya",
        "khm": "Khmer", "mya": "Burmese",
    },
    iso_map={
        "eng": "en", "deu": "de", "spa": "es", "ita": "it", "rus": "ru",
        "zho": "zh", "pol": "pl", "tur": "tr", "arb": "ar", "hin": "hi",
        "fas": "fa", "urd": "ur", "ben": "bn", "amh": "am", "swa": "sw",
        "hau": "ha", "nep": "ne", "pan": "pa", "tel": "te", "ori": "or",
        "khm": "km", "mya": "my",
    },
    tiers={
        1: ["eng", "deu", "spa", "ita", "rus", "zho", "pol", "tur"],
        2: ["arb", "hin", "fas", "urd", "ben"],
        3: ["amh", "swa", "hau", "nep", "pan", "tel", "ori", "khm", "mya"],
    },
    related_languages={
        "nep": "hin",
        "pan": "hin",
        "ori": "ben",
        "urd": "hin",
    },
    imbalanced={
        "amh": {"minority_class": 0, "ratio": 3.0},
        "hau": {"minority_class": 1, "ratio": 8.0},
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic polarization data")
    lang_group = parser.add_mutually_exclusive_group(required=True)
    lang_group.add_argument("--language", "-l", help="Language code (e.g. eng)")
    lang_group.add_argument("--all", "-a", action="store_true")

    parser.add_argument("--num-samples", "-n", type=int, default=1000)
    parser.add_argument("--output-dir", "-o", default="./synthetic_output")
    parser.add_argument("--data-dir", "-d", default=None,
                        help="Directory with original training CSVs for paraphrase strategy")
    parser.add_argument("--model", "-m", default="gpt-4o-mini")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    generator = SyntheticDataGenerator(
        task=POLARIZATION_TASK,
        llm_provider="openai",
        llm_model=args.model,
        lang_config=SEMEVAL_LANGUAGES,
    )

    strategies = ["direct", "paraphrase", "contrastive"]
    weights = [0.5, 0.3, 0.2]

    if args.language:
        original_df = None
        if args.data_dir:
            from pathlib import Path
            from synthetictext.utils import load_data

            p = Path(args.data_dir) / f"{args.language}.csv"
            if p.exists():
                original_df = load_data(p, label_column="polarization")

        df = generator.generate(
            language=args.language,
            num_samples=args.num_samples,
            strategies=strategies,
            strategy_weights=weights,
            original_df=original_df,
            save_path=f"{args.output_dir}/{args.language}_synthetic.csv",
        )
        print(f"\nGenerated {len(df)} samples for {args.language}")
        if len(df) > 0:
            print(f"Distribution: {df['label'].value_counts().to_dict()}")
    else:
        generator.generate_all(
            num_samples=args.num_samples,
            strategies=strategies,
            strategy_weights=weights,
            original_data_dir=args.data_dir,
            label_column="polarization",
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
