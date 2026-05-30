"""Command-line interface for synthetictext."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _load_task_from_file(path: str):  # type: ignore[return]
    from synthetictext.task import TaskSpec

    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML config files.") from exc
        with open(p) as f:
            data = yaml.safe_load(f)
    elif p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix} (use .json or .yaml)")
    return TaskSpec.from_dict(data)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="synthetictext",
        description="Generate synthetic text classification training data using LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Generate from a JSON task config
  synthetictext generate --config task.json --language en --num-samples 1000

  # Generate for all languages defined in the config
  synthetictext generate --config task.json --all --num-samples 500

  # Specify strategies and output directory
  synthetictext generate --config task.json -l en -n 1000 \\
      --strategies direct paraphrase contrastive \\
      --weights 0.5 0.3 0.2 \\
      --output-dir ./synthetic_data
""",
    )

    sub = parser.add_subparsers(dest="command")

    # --- generate subcommand ---
    gen = sub.add_parser("generate", help="Generate synthetic data")
    gen.add_argument(
        "--config", "-c", required=True, help="Path to task config file (JSON or YAML)"
    )

    lang_group = gen.add_mutually_exclusive_group(required=True)
    lang_group.add_argument("--language", "-l", help="Single language code")
    lang_group.add_argument("--all", "-a", action="store_true", help="All configured languages")

    gen.add_argument("--num-samples", "-n", type=int, default=1000)
    gen.add_argument(
        "--strategies",
        nargs="+",
        default=["direct"],
        choices=["direct", "paraphrase", "contrastive", "backtranslation", "pivot"],
    )
    gen.add_argument("--weights", nargs="+", type=float, default=None)
    gen.add_argument("--model", "-m", default=None, help="LLM model name")
    gen.add_argument("--provider", default="openai", help="LLM provider (default: openai)")
    gen.add_argument("--data-dir", default=None, help="Directory with original training CSVs")
    gen.add_argument("--label-column", default="label", help="Label column name in CSVs")
    gen.add_argument("--output-dir", "-o", default=None, help="Output directory")
    gen.add_argument("--no-filter", action="store_true", help="Disable quality filtering")

    # --- filter subcommand ---
    filt = sub.add_parser("filter", help="Filter an existing synthetic dataset")
    filt.add_argument("--input", "-i", required=True, help="Path to synthetic CSV")
    filt.add_argument("--output", "-o", default=None, help="Output path")
    filt.add_argument("--config", "-c", required=True, help="Task config file")
    filt.add_argument("--original-data", default=None, help="Original training CSV (for dedup)")
    filt.add_argument("--language", "-l", default="en")
    filt.add_argument("--no-embeddings", action="store_true")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        _run_generate(args)
    elif args.command == "filter":
        _run_filter(args)


def _run_generate(args: argparse.Namespace) -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    task = _load_task_from_file(args.config)

    from synthetictext.pipeline import SyntheticDataGenerator

    generator = SyntheticDataGenerator(
        task=task,
        llm_provider=args.provider,
        llm_model=args.model,
    )

    if args.language:
        original_df = None
        if args.data_dir:
            data_path = Path(args.data_dir) / f"{args.language}.csv"
            if data_path.exists():
                from synthetictext.utils import load_data

                original_df = load_data(data_path, label_column=args.label_column)

        save_path = None
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            save_path = str(Path(args.output_dir) / f"{args.language}_synthetic.csv")

        df = generator.generate(
            language=args.language,
            num_samples=args.num_samples,
            strategies=args.strategies,
            strategy_weights=args.weights,
            original_df=original_df,
            apply_filters=not args.no_filter,
            save_path=save_path,
        )
        print(f"\nGenerated {len(df)} samples for {args.language}")
        if len(df) > 0:
            print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    else:
        generator.generate_all(
            num_samples=args.num_samples,
            strategies=args.strategies,
            strategy_weights=args.weights,
            original_data_dir=args.data_dir,
            label_column=args.label_column,
            apply_filters=not args.no_filter,
            output_dir=args.output_dir,
        )


def _run_filter(args: argparse.Namespace) -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import pandas as pd

    from synthetictext.filters.pipeline import FilterPipeline
    from synthetictext.utils import save_data

    task = _load_task_from_file(args.config)
    synthetic_df = pd.read_csv(args.input)
    print(f"Loaded {len(synthetic_df)} samples from {args.input}")

    original_df = None
    if args.original_data:
        original_df = pd.read_csv(args.original_data)

    pipeline = FilterPipeline.default(task, use_embeddings=not args.no_embeddings)
    filtered = pipeline.run(synthetic_df, original_df=original_df, language=args.language)

    output = args.output or str(Path(args.input).with_suffix("")) + "_filtered.csv"
    save_data(filtered, output)
    print(f"Filtered: {len(synthetic_df)} -> {len(filtered)} samples")


if __name__ == "__main__":
    main()
