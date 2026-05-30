# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-30

### Added

- `TaskSpec` and `LanguageConfig` dataclasses for defining classification tasks and multilingual configurations
- Five generation strategies: direct, paraphrase, contrastive, backtranslation, and pivot translation
- `SyntheticDataGenerator` orchestrator with `generate()` and `generate_all()` methods
- Quality filtering pipeline with composable filters:
  - `BasicFilter` (empty/short/invalid text)
  - `LeakageFilter` (label leakage detection)
  - `EmbeddingDeduplicator` (semantic near-duplicate removal)
  - `MarkerFilter` (keyword presence validation)
  - `LLMJudgeFilter` (LLM-based quality scoring)
  - `TranslationQualityFilter` (round-trip translation consistency)
- Provider abstractions: `BaseLLMProvider` and `BaseTranslationProvider`
- Built-in providers: OpenAI (LLM) and Google Cloud Translate (translation)
- Configurable prompt templates with `PromptRenderer`
- CLI with `generate` and `filter` subcommands
- Task configuration via JSON/YAML files
- Tutorials and examples for sentiment analysis, toxicity detection, and polarization detection
