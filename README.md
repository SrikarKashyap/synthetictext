# synthetictext

LLM-powered synthetic text data generation for text classification tasks.

`synthetictext` generates high-quality synthetic training data for any text classification task across multiple languages. It provides five generation strategies, a multi-stage quality filtering pipeline, and a simple Python API.

## Features

- **Task-agnostic**: Define any binary or multi-class text classification task via a `TaskSpec`
- **5 generation strategies**: Direct generation, paraphrasing, contrastive pairs, backtranslation, and pivot translation
- **Multi-stage quality filtering**: Deduplication, label leakage detection, embedding-based dedup, LLM-as-judge, and keyword marker checks
- **Multilingual**: Generate data in any language supported by your LLM provider, with optional cross-lingual transfer for low-resource languages
- **Provider-agnostic**: Built-in support for OpenAI; extensible via `BaseLLMProvider` and `BaseTranslationProvider` interfaces
- **CLI and Python API**: Use from scripts or the command line

## Installation

```bash
# Core (no LLM provider included)
pip install synthetictext

# With OpenAI support (most common)
pip install synthetictext[openai]

# With embedding-based deduplication
pip install synthetictext[embeddings]

# With Google Cloud Translation (for backtranslation/pivot strategies)
pip install synthetictext[google-translate]

# With YAML config file support for the CLI
pip install synthetictext[yaml]

# Everything
pip install synthetictext[all]
```

## Authentication

The library needs an API key for the LLM provider. There are three ways to provide it:

**Option 1: Environment variable (recommended)**

```bash
export OPENAI_API_KEY="sk-..."
```

Then just use the shorthand -- the OpenAI SDK picks up the env var automatically:

```python
generator = SyntheticDataGenerator(task=task, llm_provider="openai")
```

**Option 2: Explicit API key**

```python
generator = SyntheticDataGenerator(
    task=task,
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    api_key="sk-...",
)
```

**Option 3: Direct provider construction** (full control)

```python
from synthetictext.providers.openai_provider import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...", default_model="gpt-4o")
generator = SyntheticDataGenerator(task=task, llm_provider=provider)
```

The model defaults to `gpt-4o-mini` but can be overridden via `llm_model` or `--model` on the CLI.

## Quick Start

```python
from synthetictext import TaskSpec, SyntheticDataGenerator

# 1. Define your classification task
task = TaskSpec(
    name="Sentiment Analysis",
    labels={0: "negative", 1: "positive"},
    description="Classify product reviews as positive or negative sentiment.",
    label_descriptions={
        0: "A review expressing dissatisfaction, criticism, or negative experience.",
        1: "A review expressing satisfaction, praise, or positive experience.",
    },
    topics={
        "electronics": ["smartphones", "laptops", "headphones"],
        "food": ["restaurants", "delivery", "recipes"],
    },
)

# 2. Create a generator
generator = SyntheticDataGenerator(
    task=task,
    llm_provider="openai",  # uses OPENAI_API_KEY env var
    llm_model="gpt-4o-mini",
)

# 3. Generate data
df = generator.generate(
    language="English",
    num_samples=1000,
    strategies=["direct", "paraphrase", "contrastive"],
    strategy_weights=[0.5, 0.3, 0.2],
)

print(df.head())
# Columns: id, text, label, source, generated_at, language
```

## Generation Strategies

| Strategy | Description | Requires |
|----------|-------------|----------|
| **direct** | Generate new samples in the target language | LLM |
| **paraphrase** | Rewrite existing samples preserving labels | LLM + training data |
| **contrastive** | Generate minimal pairs (one per class, same topic) | LLM |
| **backtranslation** | Round-trip translate through a pivot language | Translation API + training data |
| **pivot** | Generate in English, translate to target language | LLM + Translation API |

## Quality Filtering

The default filtering pipeline runs these steps in order:

1. **BasicFilter** -- remove empty, null, or out-of-range-length samples
2. **LeakageFilter** -- remove samples containing label leakage patterns (e.g., "this is a positive example")
3. **EmbeddingDeduplicator** -- remove near-duplicates using multilingual sentence embeddings (cosine similarity > 0.90)
4. **MarkerFilter** -- (optional) ensure samples for specific labels contain expected keywords

Additional optional filters:

- **LLMJudgeFilter** -- use an LLM to validate realism, label correctness, clarity, and grammar
- **TranslationQualityFilter** -- check round-trip translation consistency for pivot/backtranslation samples

## Multilingual Generation

```python
from synthetictext import LanguageConfig

lang_config = LanguageConfig(
    languages={"en": "English", "de": "German", "es": "Spanish"},
    related_languages={"es": "en"},  # cross-lingual transfer
)

generator = SyntheticDataGenerator(
    task=task,
    llm_provider="openai",
    lang_config=lang_config,
)

# Generate for all configured languages
results = generator.generate_all(num_samples=500)
```

## CLI Usage

```bash
# Generate from a JSON task config
synthetictext generate --config task.json --language en --num-samples 1000

# Generate for all languages
synthetictext generate --config task.json --all --num-samples 500 --output-dir ./output

# Multiple strategies with weights
synthetictext generate --config task.json -l en -n 1000 \
    --strategies direct paraphrase contrastive \
    --weights 0.5 0.3 0.2

# Filter existing synthetic data
synthetictext filter --config task.json --input synthetic.csv --output filtered.csv
```

### Task Config File (JSON)

```json
{
    "name": "Toxicity Detection",
    "labels": {"0": "non-toxic", "1": "toxic"},
    "description": "Classify social media posts as toxic or non-toxic.",
    "label_descriptions": {
        "0": "A post that discusses topics respectfully.",
        "1": "A post containing insults, threats, or dehumanizing language."
    },
    "text_domain": "social media post",
    "word_count_range": [20, 80],
    "topics": {
        "political": ["elections", "government policy"],
        "social": ["gender issues", "immigration"]
    }
}
```

## Custom Providers

Extend `BaseLLMProvider` to use any LLM backend:

```python
from synthetictext.providers.base import BaseLLMProvider

class AnthropicProvider(BaseLLMProvider):
    def generate(self, prompt, *, model=None, temperature=0.9,
                 max_tokens=250, system_prompt=None):
        # Your Anthropic API call here
        ...
```

## Tutorials

Step-by-step Jupyter notebooks in [`tutorials/`](tutorials/):

1. **[Quick Start](tutorials/01_quickstart.ipynb)** -- defining tasks, generating data, combining strategies
2. **[Quality Filtering](tutorials/02_quality_filtering.ipynb)** -- using built-in filters, building custom filter pipelines
3. **[Multilingual Generation](tutorials/03_multilingual_generation.ipynb)** -- `LanguageConfig`, backtranslation, pivot translation, tier-based strategies

## Examples

- [`examples/polarization_detection.py`](examples/polarization_detection.py) -- Full recreation of the SemEval-2026 Task 9 pipeline (22 languages)
- [`examples/toxicity_detection.py`](examples/toxicity_detection.py) -- Simple binary toxicity classifier data generation

## Development

```bash
git clone https://github.com/srikarkashyap/synthetictext.git
cd synthetictext
pip install -e ".[dev]"
pytest
```

## Origin

This library was developed from the synthetic data pipeline used in the PSK system for [SemEval-2026 Task 9: Multilingual Polarization Detection](https://github.com/srikarkashyap/synthetictext), where it generated training data across 22 languages and contributed to a 2nd-place finish.

## License

MIT
