# Future Directions for synthetictext

Ideas for expanding the library, informed by our experience building and evaluating synthetic data pipelines for SemEval-2026 Task 9 (multilingual polarization detection across 22 languages).

---

## 1. Additional LLM Providers

Currently only OpenAI is shipped as a built-in provider. High-priority additions:

- **Anthropic (Claude)** -- straightforward to implement; the `BaseLLMProvider` interface maps directly to the Messages API.
- **LiteLLM integration** -- a single provider that proxies to 100+ backends (OpenAI, Anthropic, Cohere, local models via Ollama/vLLM, etc.). This would make synthetictext model-agnostic in one step.
- **Local model provider** -- wrap `transformers` or `vllm` for users who want to generate without API costs. Particularly relevant for languages where commercial APIs have weak coverage.
- **Batch API support** -- OpenAI and Anthropic both offer batch endpoints at 50% cost. For generating thousands of samples, this is significant. Would require an async generation mode.

## 2. Multi-Class and Multi-Label Support

The current library is optimized for binary classification but has the data structures for multi-class. Concrete next steps:

- **Multi-class contrastive generation** -- currently restricted to binary. Could generate N-way contrasts (one sample per class on the same topic) or pairwise contrasts between all class pairs.
- **Multi-label tasks** -- e.g., a sample can be both "toxic" and "political". Requires changes to `TaskSpec` (labels become sets), prompt templates, and filter logic.
- **Hierarchical labels** -- e.g., "hate speech > racial" vs "hate speech > religious". Would enable generating samples at different granularity levels.

## 3. Smarter Generation Strategies

### 3.1 Curriculum-Aware Generation
From our ablation study, we found that synthetic data helps most for low-resource languages (+3%) and least for high-resource ones (+1.9%). A curriculum-aware generator could:
- Estimate how much synthetic data each language/class needs based on training set size
- Automatically select optimal synthetic ratios (we found 30% was best on average, but it varied from 0% to 50% per language)
- Generate harder examples as training progresses (curriculum difficulty)

### 3.2 Topic-Conditioned Generation with Coverage Tracking
Our Italian results revealed a critical failure mode: the training data had zero examples for "political" and "other" topic categories, but these accounted for 41% of the test set. A coverage tracker could:
- Analyze existing training data to identify topic gaps
- Steer synthetic generation toward underrepresented topics
- Report coverage statistics after generation

### 3.3 Counterfactual Augmentation
Beyond simple contrastive pairs, generate minimal-edit counterfactuals:
- Take a real positive sample, change the minimum number of tokens to flip it to negative (and vice versa)
- This sharpens decision boundaries and was shown to improve robustness in hate speech detection (Mostafazadeh Davani et al., 2021)

### 3.4 Style Transfer
Generate synthetic samples that match the style distribution of the real training data:
- Extract style features (formality, length, vocabulary complexity) from real data
- Condition generation on these features
- Particularly useful when synthetic data sounds "too clean" compared to real social media text

## 4. Advanced Quality Filtering

### 4.1 Classifier-in-the-Loop Filtering
Train a lightweight classifier on the existing real data, then use it to score synthetic samples:
- Reject synthetic samples the classifier is already very confident about (they don't add information)
- Keep samples near the decision boundary (they're most informative)
- This is related to active learning and could significantly improve data efficiency

### 4.2 Calibration-Aware Generation
We found severe probability miscalibration in our models (Russian at mean prob 0.246, Khmer at 0.919). Synthetic data could be generated specifically to improve calibration:
- Oversample the probability ranges where the model is least calibrated
- Generate adversarial examples that target the model's weak spots

### 4.3 Cross-Lingual Quality Estimation
For low-resource languages, we can't easily judge generation quality. Ideas:
- Use multilingual embeddings to compare synthetic samples against a high-resource reference
- Detect code-switching or language contamination (GPT-4o-mini sometimes mixes English into low-resource outputs)
- Automated fluency scoring using perplexity from a multilingual language model

### 4.4 Diversity Metrics
Track and optimize for diversity across the generated dataset:
- Embedding-space coverage (are we generating in all regions of the semantic space?)
- Lexical diversity (type-token ratio, vocabulary breadth)
- Topic coverage heatmaps

## 5. Translation Provider Expansion

- **DeepL** -- higher quality than Google Translate for European languages
- **NLLB (No Language Left Behind)** -- Meta's open-source translation model covering 200 languages, runnable locally
- **Multi-provider routing** -- automatically select the best translation provider per language pair based on known quality benchmarks

## 6. Evaluation and Analysis Tools

The library currently generates and filters data but doesn't help users evaluate whether the synthetic data actually helps. Add:

- **Ablation runner** -- train a model with 0%, 10%, 20%, 30%, 50% synthetic ratios and compare validation F1 (this is what we did manually for 220 runs)
- **Distribution comparison** -- compare label distribution, text length distribution, vocabulary overlap between real and synthetic data
- **Embedding visualization** -- t-SNE/UMAP plots of real vs synthetic samples colored by label, to visually assess quality
- **Quality report** -- auto-generated HTML/markdown report summarizing generation statistics, filter drop rates, and quality metrics

## 7. Async and Parallel Generation

For large-scale generation (>10K samples, 20+ languages), the current synchronous approach is slow:
- **Async API calls** -- use `asyncio` with `openai`'s async client for 5-10x throughput
- **Rate limiter** -- built-in token-bucket rate limiting to stay within API quotas
- **Parallel language generation** -- generate for multiple languages concurrently
- **Checkpoint/resume** -- save progress periodically so generation can resume after interruption

## 8. Data Versioning and Reproducibility

- **Generation manifests** -- record every parameter (model, temperature, prompts, random seed) used to generate each sample, enabling exact reproduction
- **Deterministic mode** -- set all random seeds for reproducible generation (within the limits of LLM stochasticity)
- **Integration with DVC** -- version synthetic datasets alongside model training code

## 9. Domain-Specific Extensions

Package pre-built `TaskSpec` configurations for common NLP tasks:

- **Hate speech / toxicity detection** (with Jigsaw-style multi-label: toxic, severe_toxic, obscene, threat, insult, identity_hate)
- **Sentiment analysis** (binary, 3-class, 5-class)
- **Stance detection** (favor, against, neutral)
- **Misinformation / fake news detection**
- **Emotion classification** (Ekman 6, GoEmotions 27)
- **Propaganda detection** (with technique-level labels)

These would ship as `synthetictext.presets` and provide ready-to-use configurations with curated topic lists and marker keywords.

## 10. Research Directions

### 10.1 When Does Synthetic Data Help?
Our findings suggest:
- Model capacity matters more than data augmentation (+2.7% from 12B vs 2B, vs +2-3% from synthetic data)
- Low-resource languages benefit most
- Some languages are actively hurt by synthetic data (Swahili, Khmer at high ratios)

The library could include benchmarking tools to systematically study these effects across tasks and languages.

### 10.2 Cross-Architecture Generalization
We found that XLM-RoBERTa and Qwen3 showed strong dev performance but catastrophic test failure (30-50% F1 drops), while Gemma generalized reliably. Investigating whether synthetic data quality/style interacts with architecture choice is an open question.

### 10.3 Differential Privacy
Generate synthetic data with formal privacy guarantees (building on work by SynthTextEval). This would make the library suitable for sensitive domains like healthcare and finance.

---

## Contributing

If you're interested in working on any of these directions, please open an issue to discuss the approach before submitting a PR. We especially welcome contributions for new LLM providers, translation backends, and domain-specific presets.
