"""Default prompt templates for every generation strategy.

Each template uses ``str.format`` placeholders.  Available variables:

* ``{language}``           -- full language name (e.g. "English")
* ``{task_name}``          -- from ``TaskSpec.name``
* ``{task_description}``   -- from ``TaskSpec.description``
* ``{label_name}``         -- short label name for the target class
* ``{label_description}``  -- detailed description of the target class
* ``{other_label_name}``   -- short label name for the opposing class (binary)
* ``{other_label_description}`` -- description of the opposing class (binary)
* ``{topic}``              -- a randomly sampled topic hint (may be empty)
* ``{text_domain}``        -- e.g. "social media post"
* ``{word_min}`` / ``{word_max}`` -- word-count range
* ``{text}``               -- the source text (paraphrasing / contrastive)
* ``{chunk}``              -- source chunk for RAG Q&A generation
* ``{num_samples}``        -- number of Q&A pairs to generate
"""

DIRECT_GENERATION_PROMPT = """\
You are generating synthetic training data for {task_name} in {language}.

Task: {task_description}

Generate a realistic {text_domain} in {language} that should be classified as "{label_name}".
{label_description}

Requirements:
- Feel authentic to {language} discourse
- Be {word_min}-{word_max} words in length
- Topic hint: {topic}

IMPORTANT: Generate ONLY the {text_domain} text in {language}.
Do NOT include labels, explanations, translations, or meta-commentary.
Do NOT start with phrases like "Here is" or "Post:".
"""

PARAPHRASE_PROMPT = """\
Rewrite the following {language} text using different words and phrasing.

Task context: {task_description}

Requirements:
- Keep the EXACT same meaning and sentiment
- Maintain the same classification (if it is "{label_name}", stay "{label_name}")
- Use natural {language} expressions
- Similar length to original (within 20% word count)

Original text:
{text}

IMPORTANT: Output ONLY the rewritten text in {language}. No explanations or meta-commentary.
"""

CONTRASTIVE_PAIR_PROMPT = """\
Generate two versions of a {language} {text_domain} about: {topic}

Task: {task_description}

VERSION A ("{label_name}"):
{label_description}

VERSION B ("{other_label_name}"):
{other_label_description}

Both versions should:
- Be similar in length ({word_min}-{word_max} words)
- Discuss the exact same topic/event
- Feel authentic to {language} discourse
- Be written entirely in {language}

Output format (use exactly this format):
Respond with ONLY a JSON object (no markdown formatting):
{{"{label_name}": "[text in {language}]", "{other_label_name}": "[text in {language}]"}}
"""

LLM_JUDGE_PROMPT = """\
You are evaluating synthetic training data for {task_name}.

Task: {task_description}

Text: {text}
Assigned Label: {label_name}
Language: {language}

Evaluate the following:
1. Is this text realistic for {text_domain} in this language? (yes/no)
2. Does the assigned label correctly match the text content? (yes/no)
3. Is the text clear and unambiguous for this classification? (yes/no)
4. Is the text grammatically acceptable? (yes/no)

Respond with ONLY a JSON object (no markdown formatting):
{{"realistic": true/false, "label_correct": true/false,
"clear": true/false, "grammatical": true/false}}
"""

RAG_QA_PROMPT = """\
You are generating evaluation data for a retrieval-augmented generation (RAG) system.

Create {num_samples} question-answer pairs in {language} that are answerable ONLY from the source chunk below.

Requirements:
- Questions must test specific facts, relationships, or details present in the chunk
- Answers must be concise and fully grounded in the chunk
- Do not use outside knowledge
- Do not create questions that cannot be answered from the chunk
- Do not quote large parts of the chunk unless necessary

Source chunk:
{chunk}

Respond with ONLY a JSON object (no markdown formatting):
{{"qa_pairs": [{{"question": "...", "answer": "..."}}]}}
"""

DEFAULT_PROMPTS = {
    "direct": DIRECT_GENERATION_PROMPT,
    "paraphrase": PARAPHRASE_PROMPT,
    "contrastive": CONTRASTIVE_PAIR_PROMPT,
    "judge": LLM_JUDGE_PROMPT,
    "rag_qa": RAG_QA_PROMPT,
}
