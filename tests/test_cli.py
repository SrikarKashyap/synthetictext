"""Tests for command-line entrypoints."""
from __future__ import annotations

import pandas as pd


def test_rag_qa_cli_writes_output(tmp_path, monkeypatch) -> None:
    from synthetictext.cli import main

    class FakeRAGQAGenerator:
        def __init__(self, llm_provider="openai", llm_model=None) -> None:
            self.llm_provider = llm_provider
            self.llm_model = llm_model

        def generate(self, chunk: str, num_samples: int, language: str) -> pd.DataFrame:
            assert chunk == "Source chunk text."
            assert num_samples == 1
            assert language == "English"
            return pd.DataFrame([
                {
                    "id": "ragqa_1",
                    "question": "What is this?",
                    "answer": "Source chunk text.",
                    "source": "rag_qa",
                    "metadata": {"language": language},
                }
            ])

    monkeypatch.setattr("synthetictext.qa.RAGQAGenerator", FakeRAGQAGenerator)

    input_path = tmp_path / "chunk.txt"
    output_path = tmp_path / "qa.csv"
    input_path.write_text("Source chunk text.", encoding="utf-8")

    main([
        "rag-qa",
        "--input",
        str(input_path),
        "--num-samples",
        "1",
        "--language",
        "English",
        "--output",
        str(output_path),
    ])

    df = pd.read_csv(output_path)
    assert df.loc[0, "question"] == "What is this?"
    assert df.loc[0, "answer"] == "Source chunk text."
