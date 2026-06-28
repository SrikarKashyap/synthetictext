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


def test_templated_cli_writes_output(tmp_path, monkeypatch) -> None:
    from synthetictext.cli import main

    class FakeTemplatedGenerator:
        def __init__(self, llm_provider="openai", llm_model=None) -> None:
            self.llm_provider = llm_provider
            self.llm_model = llm_model

        def generate(
            self,
            text: str,
            source_topic: str,
            target_topic: str,
            style: str,
            num_samples: int,
            language: str,
        ) -> pd.DataFrame:
            assert text == "Source text."
            assert source_topic == "electric vehicles"
            assert target_topic == "urban gardening"
            assert style == "brief note"
            assert num_samples == 1
            assert language == "English"
            return pd.DataFrame([
                {
                    "id": "templated_1",
                    "text": "Generated garden text.",
                    "source": "templated_generation",
                    "metadata": {"language": language},
                }
            ])

    monkeypatch.setattr("synthetictext.templated.TemplatedGenerator", FakeTemplatedGenerator)

    input_path = tmp_path / "source.txt"
    output_path = tmp_path / "templated.csv"
    input_path.write_text("Source text.", encoding="utf-8")

    main([
        "templated",
        "--input",
        str(input_path),
        "--source-topic",
        "electric vehicles",
        "--target-topic",
        "urban gardening",
        "--style",
        "brief note",
        "--num-samples",
        "1",
        "--language",
        "English",
        "--output",
        str(output_path),
    ])

    df = pd.read_csv(output_path)
    assert df.loc[0, "text"] == "Generated garden text."
    assert df.loc[0, "source"] == "templated_generation"
