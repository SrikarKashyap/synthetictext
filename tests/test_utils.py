"""Tests for utility functions."""
from __future__ import annotations

from synthetictext.utils import clean_generated_text, generate_sample_id, parse_json_response


class TestCleanGeneratedText:
    def test_strips_whitespace(self) -> None:
        assert clean_generated_text("  hello  ") == "hello"

    def test_removes_common_prefix(self) -> None:
        assert clean_generated_text("Here is a great post") == "a great post"

    def test_removes_prefix_with_colon(self) -> None:
        assert clean_generated_text("Text: The actual content") == "The actual content"

    def test_removes_surrounding_quotes(self) -> None:
        assert clean_generated_text('"Hello world"') == "Hello world"

    def test_empty_input(self) -> None:
        assert clean_generated_text("") == ""
        assert clean_generated_text(None) == ""  # type: ignore[arg-type]


class TestParseJsonResponse:
    def test_direct_json(self) -> None:
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced(self) -> None:
        text = '```json\n{"a": true}\n```'
        assert parse_json_response(text) == {"a": True}

    def test_json_in_text(self) -> None:
        text = 'Some text before {"x": 1} and after'
        assert parse_json_response(text) == {"x": 1}

    def test_nested_json_in_markdown_fence(self) -> None:
        text = '```json\n{"qa_pairs": [{"question": "Q?", "answer": "A."}]}\n```'
        assert parse_json_response(text) == {
            "qa_pairs": [{"question": "Q?", "answer": "A."}]
        }

    def test_invalid_returns_empty(self) -> None:
        assert parse_json_response("not json at all") == {}

    def test_top_level_array_returns_empty(self) -> None:
        assert parse_json_response('["not", "an", "object"]') == {}


class TestGenerateSampleId:
    def test_format(self) -> None:
        sid = generate_sample_id("en", 0, "direct")
        assert sid.startswith("en_syn_")
        assert len(sid) > 10

    def test_unique(self) -> None:
        ids = {generate_sample_id("en", i, "test") for i in range(50)}
        assert len(ids) == 50
