"""Tests for safe JSON parsing utility."""

import pytest

from src.utils.json_parser import safe_parse_json


class TestSafeParseJson:

    def test_valid_json(self):
        result = safe_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_code_block(self):
        raw = '```json\n{"key": "value"}\n```'
        result = safe_parse_json(raw)
        assert result == {"key": "value"}

    def test_json_in_plain_code_block(self):
        raw = '```\n{"key": "value"}\n```'
        result = safe_parse_json(raw)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"key": "value"}\nDone.'
        result = safe_parse_json(raw)
        assert result == {"key": "value"}

    def test_empty_string_returns_none(self):
        assert safe_parse_json("") is None

    def test_whitespace_only_returns_none(self):
        assert safe_parse_json("   \n  ") is None

    def test_garbage_returns_none(self):
        assert safe_parse_json("not json at all") is None

    def test_strips_bom(self):
        raw = '\ufeff{"key": "value"}'
        result = safe_parse_json(raw)
        assert result == {"key": "value"}

    def test_nested_json_objects(self):
        raw = '{"users": {"1": {"facts": {"job": "dev"}}}}'
        result = safe_parse_json(raw)
        assert result["users"]["1"]["facts"]["job"] == "dev"
