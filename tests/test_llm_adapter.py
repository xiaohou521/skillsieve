"""Tests for LLMAdapter — no real API calls; only construction and parse_json_response."""
import pytest

from skillsieve.adapters.llm_adapter import LLMAdapter


def test_adapter_creation():
    adapter = LLMAdapter(profile="kimi")
    assert adapter.model == "kimi-2.5"
    assert adapter.api_type == "anthropic"


def test_adapter_minimax():
    adapter = LLMAdapter(profile="minimax")
    assert adapter.model == "MiniMax-M2.7"
    assert adapter.api_type == "openai"


def test_adapter_baidu():
    adapter = LLMAdapter(profile="baidu")
    assert adapter.model == "deepseek-v3.1-250821"
    assert adapter.api_type == "openai"


def test_parse_json_response():
    raw = "```json\n{\"verdict\": \"MALICIOUS\"}\n```"
    result = LLMAdapter.parse_json_response(raw)
    assert result == {"verdict": "MALICIOUS"}


def test_parse_json_plain():
    raw = '{"verdict": "SAFE", "confidence": 0.9}'
    result = LLMAdapter.parse_json_response(raw)
    assert result == {"verdict": "SAFE", "confidence": 0.9}


def test_parse_json_fallback():
    raw = "Not valid JSON"
    result = LLMAdapter.parse_json_response(raw)
    assert result == {}
