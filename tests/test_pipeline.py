"""Tests for skillsieve.core.pipeline.Pipeline."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from skillsieve.core.pipeline import Pipeline
from skillsieve.models import Verdict

FIXTURES = Path(__file__).parent / "fixtures"


def test_pipeline_benign_stops_at_layer1():
    """A benign skill should be caught safe at Layer 1 and not escalate."""
    pipeline = Pipeline()
    result = asyncio.run(pipeline.scan(FIXTURES / "benign_skill"))
    assert result.final_verdict == Verdict.SAFE
    assert result.layer_stopped == 1
    assert len(result.layer_results) == 1


def test_pipeline_malicious_escalates():
    """A malicious skill should escalate all three layers and return MALICIOUS."""
    l2_response = json.dumps({
        "risk_score": 0.9,
        "evidence": ["very suspicious"],
    })
    l3_response = json.dumps({
        "verdict": "MALICIOUS",
        "confidence": 0.95,
        "reasoning": "clearly malicious",
        "evidence": ["bad"],
        "attack_types": ["credential_theft"],
    })

    call_count = {"n": 0}

    async def _side_effect(prompt: str, system: str = "") -> str:
        call_count["n"] += 1
        # Layer 2 uses prompts with task titles like "Intent Alignment Analysis"
        if any(
            kw in prompt
            for kw in [
                "Intent Alignment",
                "Permission Justification",
                "Covert Behavior",
                "Cross-File Consistency",
            ]
        ):
            return l2_response
        # Layer 3 juror prompts
        return l3_response

    with patch(
        "skillsieve.adapters.llm_adapter.LLMAdapter.complete",
        new_callable=AsyncMock,
        side_effect=_side_effect,
    ):
        pipeline = Pipeline(layer3_profiles=["kimi", "minimax", "baidu"])
        result = asyncio.run(pipeline.scan(FIXTURES / "malicious_skill"))

    assert result.final_verdict == Verdict.MALICIOUS
    assert result.layer_stopped == 3
    assert len(result.layer_results) == 3
    assert result.report is not None


def test_pipeline_max_layer():
    """When max_layer=1 the pipeline must not escalate even for a malicious skill."""
    pipeline = Pipeline(max_layer=1)
    result = asyncio.run(pipeline.scan(FIXTURES / "malicious_skill"))
    assert result.layer_stopped == 1
    assert len(result.layer_results) == 1
