"""Tests for skillsieve.core.layer2.semantic_engine.SemanticEngine."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from skillsieve.core.layer2.semantic_engine import SemanticEngine
from skillsieve.core.layer2.prompts import TASK_NAMES
from skillsieve.models import LayerResult, SkillPackage, SkillFrontmatter, Verdict


@pytest.fixture
def simple_package() -> SkillPackage:
    """A minimal SkillPackage for testing."""
    return SkillPackage(
        path=Path("/tmp/test-skill"),
        name="test-skill",
        frontmatter=SkillFrontmatter(
            name="test-skill",
            description="A simple test skill",
        ),
        skill_md_content="# Test Skill\nDoes a simple thing.",
        script_files={"main.py": "print('hello world')"},
    )


@pytest.mark.asyncio
async def test_semantic_engine_analyzes_all_four_tasks(simple_package):
    """SemanticEngine must call the LLM for each of the 4 sub-tasks and return a LayerResult."""
    safe_response = '{"risk_score": 0.1, "evidence": ["nothing suspicious"]}'

    with patch(
        "skillsieve.core.layer2.semantic_engine.LLMAdapter.complete",
        new_callable=AsyncMock,
        return_value=safe_response,
    ):
        engine = SemanticEngine(model="deepseek/deepseek-chat")
        result = await engine.analyze(
            pkg=simple_package,
            layer1_evidence=["no flags"],
        )

    # Must return a LayerResult for layer 2
    assert isinstance(result, LayerResult)
    assert result.layer == 2

    # All 4 tasks produced low scores → SAFE verdict
    assert result.verdict == Verdict.SAFE

    # Details must record all 4 subtask scores
    subtask_scores = result.details.get("subtask_scores", {})
    assert set(subtask_scores.keys()) == set(TASK_NAMES)
    for task in TASK_NAMES:
        assert abs(subtask_scores[task] - 0.1) < 1e-9

    # Evidence must be present (prefixed from all tasks)
    assert len(result.evidence) == 4  # one evidence item per task
    for task in TASK_NAMES:
        assert any(f"[L2:{task}]" in e for e in result.evidence)
