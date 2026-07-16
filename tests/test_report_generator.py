"""Tests for skillsieve.core.layer3.report_generator.generate_report."""
from __future__ import annotations

import pytest

from skillsieve.core.layer3.report_generator import generate_report
from skillsieve.models import LayerResult, Verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l1_safe() -> LayerResult:
    return LayerResult(
        layer=1,
        verdict=Verdict.SAFE,
        confidence=0.95,
        evidence=["no suspicious patterns found"],
        details={},
    )


def _l1_malicious() -> LayerResult:
    return LayerResult(
        layer=1,
        verdict=Verdict.MALICIOUS,
        confidence=0.85,
        evidence=["curl to evil.com", "base64-encoded payload"],
        details={},
    )


def _l2_malicious() -> LayerResult:
    return LayerResult(
        layer=2,
        verdict=Verdict.MALICIOUS,
        confidence=0.9,
        evidence=["intent misaligned with description", "covert exfiltration pattern"],
        details={"attack_types": ["data_exfil"]},
    )


def _l3_malicious() -> LayerResult:
    return LayerResult(
        layer=3,
        verdict=Verdict.MALICIOUS,
        confidence=0.92,
        evidence=["[Juror 1] subprocess call to curl evil.com"],
        details={
            "consensus": "unanimous",
            "round": 1,
            "recommend_human_review": False,
            "juror_results": [
                {
                    "verdict": "MALICIOUS",
                    "confidence": 0.95,
                    "reasoning": "Exfiltrates data via curl.",
                    "evidence": ["subprocess call to curl evil.com"],
                    "attack_types": ["data_exfil", "remote_exec"],
                },
                {
                    "verdict": "MALICIOUS",
                    "confidence": 0.91,
                    "reasoning": "Hidden network call detected.",
                    "evidence": ["network call in obfuscated code"],
                    "attack_types": ["data_exfil"],
                },
                {
                    "verdict": "MALICIOUS",
                    "confidence": 0.90,
                    "reasoning": "Credential theft pattern.",
                    "evidence": ["reads ~/.ssh/id_rsa"],
                    "attack_types": ["credential_theft"],
                },
            ],
        },
    )


# ---------------------------------------------------------------------------
# 1. Full MALICIOUS report
# ---------------------------------------------------------------------------

def test_generate_malicious_report():
    layer_results = [_l1_malicious(), _l2_malicious(), _l3_malicious()]
    report = generate_report("evil-skill", "1.0.0", layer_results)

    # Skill name and version
    assert "evil-skill" in report
    assert "1.0.0" in report

    # Verdict
    assert "MALICIOUS" in report

    # Attack Types section
    assert "Attack Types" in report
    assert "data_exfil" in report

    # Evidence Chain section
    assert "Evidence Chain" in report
    # Evidence from all three layers should appear
    assert "curl to evil.com" in report
    assert "covert exfiltration pattern" in report

    # Juror Opinions section
    assert "Juror Opinions" in report
    assert "Juror 1" in report
    assert "Juror 2" in report
    assert "Juror 3" in report

    # Recommended Action section
    assert "Recommended Action" in report
    assert "Block" in report or "quarantine" in report


# ---------------------------------------------------------------------------
# 2. SAFE report is brief (<500 chars)
# ---------------------------------------------------------------------------

def test_safe_report_is_brief():
    layer_results = [_l1_safe()]
    report = generate_report("nice-skill", "0.1.0", layer_results)

    assert "SAFE" in report
    assert "nice-skill" in report
    assert len(report) < 500


# ---------------------------------------------------------------------------
# 3. SAFE report does NOT contain juror/evidence chain sections
# ---------------------------------------------------------------------------

def test_safe_report_no_verbose_sections():
    layer_results = [_l1_safe()]
    report = generate_report("nice-skill", "0.1.0", layer_results)

    assert "Evidence Chain" not in report
    assert "Juror Opinions" not in report
    assert "Attack Types" not in report


# ---------------------------------------------------------------------------
# 4. MALICIOUS report with contested verdict recommends human review
# ---------------------------------------------------------------------------

def test_malicious_contested_recommends_human_review():
    l3 = LayerResult(
        layer=3,
        verdict=Verdict.MALICIOUS,
        confidence=0.55,
        evidence=["[Juror 1] suspicious pattern"],
        details={
            "consensus": "majority",
            "round": 2,
            "recommend_human_review": True,
            "juror_results": [
                {"verdict": "MALICIOUS", "confidence": 0.6, "reasoning": "Looks bad", "evidence": [], "attack_types": []},
            ],
        },
    )
    report = generate_report("maybe-evil", "2.0.0", [l3])
    assert "human review" in report.lower() or "escalate" in report.lower()
