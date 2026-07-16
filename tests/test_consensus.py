"""Tests for skillsieve.core.layer3.consensus.compute_consensus."""
from __future__ import annotations

import pytest

from skillsieve.core.layer3.consensus import compute_consensus


def _make_results(*verdicts: str) -> list[dict]:
    return [{"verdict": v, "confidence": 0.9} for v in verdicts]


# ---------------------------------------------------------------------------
# 1. Unanimous MALICIOUS
# ---------------------------------------------------------------------------

def test_unanimous_malicious():
    results = _make_results("MALICIOUS", "MALICIOUS", "MALICIOUS")
    out = compute_consensus(results, round_num=1)

    assert out["verdict"] == "MALICIOUS"
    assert out["consensus"] == "unanimous"
    assert out["round"] == 1
    assert not out.get("needs_debate")


# ---------------------------------------------------------------------------
# 2. Unanimous SAFE
# ---------------------------------------------------------------------------

def test_unanimous_safe():
    results = _make_results("SAFE", "SAFE", "SAFE")
    out = compute_consensus(results, round_num=1)

    assert out["verdict"] == "SAFE"
    assert out["consensus"] == "unanimous"
    assert out["round"] == 1
    assert not out.get("needs_debate")


# ---------------------------------------------------------------------------
# 3. Round-2 majority MALICIOUS (2 MALICIOUS + 1 SAFE)
# ---------------------------------------------------------------------------

def test_majority_malicious():
    results = _make_results("MALICIOUS", "MALICIOUS", "SAFE")
    out = compute_consensus(results, round_num=2)

    assert out["verdict"] == "MALICIOUS"
    assert out["consensus"] == "majority"
    assert out["round"] == 2
    assert not out.get("needs_debate")
    assert not out.get("recommend_human_review")


# ---------------------------------------------------------------------------
# 4. Round-2 2 SAFE + 1 MALICIOUS → held MALICIOUS + human review
# ---------------------------------------------------------------------------

def test_no_consensus_escalates():
    results = _make_results("MALICIOUS", "SAFE", "SAFE")
    out = compute_consensus(results, round_num=2)

    assert out["verdict"] == "MALICIOUS"
    assert out["consensus"] == "contested"
    assert out["round"] == 2
    assert out.get("recommend_human_review") is True
    assert not out.get("needs_debate")


# ---------------------------------------------------------------------------
# Bonus: Round-1 split triggers needs_debate
# ---------------------------------------------------------------------------

def test_round1_split_needs_debate():
    results = _make_results("MALICIOUS", "SAFE", "SAFE")
    out = compute_consensus(results, round_num=1)

    assert out.get("needs_debate") is True
    assert "juror_results" in out
    assert out["juror_results"] == results
