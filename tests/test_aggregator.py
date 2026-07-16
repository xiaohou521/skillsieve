"""Tests for skillsieve.core.layer2.aggregator."""
from __future__ import annotations

import pytest

from skillsieve.core.layer2.aggregator import aggregate_subtasks, THRESHOLD
from skillsieve.models import Verdict


def _make_results(scores: dict[str, float]) -> dict[str, dict]:
    """Helper: build subtask result dicts from a task->score mapping."""
    return {
        task: {"risk_score": score, "evidence": [f"evidence for {task}"]}
        for task, score in scores.items()
    }


def test_aggregate_low_risk():
    """All low risk scores (0.1-0.2) should produce a SAFE verdict."""
    results = _make_results(
        {
            "intent_alignment": 0.1,
            "permission_justification": 0.2,
            "covert_behavior": 0.1,
            "cross_file_consistency": 0.2,
        }
    )
    layer_result = aggregate_subtasks(results)

    assert layer_result.layer == 2
    assert layer_result.verdict == Verdict.SAFE
    assert layer_result.confidence < THRESHOLD


def test_aggregate_high_risk():
    """All high risk scores (0.6-0.9) should produce a SUSPICIOUS verdict."""
    results = _make_results(
        {
            "intent_alignment": 0.9,
            "permission_justification": 0.7,
            "covert_behavior": 0.8,
            "cross_file_consistency": 0.6,
        }
    )
    layer_result = aggregate_subtasks(results)

    assert layer_result.layer == 2
    assert layer_result.verdict == Verdict.SUSPICIOUS
    assert layer_result.confidence >= THRESHOLD


def test_aggregate_partial_risk():
    """Mixed scores: 0.8*0.35 + 0.1*0.25 + 0.9*0.25 + 0.0*0.15 = 0.53 → SUSPICIOUS."""
    results = _make_results(
        {
            "intent_alignment": 0.8,       # weight 0.35  → 0.280
            "permission_justification": 0.1, # weight 0.25  → 0.025
            "covert_behavior": 0.9,         # weight 0.25  → 0.225
            "cross_file_consistency": 0.0,  # weight 0.15  → 0.000
        }
    )
    layer_result = aggregate_subtasks(results)

    assert layer_result.layer == 2
    assert layer_result.verdict == Verdict.SUSPICIOUS
    # Expected weighted score: 0.280 + 0.025 + 0.225 + 0.000 = 0.530
    assert abs(layer_result.confidence - 0.53) < 1e-9


def test_aggregate_evidence_prefixed():
    """Evidence items must be prefixed with [L2:task_name]."""
    results = {
        "intent_alignment": {"risk_score": 0.1, "evidence": ["claim mismatch"]},
        "permission_justification": {"risk_score": 0.1, "evidence": []},
        "covert_behavior": {"risk_score": 0.1, "evidence": []},
        "cross_file_consistency": {"risk_score": 0.1, "evidence": []},
    }
    layer_result = aggregate_subtasks(results)

    assert any("[L2:intent_alignment] claim mismatch" in e for e in layer_result.evidence)
