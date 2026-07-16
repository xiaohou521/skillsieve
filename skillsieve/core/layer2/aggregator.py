"""Layer 2 aggregator: combines results from all 4 semantic sub-tasks into a LayerResult."""
from __future__ import annotations

from skillsieve.models import LayerResult, Verdict

WEIGHTS: dict[str, float] = {
    "intent_alignment": 0.35,
    "permission_justification": 0.25,
    "covert_behavior": 0.25,
    "cross_file_consistency": 0.15,
}

THRESHOLD: float = 0.4


def aggregate_subtasks(
    results: dict[str, dict],
    threshold: float = THRESHOLD,
) -> LayerResult:
    """Combine sub-task results into a single :class:`LayerResult`.

    Parameters
    ----------
    results:
        Mapping of task name to the parsed JSON dict returned by the LLM for
        that task.  Each dict is expected to contain at least a ``risk_score``
        key (float 0-1) and an ``evidence`` key (list[str]).
    threshold:
        Weighted score threshold above which the verdict is ``SUSPICIOUS``.
        Defaults to :data:`THRESHOLD` (0.4).

    Returns
    -------
    LayerResult
        Layer 2 result with weighted confidence score and aggregated evidence.
    """
    weighted_score: float = 0.0
    total_weight: float = 0.0
    evidence: list[str] = []

    for task_name, weight in WEIGHTS.items():
        task_result = results.get(task_name, {})
        risk_score = float(task_result.get("risk_score", 0.0))

        weighted_score += risk_score * weight
        total_weight += weight

        # Collect and prefix evidence from this sub-task
        task_evidence = task_result.get("evidence", [])
        if isinstance(task_evidence, list):
            for item in task_evidence:
                evidence.append(f"[L2:{task_name}] {item}")

    # Normalise in case some tasks are missing (total_weight < 1.0)
    if total_weight > 0.0:
        confidence = weighted_score / total_weight
    else:
        confidence = 0.0

    verdict = Verdict.SUSPICIOUS if confidence >= threshold else Verdict.SAFE

    return LayerResult(
        layer=2,
        verdict=verdict,
        confidence=confidence,
        evidence=evidence,
        details={
            "subtask_scores": {
                task: results.get(task, {}).get("risk_score", 0.0)
                for task in WEIGHTS
            },
            "threshold": threshold,
            "weighted_score": weighted_score,
        },
    )
