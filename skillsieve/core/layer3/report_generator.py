"""Layer 3 report generator: produce human-readable scan reports."""
from __future__ import annotations

from skillsieve.models import LayerResult, Verdict


def generate_report(
    skill_name: str,
    version: str,
    layer_results: list[LayerResult],
) -> str:
    """Generate a scan report from the collected layer results.

    For a **SAFE** final verdict a brief 4-line summary is returned.
    For a **MALICIOUS** final verdict a full report is produced including:

    * Attack Types (collected from Layer 3 juror results)
    * Evidence Chain (all evidence from every layer, numbered)
    * Juror Opinions (from ``details["juror_results"]`` in the Layer 3 result)
    * Recommended Action

    The final verdict is taken from the *last* LayerResult in
    ``layer_results``; if the list is empty, SAFE is assumed.

    Parameters
    ----------
    skill_name:
        The name of the skill being reported on.
    version:
        The version string of the skill.
    layer_results:
        Ordered list of :class:`~skillsieve.models.LayerResult` objects
        (Layer 1 first, Layer 3 last if present).

    Returns
    -------
    str
        The formatted report string.
    """
    final_verdict = layer_results[-1].verdict if layer_results else Verdict.SAFE

    if final_verdict != Verdict.MALICIOUS:
        return _safe_report(skill_name, version, layer_results)

    return _malicious_report(skill_name, version, layer_results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_report(
    skill_name: str,
    version: str,
    layer_results: list[LayerResult],
) -> str:
    layers_checked = ", ".join(f"L{r.layer}" for r in layer_results) or "none"
    avg_confidence = (
        sum(r.confidence for r in layer_results) / len(layer_results)
        if layer_results
        else 0.0
    )
    return (
        f"SkillSieve Scan Report\n"
        f"Skill: {skill_name} v{version}\n"
        f"Verdict: SAFE (confidence {avg_confidence:.2f})\n"
        f"Layers checked: {layers_checked}\n"
    )


def _malicious_report(
    skill_name: str,
    version: str,
    layer_results: list[LayerResult],
) -> str:
    lines: list[str] = [
        "SkillSieve Scan Report",
        f"Skill: {skill_name} v{version}",
        f"Verdict: MALICIOUS",
        "",
    ]

    # --- Attack Types ---
    attack_types: list[str] = _collect_attack_types(layer_results)
    lines.append("Attack Types:")
    if attack_types:
        for at in attack_types:
            lines.append(f"  - {at}")
    else:
        lines.append("  (unknown)")
    lines.append("")

    # --- Evidence Chain ---
    lines.append("Evidence Chain:")
    evidence_num = 1
    for lr in layer_results:
        for item in lr.evidence:
            lines.append(f"  {evidence_num}. [L{lr.layer}] {item}")
            evidence_num += 1
    if evidence_num == 1:
        lines.append("  (no evidence recorded)")
    lines.append("")

    # --- Juror Opinions ---
    layer3_result = next((r for r in layer_results if r.layer == 3), None)
    lines.append("Juror Opinions:")
    if layer3_result is not None:
        juror_results: list[dict] = layer3_result.details.get("juror_results", [])
        if juror_results:
            for idx, jr in enumerate(juror_results, start=1):
                verdict = jr.get("verdict", "UNKNOWN")
                confidence = jr.get("confidence", 0.0)
                reasoning = jr.get("reasoning", "")
                lines.append(
                    f"  Juror {idx}: {verdict} (confidence {confidence:.2f}) — {reasoning}"
                )
        else:
            lines.append("  (no juror details available)")
    else:
        lines.append("  (no Layer 3 result)")
    lines.append("")

    # --- Recommended Action ---
    recommend_human = (
        layer3_result is not None
        and layer3_result.details.get("recommend_human_review", False)
    )
    lines.append("Recommended Action:")
    if recommend_human:
        lines.append(
            "  Contested verdict — escalate to human review before taking action."
        )
    else:
        lines.append(
            "  Block / quarantine this skill immediately. Do not install or execute."
        )
    lines.append("")

    return "\n".join(lines)


def _collect_attack_types(layer_results: list[LayerResult]) -> list[str]:
    """Gather unique attack type strings from all layer results."""
    seen: set[str] = set()
    result: list[str] = []

    for lr in layer_results:
        # From details directly
        for at in lr.details.get("attack_types", []):
            at_str = str(at)
            if at_str not in seen:
                seen.add(at_str)
                result.append(at_str)

        # From juror_results inside Layer 3 details
        for jr in lr.details.get("juror_results", []):
            for at in jr.get("attack_types", []):
                at_str = str(at)
                if at_str not in seen:
                    seen.add(at_str)
                    result.append(at_str)

    return result
