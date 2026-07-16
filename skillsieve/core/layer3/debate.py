"""Layer 3 debate round: each juror sees the other jurors' reasoning and may revise."""
from __future__ import annotations

import asyncio
import json

from skillsieve.adapters.llm_adapter import LLMAdapter


def build_debate_prompt(own_result: dict, other_results: list[dict]) -> str:
    """Build a debate prompt for a single juror.

    The juror sees its own verdict / reasoning and the reasoning + evidence
    submitted by the other jurors, then is asked to MAINTAIN or CHANGE its
    verdict with an updated confidence score.

    Parameters
    ----------
    own_result:
        The dict returned by this juror in Round 1.  Expected keys:
        ``verdict``, ``confidence``, ``reasoning``, ``evidence``.
    other_results:
        The Round-1 dicts for the other jurors (same expected keys).

    Returns
    -------
    str
        A fully-formed prompt string ready to send to the LLM.
    """
    own_verdict = own_result.get("verdict", "UNKNOWN")
    own_confidence = own_result.get("confidence", 0.5)
    own_reasoning = own_result.get("reasoning", "")
    own_evidence = own_result.get("evidence", [])

    other_sections: list[str] = []
    for idx, other in enumerate(other_results, start=1):
        other_verdict = other.get("verdict", "UNKNOWN")
        other_confidence = other.get("confidence", 0.5)
        other_reasoning = other.get("reasoning", "")
        other_evidence = other.get("evidence", [])
        evidence_text = "\n".join(f"  - {e}" for e in other_evidence) or "  (none)"
        other_sections.append(
            f"Juror {idx}:\n"
            f"  Verdict: {other_verdict} (confidence {other_confidence})\n"
            f"  Reasoning: {other_reasoning}\n"
            f"  Evidence:\n{evidence_text}"
        )

    own_evidence_text = "\n".join(f"  - {e}" for e in own_evidence) or "  (none)"
    others_text = "\n\n".join(other_sections) or "(no other jurors)"

    return (
        "You are a security juror deliberating on whether a skill is MALICIOUS or SAFE.\n\n"
        "=== YOUR ROUND-1 VERDICT ===\n"
        f"Verdict: {own_verdict}\n"
        f"Confidence: {own_confidence}\n"
        f"Reasoning: {own_reasoning}\n"
        f"Evidence:\n{own_evidence_text}\n\n"
        "=== OTHER JURORS' ARGUMENTS ===\n"
        f"{others_text}\n\n"
        "Having reviewed the other jurors' reasoning and evidence, decide whether to "
        "MAINTAIN or CHANGE your verdict.\n\n"
        "Respond with a JSON object only (no extra text), with the following keys:\n"
        '  "verdict"     : "MALICIOUS" or "SAFE"\n'
        '  "confidence"  : float between 0.0 and 1.0\n'
        '  "reasoning"   : short explanation of your final position\n'
        '  "evidence"    : list of strings supporting your verdict\n'
        '  "changed"     : true if you changed your verdict, false otherwise\n'
    )


async def run_debate_round(
    juror_results: list[dict],
    adapters: list[LLMAdapter],
) -> list[dict]:
    """Run one debate round: each juror reconsiders given the others' arguments.

    All LLM calls are executed in parallel via :func:`asyncio.gather`.

    Parameters
    ----------
    juror_results:
        Round-1 results, one per juror.  Must be the same length as
        ``adapters``.
    adapters:
        One :class:`~skillsieve.adapters.llm_adapter.LLMAdapter` per juror.

    Returns
    -------
    list[dict]
        Updated juror results after the debate.  If an LLM call fails or
        returns un-parseable JSON the original Round-1 result is kept.
    """

    async def _debate_one(idx: int) -> dict:
        own = juror_results[idx]
        others = [r for i, r in enumerate(juror_results) if i != idx]
        prompt = build_debate_prompt(own, others)
        try:
            raw = await adapters[idx].complete(prompt=prompt)
            parsed = LLMAdapter.parse_json_response(raw)
            if parsed:
                # Preserve juror index metadata if present
                parsed.setdefault("verdict", own.get("verdict", "SAFE"))
                parsed.setdefault("confidence", own.get("confidence", 0.5))
                return parsed
        except Exception:
            pass
        # Fall back to original result
        return own

    updated = await asyncio.gather(*[_debate_one(i) for i in range(len(juror_results))])
    return list(updated)
