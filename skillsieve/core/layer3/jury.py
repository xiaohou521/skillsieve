"""Layer 3 Jury: three LLM jurors deliberate independently, then debate if split."""
from __future__ import annotations

import asyncio

from skillsieve.adapters.llm_adapter import LLMAdapter
from skillsieve.models import LayerResult, SkillPackage, Verdict
from skillsieve.core.layer3.consensus import compute_consensus
from skillsieve.core.layer3.debate import run_debate_round

_DEFAULT_PROFILES: list[str] = ["glm", "qwen", "baidu"]

_JUROR_SYSTEM = (
    "You are a security expert reviewing an AI agent skill for malicious behaviour. "
    "Analyse the provided evidence and return a JSON verdict."
)

_JUROR_PROMPT_TEMPLATE = """\
You are an independent juror deciding whether an AI skill package is malicious.

Layer~1 (regex/AST) and Layer~2 (LLM semantic analysis) have already analysed this
skill and flagged it as SUSPICIOUS, but with a Layer~2 aggregate risk of {layer2_confidence:.2f}
on a scale of 0--1 (0.40 is the escalation threshold; higher means more confident
it is malicious). Your task is to independently re-evaluate the skill, not to
confirm earlier layers. Many legitimate crypto, DeFi, identity, and developer
tools trigger these rules for domain-inherent reasons (handling private keys,
making API calls, reading config files). Flag the skill as MALICIOUS only if
its behaviour cannot be explained by a legitimate purpose that matches its
stated description.

## Skill metadata
Name: {name}
Description: {description}

## Layer 1 static-analysis evidence
{layer1_evidence}

## Layer 2 semantic-analysis evidence (aggregate risk {layer2_confidence:.2f})
{layer2_evidence}

## Skill content (SKILL.md excerpt)
{skill_md}

## Script files
{scripts}

Before voting, consider:
1. What is the most plausible legitimate purpose of this skill?
2. Are the flagged patterns consistent with that legitimate purpose, or do they
   indicate covert behaviour (credential exfiltration, hidden C2, runtime code
   download, social-engineering instructions, permission bypass)?
3. Would a user of the stated domain (e.g., a DeFi trader, a DID developer)
   expect to see these patterns?

Only vote MALICIOUS if the skill does something that a legitimate version of
the stated tool would not do.

Respond with JSON only (no extra text):
{{
  "verdict": "MALICIOUS" or "SAFE",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<brief explanation>",
  "evidence": ["<evidence string>", ...],
  "attack_types": ["<attack type if malicious>", ...]
}}
"""


def _build_juror_prompt(
    pkg: SkillPackage,
    layer1_evidence: list[str],
    layer2_evidence: list[str],
    layer2_confidence: float = 0.5,
) -> str:
    MAX_MD = 4000
    MAX_SCRIPT = 2000
    MAX_TOTAL = 8000

    scripts_parts = []
    total = 0
    for fname, content in pkg.script_files.items():
        if total >= MAX_TOTAL:
            scripts_parts.append(f"### {fname}\n(skipped)")
            continue
        text = content[:MAX_SCRIPT] + "\n... [truncated]" if len(content) > MAX_SCRIPT else content
        scripts_parts.append(f"### {fname}\n```\n{text}\n```")
        total += len(text)
    scripts_text = "\n\n".join(scripts_parts) if scripts_parts else "(no scripts)"

    skill_md = pkg.skill_md_content or "(empty)"
    if len(skill_md) > MAX_MD:
        skill_md = skill_md[:MAX_MD] + "\n... [truncated]"

    return _JUROR_PROMPT_TEMPLATE.format(
        name=pkg.frontmatter.name or pkg.name,
        description=pkg.frontmatter.description or "(no description)",
        layer1_evidence="\n".join(f"- {e}" for e in layer1_evidence) or "(none)",
        layer2_evidence="\n".join(f"- {e}" for e in layer2_evidence) or "(none)",
        layer2_confidence=float(layer2_confidence),
        skill_md=skill_md,
        scripts=scripts_text,
    )


def _parse_juror_response(raw: str, fallback_verdict: str = "SAFE") -> dict:
    """Parse a juror's raw LLM response, filling defaults for missing fields."""
    parsed = LLMAdapter.parse_json_response(raw)
    parsed.setdefault("verdict", fallback_verdict)
    parsed.setdefault("confidence", 0.5)
    parsed.setdefault("reasoning", "")
    parsed.setdefault("evidence", [])
    parsed.setdefault("attack_types", [])
    return parsed


class Jury:
    """Three-LLM jury that can run a debate round when the initial vote is split.

    Parameters
    ----------
    models:
        List of LiteLLM model strings.  Defaults to GPT-4o, Claude Sonnet,
        and DeepSeek Chat.
    """

    def __init__(self, profiles: list[str] | None = None, models: list[str] | None = None) -> None:
        profs = profiles if profiles is not None else list(_DEFAULT_PROFILES)
        self._adapters: list[LLMAdapter] = [
            LLMAdapter(profile=p) for p in profs
        ]

    async def adjudicate(
        self,
        pkg: SkillPackage,
        layer1_evidence: list[str],
        layer2_evidence: list[str],
        layer2_confidence: float = 0.5,
    ) -> LayerResult:
        """Run the full jury process and return a :class:`LayerResult`.

        1. All jurors independently analyse the evidence (parallel).
        2. :func:`compute_consensus` is called for Round 1.
        3. If split, a debate round is run and consensus is re-evaluated.
        4. The final verdict string is converted to a :class:`Verdict` enum.

        Parameters
        ----------
        pkg:
            The skill package to evaluate.
        layer1_evidence:
            Evidence strings from Layer 1 static analysis.
        layer2_evidence:
            Evidence strings from Layer 2 semantic analysis.

        Returns
        -------
        LayerResult
            Layer 3 result carrying the jury verdict, confidence, evidence, and
            all juror details.
        """
        prompt = _build_juror_prompt(pkg, layer1_evidence, layer2_evidence, layer2_confidence)

        # Round 1: all jurors deliberate independently in parallel
        raw_responses: list[str] = await asyncio.gather(
            *[adapter.complete(prompt=prompt, system=_JUROR_SYSTEM) for adapter in self._adapters]
        )

        juror_results: list[dict] = [
            _parse_juror_response(raw) for raw in raw_responses
        ]

        # Round 1 consensus
        consensus = compute_consensus(juror_results, round_num=1)

        if consensus.get("needs_debate"):
            # Debate round
            juror_results = await run_debate_round(juror_results, self._adapters)
            consensus = compute_consensus(juror_results, round_num=2)

        # Map verdict string → Verdict enum (default to SAFE on unknown)
        raw_verdict = str(consensus.get("verdict", "SAFE")).upper()
        try:
            verdict = Verdict(raw_verdict.lower())
        except ValueError:
            verdict = Verdict.SAFE

        # Aggregate confidence: mean of juror confidences
        confidence = sum(r.get("confidence", 0.5) for r in juror_results) / len(juror_results)

        # Aggregate evidence from all jurors
        evidence: list[str] = []
        for idx, r in enumerate(juror_results):
            for item in r.get("evidence", []):
                evidence.append(f"[Juror {idx + 1}] {item}")

        details: dict = {
            "consensus": consensus.get("consensus", ""),
            "round": consensus.get("round", 1),
            "juror_results": juror_results,
            "recommend_human_review": consensus.get("recommend_human_review", False),
        }

        return LayerResult(
            layer=3,
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            details=details,
        )
