"""Layer 2 SemanticEngine: runs 4 LLM sub-tasks in parallel and aggregates results."""
from __future__ import annotations

import asyncio

from skillsieve.adapters.llm_adapter import LLMAdapter
from skillsieve.models import LayerResult, SkillPackage
from skillsieve.core.layer2.prompts import SYSTEM_PROMPT, TASK_NAMES, build_prompt
from skillsieve.core.layer2.aggregator import aggregate_subtasks


class SemanticEngine:
    """Run all 4 semantic analysis sub-tasks in parallel using an LLM.

    Parameters
    ----------
    model:
        LiteLLM model string to use.  Defaults to ``"deepseek/deepseek-chat"``.
    """

    def __init__(self, profile: str = "glm", model: str = "") -> None:
        self._adapter = LLMAdapter(profile=profile, model=model)

    async def analyze(
        self,
        pkg: SkillPackage,
        layer1_evidence: list[str],
    ) -> LayerResult:
        """Analyze a skill package with all 4 sub-tasks running in parallel.

        Parameters
        ----------
        pkg:
            The skill package to analyse.
        layer1_evidence:
            Evidence strings produced by the Layer 1 analysis pass.

        Returns
        -------
        LayerResult
            Aggregated Layer 2 result.
        """
        # Launch all 4 sub-tasks concurrently
        subtask_results = await asyncio.gather(
            *[
                self._run_subtask(task, pkg, layer1_evidence)
                for task in TASK_NAMES
            ]
        )

        # Map task names back to their results
        results: dict[str, dict] = {
            task: result for task, result in zip(TASK_NAMES, subtask_results)
        }

        return aggregate_subtasks(results)

    async def _run_subtask(
        self,
        task: str,
        pkg: SkillPackage,
        layer1_evidence: list[str],
    ) -> dict:
        """Run a single named sub-task and return the parsed JSON result.

        On JSON parse failure a default result with ``risk_score=0.5`` is
        returned so the aggregator can still produce a meaningful verdict.

        Parameters
        ----------
        task:
            Name of the sub-task (one of :data:`~skillsieve.core.layer2.prompts.TASK_NAMES`).
        pkg:
            The skill package being analysed.
        layer1_evidence:
            Layer 1 evidence strings passed through to the prompt.

        Returns
        -------
        dict
            Parsed LLM response or a default fallback dict.
        """
        prompt = build_prompt(
            task=task,
            skill_md=pkg.skill_md_content,
            script_contents=pkg.script_files,
            layer1_flags=layer1_evidence,
        )

        raw = await self._adapter.complete(prompt=prompt, system=SYSTEM_PROMPT)
        parsed = LLMAdapter.parse_json_response(raw)

        if not parsed:
            # Parse failure: return a cautious default
            return {
                "risk_score": 0.5,
                "evidence": [f"JSON parse failure for task {task!r}"],
            }

        return parsed
