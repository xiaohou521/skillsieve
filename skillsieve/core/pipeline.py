"""Core three-layer pipeline orchestrator for SkillSieve."""
from __future__ import annotations

from pathlib import Path

from skillsieve.core.layer1.classifier import Layer1Classifier
from skillsieve.core.layer2.semantic_engine import SemanticEngine
from skillsieve.core.layer3.jury import Jury
from skillsieve.core.layer3.report_generator import generate_report
from skillsieve.models import LayerResult, ScanResult, Verdict
from skillsieve.parser import parse_skill_package


class Pipeline:
    """Orchestrates Layer 1 → Layer 2 → Layer 3 analysis of a skill package.

    Parameters
    ----------
    max_layer:
        Maximum layer to run (1, 2, or 3).  Useful for testing or cost control.
    layer1_threshold:
        Suspicion threshold passed to :class:`Layer1Classifier`.
    layer2_profile:
        LLM profile for :class:`SemanticEngine` (kimi/minimax/baidu).
    layer3_profiles:
        List of LLM profiles for the :class:`Jury`.  ``None`` uses
        the jury's built-in defaults (kimi, minimax, baidu).
    """

    def __init__(
        self,
        max_layer: int = 3,
        layer1_threshold: float = 0.3,
        layer2_profile: str = "kimi",
        layer3_profiles: list[str] | None = None,
    ) -> None:
        self._max_layer = max_layer
        self._l1 = Layer1Classifier(threshold=layer1_threshold)
        self._l2 = SemanticEngine(profile=layer2_profile)
        self._l3 = Jury(profiles=layer3_profiles)

    async def scan(self, skill_path: Path) -> ScanResult:
        """Scan a skill package directory through the configured pipeline layers.

        Parameters
        ----------
        skill_path:
            Path to the skill package directory (must contain a ``SKILL.md``).

        Returns
        -------
        ScanResult
            The aggregated scan result carrying the final verdict, confidence,
            the layer at which the pipeline stopped, and all per-layer results.
        """
        pkg = parse_skill_package(Path(skill_path))
        layer_results: list[LayerResult] = []

        # --- Layer 1: static heuristic classifier ---
        l1_result = self._l1.classify(pkg)
        layer_results.append(l1_result)
        if l1_result.verdict == Verdict.SAFE or self._max_layer < 2:
            return self._build_result(pkg, layer_results)

        # --- Layer 2: LLM-based semantic engine ---
        l2_result = await self._l2.analyze(pkg, layer1_evidence=l1_result.evidence)
        layer_results.append(l2_result)
        if l2_result.verdict == Verdict.SAFE or self._max_layer < 3:
            return self._build_result(pkg, layer_results)

        # --- Layer 3: multi-model jury ---
        l3_result = await self._l3.adjudicate(
            pkg,
            layer1_evidence=l1_result.evidence,
            layer2_evidence=l2_result.evidence,
            layer2_confidence=l2_result.confidence,
        )
        layer_results.append(l3_result)
        return self._build_result(pkg, layer_results)

    @staticmethod
    def _build_result(pkg, layer_results: list[LayerResult]) -> ScanResult:
        """Assemble the final :class:`ScanResult` from accumulated layer results.

        A ``SUSPICIOUS`` verdict at the final (maximum) layer is treated as
        ``SAFE`` — there is not enough evidence to escalate further.
        A ``MALICIOUS`` verdict triggers report generation.
        """
        final = layer_results[-1]
        final_verdict = final.verdict

        # At the last layer, SUSPICIOUS is escalated to MALICIOUS
        # (enough layers have flagged it as risky)
        if final_verdict == Verdict.SUSPICIOUS:
            final_verdict = Verdict.MALICIOUS

        report: str | None = None
        if final_verdict == Verdict.MALICIOUS:
            report = generate_report(
                pkg.name,
                pkg.frontmatter.version,
                layer_results,
            )

        return ScanResult(
            skill_name=pkg.name,
            final_verdict=final_verdict,
            final_confidence=final.confidence,
            layer_stopped=final.layer,
            layer_results=layer_results,
            report=report,
        )
