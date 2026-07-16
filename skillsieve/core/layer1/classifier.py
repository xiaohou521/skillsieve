"""Layer 1 XGBoost Classifier.

Assembles features from all 4 Layer 1 modules (PatternMatcher, ASTAnalyzer,
MetadataChecker, SkillMdAnalyzer) and classifies using XGBoost when a trained
model is available, or a heuristic weighted score as fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from skillsieve.models import LayerResult, SkillPackage, Verdict
from skillsieve.core.layer1.pattern_matcher import PatternMatcher
from skillsieve.core.layer1.ast_analyzer import ASTAnalyzer
from skillsieve.core.layer1.metadata_checker import MetadataChecker
from skillsieve.core.layer1.skillmd_analyzer import SkillMdAnalyzer

_TOP_NAMES_PATH = Path(__file__).resolve().parent.parent.parent / "rules" / "top_skill_names.json"


def _load_top_skill_names() -> list[str]:
    """Load the bundled top skill names list for typosquatting detection."""
    if _TOP_NAMES_PATH.exists():
        return json.loads(_TOP_NAMES_PATH.read_text())
    return []


class Layer1Classifier:
    """Classify a SkillPackage using assembled features from all 4 Layer 1 modules.

    Parameters
    ----------
    model_path:
        Optional path to a saved XGBoost model file. When ``None`` (or the
        path does not exist), the classifier falls back to the heuristic scorer.
    threshold:
        Probability threshold above which a package is classified as
        ``SUSPICIOUS``.  Default is ``0.3``.
    top_skill_names:
        List of popular/known-good skill names passed to ``MetadataChecker``
        for typosquatting detection.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = 0.3,
        top_skill_names: list[str] | None = None,
    ) -> None:
        self._pattern_matcher = PatternMatcher()
        self._ast_analyzer = ASTAnalyzer()
        self._metadata_checker = MetadataChecker(
            top_skill_names=top_skill_names if top_skill_names is not None else _load_top_skill_names(),
        )
        self._skillmd_analyzer = SkillMdAnalyzer()
        self._threshold = threshold
        self._model = None

        if model_path is not None:
            model_path = Path(model_path)
            if model_path.exists():
                try:
                    import xgboost as xgb  # type: ignore
                    booster = xgb.Booster()
                    booster.load_model(str(model_path))
                    self._model = booster
                except Exception:
                    # If loading fails for any reason, fall back to heuristic
                    self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble_features(self, pkg: SkillPackage) -> tuple[np.ndarray, list[str]]:
        """Collect features from all 4 modules and return a sorted feature vector.

        Booleans are converted to ``0.0`` / ``1.0``; integers and floats are
        kept as ``float``.

        Returns
        -------
        (feature_vector, feature_names) where both sequences are ordered
        alphabetically by feature name.
        """
        raw: dict[str, float] = {}

        # --- Module A: PatternMatcher ---
        pattern_features = self._pattern_matcher.extract_features(pkg)
        for key, val in pattern_features.items():
            # Rename 'total' to 'pattern_match_count' for clarity
            canonical = "pattern_match_count" if key == "total" else f"pattern_{key}"
            raw[canonical] = float(val)

        # --- Module B: ASTAnalyzer ---
        ast_features = self._ast_analyzer.extract_features(pkg)
        for key, val in ast_features.items():
            raw[key] = float(val)

        # --- Module C: MetadataChecker ---
        meta_features = self._metadata_checker.extract_features(pkg)
        for key, val in meta_features.items():
            raw[key] = float(bool(val)) if isinstance(val, bool) else float(val)

        # --- Module D: SkillMdAnalyzer ---
        skillmd_features = self._skillmd_analyzer.extract_features(pkg)
        for key, val in skillmd_features.items():
            raw[key] = float(bool(val)) if isinstance(val, bool) else float(val)

        # Sort by name for deterministic ordering
        sorted_names = sorted(raw.keys())
        vec = np.array([raw[n] for n in sorted_names], dtype=float)
        return vec, sorted_names

    def classify(self, pkg: SkillPackage) -> LayerResult:
        """Classify *pkg* and return a :class:`LayerResult`.

        Uses the XGBoost model when available; otherwise falls back to the
        heuristic scorer.
        """
        vec, names = self.assemble_features(pkg)
        matches = self._pattern_matcher.scan(pkg)

        if self._model is not None:
            score = self._xgboost_score(vec, names)
        else:
            score = self._heuristic_score(vec, names, matches)

        verdict = Verdict.SUSPICIOUS if score >= self._threshold else Verdict.SAFE

        # Build evidence list from pattern matches
        evidence: list[str] = []
        for m in matches:
            evidence.append(
                f"[{m.category}] {m.file}:{m.line} — {m.matched_text[:120]}"
            )

        # Symlink escape is a critical structural signal: a file symlinked to a
        # path outside the skill directory (e.g. ~/.ssh/id_rsa). Its target was
        # never read (see parser); force-escalate so it can never pass Layer 1.
        # Applied after scoring so it holds under both the XGBoost and heuristic
        # paths.
        if pkg.symlink_escapes:
            score = max(score, 0.95)
            verdict = Verdict.SUSPICIOUS
            for esc in pkg.symlink_escapes:
                evidence.append(
                    f"[symlink_escape] {esc} resolves outside the skill directory (target not read)"
                )

        details: dict[str, Any] = {
            "score": score,
            "threshold": self._threshold,
            "model_used": "xgboost" if self._model is not None else "heuristic",
            "pattern_match_count": len(matches),
            "symlink_escapes": list(pkg.symlink_escapes),
        }

        return LayerResult(
            layer=1,
            verdict=verdict,
            confidence=score,
            evidence=evidence,
            details=details,
        )

    def _heuristic_score(
        self,
        vec: np.ndarray,
        names: list[str],
        matches: list,
    ) -> float:
        """Compute a weighted heuristic risk score from ``[0.0, 1.0]``.

        Scoring rules
        -------------
        - ``pattern_match_count * 0.15``, capped at 0.60
        - ``total_exec_eval_count * 0.10``, capped at 0.30
        - ``requires_sensitive_env``:  +0.10
        - ``requires_dangerous_bins``: +0.05
        - ``urgency_language_score * 0.05``, capped at 0.20
        - ``name_edit_distance <= 2``: +0.15
        Total is capped at 1.0.
        """
        feature_map = dict(zip(names, vec))

        score = 0.0

        # Pattern match contribution (weighted by category severity)
        pattern_count = feature_map.get("pattern_match_count", 0.0)
        score += min(pattern_count * 0.15, 0.60)

        # High-severity categories get extra weight even with 1 match
        category_counts = {}
        for m in matches:
            category_counts[m.category] = category_counts.get(m.category, 0) + 1
        if category_counts.get("reverse_shell", 0) > 0:
            score += 0.40
        if category_counts.get("credential_theft", 0) > 0:
            score += 0.25
        if category_counts.get("obfuscation", 0) > 0:
            score += 0.20
        if category_counts.get("prompt_injection", 0) > 0:
            score += 0.20

        # Exec/eval contribution
        exec_eval = feature_map.get("total_exec_eval_count", 0.0)
        score += min(exec_eval * 0.10, 0.30)

        # Sensitive env requirements
        if feature_map.get("requires_sensitive_env", 0.0) >= 1.0:
            score += 0.10

        # Dangerous binary requirements
        if feature_map.get("requires_dangerous_bins", 0.0) >= 1.0:
            score += 0.05

        # Urgency language
        urgency = feature_map.get("urgency_language_score", 0.0)
        score += min(urgency * 0.05, 0.20)

        # Typosquatting: edit distance <= 2
        edit_dist = feature_map.get("name_edit_distance", -1.0)
        if 0.0 < edit_dist <= 2.0:
            score += 0.15

        # Homoglyph: non-ASCII characters in skill name
        if feature_map.get("has_homoglyph", 0.0) >= 1.0:
            score += 0.30

        # Conditional trigger / time delay patterns
        if category_counts.get("conditional_trigger", 0) > 0:
            score += 0.25
        if category_counts.get("time_delay", 0) > 0:
            score += 0.25

        # Feishu/enterprise bot suspicious patterns
        if category_counts.get("feishu_suspicious", 0) > 0:
            score += 0.20

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _xgboost_score(self, vec: np.ndarray, names: list[str]) -> float:
        """Return the XGBoost predicted probability for the SUSPICIOUS class."""
        import xgboost as xgb  # type: ignore

        dmatrix = xgb.DMatrix(vec.reshape(1, -1), feature_names=names)
        proba = self._model.predict(dmatrix)
        # predict returns an array; take the first (and only) value
        return float(proba[0])
