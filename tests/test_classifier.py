"""Tests for Layer1Classifier."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skillsieve.models import SkillPackage, SkillFrontmatter, Verdict
from skillsieve.core.layer1.classifier import Layer1Classifier

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def malicious_skill() -> SkillPackage:
    from skillsieve.parser import parse_skill_package
    return parse_skill_package(FIXTURES / "malicious_skill")


@pytest.fixture
def benign_skill() -> SkillPackage:
    from skillsieve.parser import parse_skill_package
    return parse_skill_package(FIXTURES / "benign_skill")


@pytest.fixture
def classifier() -> Layer1Classifier:
    return Layer1Classifier(
        model_path=None,
        threshold=0.3,
        top_skill_names=["github-copilot", "code-formatter", "git-helper"],
    )


def test_assemble_feature_vector(classifier: Layer1Classifier, malicious_skill: SkillPackage):
    """assemble_features returns a numpy array with >= 20 features."""
    vec, names = classifier.assemble_features(malicious_skill)

    assert isinstance(vec, np.ndarray), "Feature vector must be a numpy ndarray"
    assert vec.dtype == float or np.issubdtype(vec.dtype, np.floating), (
        "Feature values must be floats"
    )
    assert len(vec) >= 20, f"Expected >= 20 features, got {len(vec)}"
    assert len(vec) == len(names), "Vector length must match names length"
    assert len(names) == len(set(names)), "Feature names must be unique"
    # Names should be sorted
    assert names == sorted(names), "Feature names must be sorted alphabetically"
    # All values must be finite floats
    assert all(np.isfinite(v) for v in vec), "All feature values must be finite"


def test_classify_without_trained_model(classifier: Layer1Classifier, malicious_skill: SkillPackage):
    """Heuristic fallback on malicious fixture should yield SUSPICIOUS verdict."""
    result = classifier.classify(malicious_skill)

    assert result.layer == 1
    assert result.verdict == Verdict.SUSPICIOUS, (
        f"Expected SUSPICIOUS for malicious skill, got {result.verdict}"
    )
    assert 0.0 <= result.confidence <= 1.0
    # There should be some evidence collected
    assert len(result.evidence) > 0


def test_classify_benign(classifier: Layer1Classifier, benign_skill: SkillPackage):
    """Benign fixture should yield SAFE verdict with heuristic scoring."""
    result = classifier.classify(benign_skill)

    assert result.layer == 1
    assert result.verdict == Verdict.SAFE, (
        f"Expected SAFE for benign skill, got {result.verdict}"
    )
    assert 0.0 <= result.confidence <= 1.0


def test_heuristic_score_zero_for_clean_package(classifier: Layer1Classifier):
    """A completely clean package should score near 0 heuristically."""
    pkg = SkillPackage(
        path=Path("/tmp/clean-skill"),
        name="clean-skill",
        frontmatter=SkillFrontmatter(),
        skill_md_content="This skill formats text.",
        script_files={},
    )
    vec, names = classifier.assemble_features(pkg)
    from skillsieve.core.layer1.pattern_matcher import PatternMatcher
    matches = PatternMatcher().scan(pkg)
    score = classifier._heuristic_score(vec, names, matches)
    assert score == 0.0, f"Expected 0.0 heuristic score for clean package, got {score}"


def test_feature_names_cover_all_modules(classifier: Layer1Classifier, malicious_skill: SkillPackage):
    """Feature vector must include contributions from all 4 modules."""
    _, names = classifier.assemble_features(malicious_skill)
    name_set = set(names)

    # Pattern matcher features
    assert "pattern_match_count" in name_set or any("pattern" in n for n in name_set), (
        "Missing pattern matcher feature"
    )
    # AST features (prefixed with total_)
    assert any("total_" in n for n in name_set), "Missing AST analyzer features"
    # Metadata features
    assert "requires_sensitive_env" in name_set or any("sensitive" in n for n in name_set), (
        "Missing metadata checker feature"
    )
    # SkillMd features
    assert "urgency_language_score" in name_set or any("urgency" in n for n in name_set), (
        "Missing skillmd analyzer feature"
    )


def test_classify_returns_layer_result_structure(classifier: Layer1Classifier, benign_skill: SkillPackage):
    """classify returns a valid LayerResult with correct structure."""
    from skillsieve.models import LayerResult
    result = classifier.classify(benign_skill)

    assert isinstance(result, LayerResult)
    assert result.layer == 1
    assert result.verdict in (Verdict.SAFE, Verdict.SUSPICIOUS)
    assert isinstance(result.confidence, float)
    assert isinstance(result.evidence, list)
    assert isinstance(result.details, dict)
