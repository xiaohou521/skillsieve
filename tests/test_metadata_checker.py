"""Tests for MetadataChecker (Layer 1 Module C)."""
from __future__ import annotations

from pathlib import Path

import pytest

from skillsieve.core.layer1.metadata_checker import MetadataChecker
from skillsieve.parser import parse_skill_package

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MALICIOUS_DIR = FIXTURE_DIR / "malicious_skill"
BENIGN_DIR = FIXTURE_DIR / "benign_skill"

TOP_SKILL_NAMES = ["github-copilot", "code-formatter", "web-search", "docker-helper"]


@pytest.fixture
def checker() -> MetadataChecker:
    return MetadataChecker(top_skill_names=TOP_SKILL_NAMES)


@pytest.fixture
def malicious_pkg():
    return parse_skill_package(MALICIOUS_DIR)


@pytest.fixture
def benign_pkg():
    return parse_skill_package(BENIGN_DIR)


# ---------------------------------------------------------------------------
# Test 1: Typosquatting detection
# ---------------------------------------------------------------------------

def test_typosquatting_detection(checker: MetadataChecker, malicious_pkg):
    """'github-copilott' is 1 edit away from 'github-copilot' → distance <= 2."""
    features = checker.extract_features(malicious_pkg)
    assert features["name_edit_distance"] <= 2, (
        f"Expected edit distance <= 2 for typosquat, got {features['name_edit_distance']}"
    )


# ---------------------------------------------------------------------------
# Test 2: Sensitive environment variable detection
# ---------------------------------------------------------------------------

def test_sensitive_env_detection(checker: MetadataChecker, malicious_pkg):
    """Malicious skill requires OPENAI_API_KEY → requires_sensitive_env is True."""
    features = checker.extract_features(malicious_pkg)
    assert features["requires_sensitive_env"] is True, (
        "Expected requires_sensitive_env=True for skill requiring OPENAI_API_KEY"
    )


# ---------------------------------------------------------------------------
# Test 3: Benign skill has zero edit distance and no sensitive env
# ---------------------------------------------------------------------------

def test_benign_skill_metadata(checker: MetadataChecker, benign_pkg):
    """'code-formatter' is an exact match in top_skill_names → edit_distance 0."""
    features = checker.extract_features(benign_pkg)
    assert features["name_edit_distance"] == 0, (
        f"Expected edit distance 0 for exact match, got {features['name_edit_distance']}"
    )
    assert features["requires_sensitive_env"] is False, (
        "Expected requires_sensitive_env=False for benign skill"
    )
