"""Tests for the ASTAnalyzer (Layer 1 Module B)."""
from __future__ import annotations

from pathlib import Path

import pytest

from skillsieve.core.layer1.ast_analyzer import ASTAnalyzer
from skillsieve.models import SkillPackage, SkillFrontmatter
from skillsieve.parser import parse_skill_package

FIXTURE_DIR = Path(__file__).parent / "fixtures"
OBFUSCATED_DIR = FIXTURE_DIR / "obfuscated_skill"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer() -> ASTAnalyzer:
    return ASTAnalyzer()


@pytest.fixture
def obfuscated_pkg() -> SkillPackage:
    return parse_skill_package(OBFUSCATED_DIR)


# ---------------------------------------------------------------------------
# Test 1: obfuscated skill has suspicious features
# ---------------------------------------------------------------------------

def test_obfuscated_skill_features(analyzer: ASTAnalyzer):
    """The obfuscated run.py should trigger env_access, exec_eval, and network_call."""
    code = (OBFUSCATED_DIR / "scripts" / "run.py").read_text()
    features = analyzer.extract_features_from_file(code, "python")

    assert features["env_access_count"] >= 1, (
        f"Expected env_access_count >= 1, got {features['env_access_count']}"
    )
    assert features["exec_eval_count"] >= 1, (
        f"Expected exec_eval_count >= 1, got {features['exec_eval_count']}"
    )
    assert features["network_call_count"] >= 1, (
        f"Expected network_call_count >= 1, got {features['network_call_count']}"
    )
    assert features["entropy_score"] > 0, (
        f"Expected entropy_score > 0, got {features['entropy_score']}"
    )


# ---------------------------------------------------------------------------
# Test 2: benign script has all counts at zero
# ---------------------------------------------------------------------------

def test_benign_script_low_features(analyzer: ASTAnalyzer):
    """A simple greeting function should produce no suspicious feature counts."""
    benign_code = """
def greet(name):
    return f"Hello, {name}!"
"""
    features = analyzer.extract_features_from_file(benign_code, "python")

    assert features["syscall_count"] == 0
    assert features["network_call_count"] == 0
    assert features["env_access_count"] == 0
    assert features["exec_eval_count"] == 0
    assert features["encoded_string_count"] == 0


# ---------------------------------------------------------------------------
# Test 3: extract_features over the full package
# ---------------------------------------------------------------------------

def test_extract_from_package(analyzer: ASTAnalyzer, obfuscated_pkg: SkillPackage):
    """aggregate features from the obfuscated fixture package are total_ prefixed."""
    features = analyzer.extract_features(obfuscated_pkg)

    # All keys should be total_-prefixed
    assert all(k.startswith("total_") for k in features), (
        f"Expected all keys to start with total_, got: {list(features.keys())}"
    )

    # The obfuscated script should show up in aggregated counts
    assert features["total_env_access_count"] >= 1
    assert features["total_exec_eval_count"] >= 1
    assert features["total_network_call_count"] >= 1
    assert features["total_entropy_score"] > 0
