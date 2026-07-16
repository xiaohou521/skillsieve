from pathlib import Path
from skillsieve.models import (
    AttackType,
    LayerResult,
    ScanResult,
    SkillFrontmatter,
    SkillPackage,
    Verdict,
)


def test_skill_package_creation():
    pkg = SkillPackage(
        path=Path("/tmp/test-skill"),
        name="test-skill",
        frontmatter=SkillFrontmatter(
            name="test-skill",
            description="A test skill",
            version="1.0.0",
        ),
        skill_md_content="# Test\nDo something useful.",
        script_files={"scripts/run.py": "print('hello')"},
    )
    assert pkg.name == "test-skill"
    assert pkg.frontmatter.description == "A test skill"
    assert "scripts/run.py" in pkg.script_files


def test_layer_result():
    result = LayerResult(
        layer=1,
        verdict=Verdict.SUSPICIOUS,
        confidence=0.65,
        evidence=["Pattern match: base64 decode in scripts/run.py:3"],
        details={"risk_score": 0.65},
    )
    assert result.layer == 1
    assert result.verdict == Verdict.SUSPICIOUS


def test_scan_result():
    l1 = LayerResult(layer=1, verdict=Verdict.SUSPICIOUS, confidence=0.7, evidence=[])
    l2 = LayerResult(
        layer=2,
        verdict=Verdict.MALICIOUS,
        confidence=0.9,
        evidence=[],
        details={"attack_types": [AttackType.CREDENTIAL_THEFT]},
    )
    result = ScanResult(
        skill_name="bad-skill",
        final_verdict=Verdict.MALICIOUS,
        final_confidence=0.9,
        layer_stopped=2,
        layer_results=[l1, l2],
    )
    assert result.layer_stopped == 2
    assert result.final_verdict == Verdict.MALICIOUS
