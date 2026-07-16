import pytest
from pathlib import Path

from dataset.labeling.annotation_tool import AnnotationTool, Annotation


def test_add_annotation():
    tool = AnnotationTool(Path("/tmp/ann_test_add.json"))
    tool.annotations.clear()

    tool.add_annotation("skill-a", "benign", annotator="alice")
    tool.add_annotation("skill-b", "malicious", attack_types=["credential_theft"], stealth_rating=3)

    assert len(tool.annotations) == 2
    assert tool.annotations[0].label == "benign"
    assert tool.annotations[1].label == "malicious"
    assert tool.annotations[1].attack_types == ["credential_theft"]
    assert tool.annotations[1].stealth_rating == 3


def test_save_and_load(tmp_path):
    out = tmp_path / "annotations.json"
    tool = AnnotationTool(out)
    tool.add_annotation("skill-x", "benign", notes="looks fine", annotator="bob")
    tool.add_annotation("skill-y", "malicious", attack_types=["obfuscation"], stealth_rating=4)
    tool.save()

    # Load into a fresh instance
    tool2 = AnnotationTool(out)
    assert len(tool2.annotations) == 2
    assert tool2.annotations[0].skill_name == "skill-x"
    assert tool2.annotations[0].label == "benign"
    assert tool2.annotations[0].notes == "looks fine"
    assert tool2.annotations[1].skill_name == "skill-y"
    assert tool2.annotations[1].attack_types == ["obfuscation"]
    assert tool2.annotations[1].stealth_rating == 4


def test_invalid_label():
    tool = AnnotationTool(Path("/tmp/ann_test_invalid_label.json"))
    with pytest.raises(ValueError, match="label must be"):
        tool.add_annotation("skill-z", "unknown")


def test_invalid_attack_type():
    tool = AnnotationTool(Path("/tmp/ann_test_invalid_attack.json"))
    with pytest.raises(ValueError, match="Unknown attack type"):
        tool.add_annotation("skill-z", "malicious", attack_types=["super_hack"])


def test_get_stats():
    tool = AnnotationTool(Path("/tmp/ann_test_stats.json"))
    tool.annotations.clear()

    tool.add_annotation("s1", "benign")
    tool.add_annotation("s2", "benign")
    tool.add_annotation("s3", "malicious", attack_types=["data_exfil"])
    tool.add_annotation("s4", "malicious", attack_types=["remote_exec"])
    tool.add_annotation("s5", "malicious", attack_types=["prompt_injection"])

    stats = tool.get_stats()
    assert stats["total"] == 5
    assert stats["benign"] == 2
    assert stats["malicious"] == 3
