import tempfile
from pathlib import Path

import pytest

from skillsieve.parser import parse_skill_package

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_benign_skill():
    pkg = parse_skill_package(FIXTURES / "benign_skill")
    assert pkg.name == "code-formatter"
    assert pkg.frontmatter.description == "Formats Python code using black"
    assert pkg.frontmatter.requires_bins == ["black"]
    assert pkg.frontmatter.requires_env == []
    assert "Format the user's Python files" in pkg.skill_md_content
    assert pkg.script_files == {}


def test_parse_malicious_skill():
    pkg = parse_skill_package(FIXTURES / "malicious_skill")
    assert pkg.name == "github-copilott"
    assert pkg.frontmatter.requires_env == ["OPENAI_API_KEY", "GITHUB_TOKEN"]
    assert pkg.frontmatter.requires_bins == ["curl"]
    assert "scripts/setup.sh" in pkg.script_files
    assert "curl -X POST" in pkg.script_files["scripts/setup.sh"]


def test_parse_missing_frontmatter():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "no-frontmatter-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("Just some plain markdown content with no frontmatter.\n")

        pkg = parse_skill_package(skill_dir)
        # Name falls back to directory name
        assert pkg.name == "no-frontmatter-skill"
        # Frontmatter fields use defaults
        assert pkg.frontmatter.requires_env == []
        assert pkg.frontmatter.requires_bins == []
        # Body content is present
        assert "plain markdown content" in pkg.skill_md_content
        assert pkg.script_files == {}
