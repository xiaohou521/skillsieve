import json
import tempfile
from pathlib import Path

from dataset.crawlers.clawhub_crawler import ClawHubCrawler, CrawlResult


def test_find_skill_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create fake skill structure
        (root / "skills" / "skill-a").mkdir(parents=True)
        (root / "skills" / "skill-a" / "SKILL.md").write_text("---\nname: a\n---\nHello")
        (root / "skills" / "skill-b").mkdir(parents=True)
        (root / "skills" / "skill-b" / "SKILL.md").write_text("---\nname: b\n---\nWorld")
        (root / "not-a-skill").mkdir()
        (root / "not-a-skill" / "README.md").write_text("nope")

        crawler = ClawHubCrawler(output_dir=root / "output")
        dirs = crawler._find_skill_dirs(root / "skills")
        assert len(dirs) == 2
        names = {d.name for d in dirs}
        assert names == {"skill-a", "skill-b"}


def test_extract_skills():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        clone = root / "clone"
        (clone / "skills" / "my-skill").mkdir(parents=True)
        (clone / "skills" / "my-skill" / "SKILL.md").write_text("---\nname: my-skill\n---\nDo stuff")
        (clone / "skills" / "my-skill" / "scripts").mkdir()
        (clone / "skills" / "my-skill" / "scripts" / "run.sh").write_text("echo hi")

        output = root / "output"
        crawler = ClawHubCrawler(output_dir=output)
        result = crawler.extract_skills(clone_dir=clone)

        assert result.total_skills == 1
        assert result.successful == 1
        assert "my-skill" in result.skill_names
        assert (output / "skills" / "my-skill" / "SKILL.md").exists()
        assert (output / "skills" / "my-skill" / "scripts" / "run.sh").exists()

        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["total"] == 1


def test_crawl_result_dataclass():
    r = CrawlResult(total_skills=10, successful=8, failed=2, skill_names=["a", "b"])
    assert r.total_skills == 10
    assert len(r.skill_names) == 2
