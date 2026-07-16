"""Git-based ClawHub skill collector.

Clones the openclaw/clawhub GitHub repo (or uses a local clone) and extracts
individual skill packages into a structured output directory.
"""
import json
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class CrawlResult:
    total_skills: int = 0
    successful: int = 0
    failed: int = 0
    skill_names: list[str] = field(default_factory=list)


class ClawHubCrawler:
    REPO_URL = "https://github.com/openclaw/clawhub.git"
    SKILLS_SUBDIR = "skills"  # subdirectory in the repo containing skills

    def __init__(self, output_dir: Path, repo_url: str | None = None):
        self.output_dir = Path(output_dir)
        self.repo_url = repo_url or self.REPO_URL
        self._clone_dir = self.output_dir / ".clone_cache"

    def clone_or_update(self) -> Path:
        """Clone the repo or pull latest if already cloned."""
        if (self._clone_dir / ".git").exists():
            subprocess.run(
                ["git", "-C", str(self._clone_dir), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )
        else:
            self._clone_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self._clone_dir)],
                check=True,
                capture_output=True,
            )
        return self._clone_dir

    def extract_skills(self, clone_dir: Path | None = None) -> CrawlResult:
        """Extract individual skill directories from the cloned repo."""
        clone = clone_dir or self._clone_dir
        skills_root = clone / self.SKILLS_SUBDIR
        result = CrawlResult()

        if not skills_root.is_dir():
            # Try flat structure (each top-level dir with SKILL.md)
            skills_root = clone

        dest = self.output_dir / "skills"
        dest.mkdir(parents=True, exist_ok=True)

        for skill_dir in self._find_skill_dirs(skills_root):
            result.total_skills += 1
            try:
                skill_name = skill_dir.name
                target = dest / skill_name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(skill_dir, target)
                result.successful += 1
                result.skill_names.append(skill_name)
            except Exception:
                result.failed += 1

        # Save manifest
        manifest = {
            "total": result.total_skills,
            "successful": result.successful,
            "skills": sorted(result.skill_names),
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return result

    def _find_skill_dirs(self, root: Path) -> list[Path]:
        """Find all directories containing a SKILL.md file."""
        return sorted(d.parent for d in root.rglob("SKILL.md") if d.is_file())

    def crawl(self) -> CrawlResult:
        """Full pipeline: clone + extract."""
        self.clone_or_update()
        return self.extract_skills()
