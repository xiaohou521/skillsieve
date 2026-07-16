"""Adapter to convert Feishu/Lark ecosystem packages into SkillSieve SkillPackage format.

Supports three source types:
1. OpenClaw-format skills (have SKILL.md) — used by larksuite/cli skills and openclaw-lark skills
2. OpenClaw plugins (have openclaw.plugin.json) — community feishu plugins
3. Standalone bots (have README.md + source code) — independent feishu bot repos
"""
from __future__ import annotations

import json
from pathlib import Path

from skillsieve.models import SkillFrontmatter, SkillPackage
from skillsieve.parser import TEXT_EXTENSIONS, parse_skill_package, _split_frontmatter, _parse_frontmatter


def parse_feishu_package(pkg_dir: Path) -> SkillPackage:
    """Auto-detect format and parse a feishu ecosystem package."""
    pkg_dir = Path(pkg_dir)

    # Type 1: Has SKILL.md — standard OpenClaw skill format
    if (pkg_dir / "SKILL.md").exists():
        return _parse_with_extra_sources(pkg_dir)

    # Type 2: Has openclaw.plugin.json — OpenClaw plugin format
    if (pkg_dir / "openclaw.plugin.json").exists():
        return _parse_plugin(pkg_dir)

    # Type 3: Standalone bot repo
    return _parse_standalone_bot(pkg_dir)


def _parse_with_extra_sources(skill_dir: Path) -> SkillPackage:
    """Parse a standard SKILL.md skill, but also collect src/ TS/JS files as scripts."""
    pkg = parse_skill_package(skill_dir)

    # Also collect TS/JS source files outside scripts/ (feishu plugins put code in src/)
    extra_sources = _collect_all_sources(skill_dir, exclude_dirs={"node_modules", ".git", "dist", "build"})
    for rel, content in extra_sources.items():
        if rel not in pkg.script_files:
            pkg.script_files[rel] = content

    return pkg


def _parse_plugin(pkg_dir: Path) -> SkillPackage:
    """Parse an OpenClaw plugin (openclaw.plugin.json + package.json + source)."""
    # Read manifest
    manifest = {}
    manifest_path = pkg_dir / "openclaw.plugin.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass

    # Read package.json for metadata
    pkg_json = {}
    pkg_json_path = pkg_dir / "package.json"
    try:
        pkg_json = json.loads(pkg_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass

    # Build frontmatter from manifest + package.json
    name = manifest.get("id") or pkg_json.get("name") or pkg_dir.name
    description = pkg_json.get("description", "")
    version = pkg_json.get("version", "")

    # Extract dependencies as env requirements (they reveal what the plugin uses)
    deps = list(pkg_json.get("dependencies", {}).keys())

    frontmatter = SkillFrontmatter(
        name=name,
        description=description,
        version=version,
        requires_bins=deps[:10],  # store top deps as metadata signal
    )

    # Build skill_md_content from README
    readme_content = _read_readme(pkg_dir)

    # Collect source files
    script_files = _collect_all_sources(pkg_dir, exclude_dirs={"node_modules", ".git", "dist", "build", "test", "tests"})

    # Also include manifest and package.json as "scripts" for pattern scanning
    script_files["openclaw.plugin.json"] = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
    if pkg_json_path.exists():
        script_files["package.json"] = pkg_json_path.read_text(encoding="utf-8")

    return SkillPackage(
        path=pkg_dir,
        name=name,
        frontmatter=frontmatter,
        skill_md_content=readme_content,
        script_files=script_files,
    )


def _parse_standalone_bot(pkg_dir: Path) -> SkillPackage:
    """Parse a standalone bot repo (README + source code)."""
    readme_content = _read_readme(pkg_dir)

    # Try to get name from package.json, go.mod, setup.py, or directory name
    name = pkg_dir.name
    description = ""
    version = ""

    pkg_json_path = pkg_dir / "package.json"
    if pkg_json_path.exists():
        try:
            pkg_json = json.loads(pkg_json_path.read_text(encoding="utf-8"))
            name = pkg_json.get("name", name)
            description = pkg_json.get("description", "")
            version = pkg_json.get("version", "")
        except (json.JSONDecodeError, OSError):
            pass

    frontmatter = SkillFrontmatter(
        name=name,
        description=description,
        version=version,
    )

    script_files = _collect_all_sources(pkg_dir, exclude_dirs={"node_modules", ".git", "dist", "build", "vendor", "__pycache__", "test", "tests"})

    return SkillPackage(
        path=pkg_dir,
        name=name,
        frontmatter=frontmatter,
        skill_md_content=readme_content,
        script_files=script_files,
    )


def _read_readme(pkg_dir: Path) -> str:
    """Find and read README file."""
    for name in ("README.md", "README.zh.md", "readme.md", "README.rst", "README"):
        p = pkg_dir / name
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
    return ""


SOURCE_EXTENSIONS = TEXT_EXTENSIONS | {".go", ".rs", ".java", ".tsx", ".jsx", ".mjs", ".cjs"}


def _collect_all_sources(
    pkg_dir: Path,
    exclude_dirs: set[str] | None = None,
    max_files: int = 50,
    max_file_size: int = 50_000,
) -> dict[str, str]:
    """Collect all source files from a package directory."""
    if exclude_dirs is None:
        exclude_dirs = {"node_modules", ".git", "dist", "build"}

    result: dict[str, str] = {}
    for file_path in sorted(pkg_dir.rglob("*")):
        if len(result) >= max_files:
            break
        if not file_path.is_file():
            continue
        # Skip excluded directories
        if any(part in exclude_dirs for part in file_path.relative_to(pkg_dir).parts):
            continue
        if file_path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if file_path.stat().st_size > max_file_size:
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        result[str(file_path.relative_to(pkg_dir))] = content

    return result


def discover_feishu_skills(ecosystem_dir: Path) -> list[SkillPackage]:
    """Discover all scannable packages from a feishu ecosystem directory.

    Handles two patterns:
    - Repos with a skills/ subdirectory (e.g., larksuite/cli, openclaw-lark):
      each subdirectory under skills/ is treated as a separate skill.
    - Repos without skills/ subdirectory: the repo root is treated as one package.
    """
    ecosystem_dir = Path(ecosystem_dir)
    packages: list[SkillPackage] = []

    for repo_dir in sorted(ecosystem_dir.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name.startswith("."):
            continue

        skills_subdir = repo_dir / "skills"
        if skills_subdir.is_dir():
            # Each child of skills/ is a separate skill
            for skill_dir in sorted(skills_subdir.iterdir()):
                if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                    continue
                try:
                    pkg = parse_feishu_package(skill_dir)
                    pkg.name = f"{repo_dir.name}/{pkg.name}"  # prefix with repo name
                    packages.append(pkg)
                except Exception:
                    continue
        else:
            # Whole repo is one package
            try:
                pkg = parse_feishu_package(repo_dir)
                packages.append(pkg)
            except Exception:
                continue

    return packages
