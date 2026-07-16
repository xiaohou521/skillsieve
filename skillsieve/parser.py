from __future__ import annotations

from pathlib import Path

import yaml

from skillsieve.models import SkillFrontmatter, SkillPackage

TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".sh", ".bash", ".zsh",
    ".json", ".yaml", ".yml", ".toml",
    ".md", ".txt", ".html", ".css", ".svg",
}


def _safe_resolve(p: Path) -> Path:
    """Resolve a path, tolerating broken links / symlink loops."""
    try:
        return p.resolve()
    except (OSError, RuntimeError):
        return p.absolute()


def _within(path: Path, base_resolved: Path) -> bool:
    """True if *path* resolves inside *base_resolved* (symlink-safe).

    A malicious skill can plant a symlink whose target is a local secret
    (``~/.ssh/id_rsa``, ``~/.aws/credentials``) or an out-of-tree file; reading
    it would leak that content into the scan and, at Layer 2, to an external LLM.
    We therefore only read files that stay inside the skill directory.
    """
    try:
        rp = path.resolve()
    except (OSError, RuntimeError):  # broken link / ELOOP
        return False
    return rp == base_resolved or rp.is_relative_to(base_resolved)


def parse_skill_package(skill_dir: Path) -> SkillPackage:
    """Parse a skill directory into a SkillPackage.

    Security: any file whose symlink target escapes ``skill_dir`` is NOT read;
    its relative path is recorded in ``symlink_escapes`` as a critical Layer 1
    signal instead.
    """
    skill_dir = Path(skill_dir)
    base = _safe_resolve(skill_dir)
    escapes: list[str] = []

    skill_md_path = skill_dir / "SKILL.md"
    if skill_md_path.is_symlink() and not _within(skill_md_path, base):
        raw = ""
        escapes.append("SKILL.md")
    elif skill_md_path.exists() and _within(skill_md_path, base):
        raw = skill_md_path.read_text(encoding="utf-8")
    else:
        raw = ""
    fm_str, body = _split_frontmatter(raw)
    frontmatter = _parse_frontmatter(fm_str)

    # Use frontmatter name if set, otherwise fall back to directory name
    name = frontmatter.name or skill_dir.name

    script_files, script_escapes = _collect_scripts(skill_dir, base)
    escapes.extend(script_escapes)

    return SkillPackage(
        path=skill_dir,
        name=name,
        frontmatter=frontmatter,
        skill_md_content=body,
        script_files=script_files,
        symlink_escapes=escapes,
    )


def _split_frontmatter(raw: str) -> tuple[str, str]:
    """Split raw SKILL.md text into (frontmatter_str, body_str).

    Frontmatter is delimited by leading '---' lines. If no frontmatter
    markers are found, returns ('', raw).
    """
    if not raw.startswith("---"):
        return "", raw

    # Find the closing '---' after the opening one
    # raw starts with '---', so search from position 3 onward
    rest = raw[3:]
    # Allow optional trailing content on the opening line (e.g. "--- \n")
    newline_pos = rest.find("\n")
    if newline_pos == -1:
        return "", raw
    rest = rest[newline_pos + 1:]  # content after the first '---\n'

    closing = rest.find("\n---")
    if closing == -1:
        # No closing marker — treat entire file as body
        return "", raw

    fm_str = rest[:closing]
    body = rest[closing + 4:]  # skip '\n---'
    # Strip any trailing content on the closing '---' line
    body_newline = body.find("\n")
    if body_newline != -1:
        body = body[body_newline + 1:]
    else:
        body = ""

    return fm_str, body


def _parse_frontmatter(fm_str: str) -> SkillFrontmatter:
    """Parse a YAML frontmatter string into a SkillFrontmatter object."""
    if not fm_str.strip():
        return SkillFrontmatter()

    try:
        data = yaml.safe_load(fm_str) or {}
    except yaml.YAMLError:
        return SkillFrontmatter()
    if not isinstance(data, dict):
        return SkillFrontmatter()

    name = data.get("name", "")
    description = data.get("description", "")
    version = str(data.get("version", ""))

    # Extract nested metadata.openclaw.requires.{env,bins}
    requires: dict = {}
    try:
        requires = data["metadata"]["openclaw"]["requires"]
        if not isinstance(requires, dict):
            requires = {}
    except (KeyError, TypeError):
        requires = {}

    # Tolerate both str-list and dict-list formats (some skills use
    # `[{name: FOO, required: true}, ...]` rather than `[FOO, ...]`).
    def _normalize_entries(raw) -> list[str]:
        items: list[str] = []
        for entry in raw or []:
            if isinstance(entry, str):
                items.append(entry)
            elif isinstance(entry, dict):
                name_val = entry.get("name") or entry.get("key") or ""
                if name_val:
                    items.append(str(name_val))
        return items

    requires_env: list[str] = _normalize_entries(requires.get("env"))
    requires_bins: list[str] = _normalize_entries(requires.get("bins"))

    return SkillFrontmatter(
        name=name,
        description=description,
        version=version,
        requires_env=requires_env,
        requires_bins=requires_bins,
    )


def _collect_scripts(skill_dir: Path, base: Path) -> tuple[dict[str, str], list[str]]:
    """Walk scripts/ and return ({relative_path: content}, escaped_paths).

    Files whose symlink target escapes *base* (the resolved skill dir) are not
    read; their relative paths are returned separately as escapes. If scripts/
    itself is a symlink pointing outside, we skip enumeration entirely (avoids
    walking an attacker-chosen external tree such as ``scripts -> /``).
    """
    scripts_dir = skill_dir / "scripts"
    escapes: list[str] = []
    if not scripts_dir.is_dir():
        return {}, escapes
    if not _within(scripts_dir, base):
        return {}, ["scripts/"]

    result: dict[str, str] = {}
    for file_path in sorted(scripts_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        relative = str(file_path.relative_to(skill_dir))
        if not _within(file_path, base):
            escapes.append(relative)  # symlink target outside skill dir — do NOT read
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        result[relative] = content

    return result, escapes
