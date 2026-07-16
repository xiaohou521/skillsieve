"""Fetch skills from IOC-listed malicious authors using GitHub API.

Strategy:
1. Load author blocklist from the 3 IOC sources.
2. For each author, list their skills in openclaw/skills via `gh api`.
3. For each skill, download SKILL.md + scripts/* via raw.githubusercontent.com.
4. Verify each against Clawdex API for label confirmation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

IOC_DIR = Path("/tmp/ioc")
OUTPUT_DIR = Path("/tmp/openclaw-skills-audit/skills")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_malicious_authors() -> list[str]:
    authors: set[str] = set()

    # piti
    p = IOC_DIR / "openclaw-security-dashboard" / "ioc" / "malicious-publishers.json"
    if p.exists():
        d = json.loads(p.read_text())
        authors.update(pub["name"] for pub in d.get("publishers", []))

    # adibirzu
    for fname in ["malicious-publishers.txt"]:
        p = IOC_DIR / "openclaw-security-monitor" / "ioc" / fname
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                authors.add(line.split()[0])

    # nightfullstar
    p = IOC_DIR / "openclaw-defender" / "references" / "blocklist.conf"
    if p.exists():
        current = None
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
                continue
            token = line.split("#")[0].strip().split()
            if not token:
                continue
            if current == "authors":
                authors.add(token[0])

    return sorted(authors)


def gh_list(path: str) -> list[dict]:
    """Use gh api to list a directory."""
    cmd = ["gh", "api", f"/repos/openclaw/skills/contents/{path}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def raw_download(path: str) -> str | None:
    """Download file content via raw.githubusercontent.com (no rate limit)."""
    cmd = ["curl", "-sfL", f"https://raw.githubusercontent.com/openclaw/skills/main/{path}"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode != 0:
            return None
        return r.stdout.decode("utf-8", errors="replace")
    except Exception:
        return None


def main():
    authors = load_malicious_authors()
    print(f"Malicious authors from IOCs: {len(authors)}")
    for a in authors:
        print(f"  {a}")

    fetched = {}
    total_skills = 0
    for i, author in enumerate(authors):
        print(f"\n[{i+1}/{len(authors)}] {author}: listing skills...")
        skills = gh_list(f"skills/{author}")
        skill_dirs = [s for s in skills if s.get("type") == "dir"]
        print(f"  found {len(skill_dirs)} skills")
        if not skill_dirs:
            continue

        fetched[author] = []
        for skill in skill_dirs:
            skill_name = skill["name"]
            skill_path = f"skills/{author}/{skill_name}"

            # List skill contents
            contents = gh_list(skill_path)
            if not contents:
                continue

            # Download SKILL.md and other files
            out_dir = OUTPUT_DIR / author / skill_name
            out_dir.mkdir(parents=True, exist_ok=True)
            files_downloaded = 0

            for item in contents:
                if item.get("type") != "file":
                    continue
                name = item["name"]
                # Focus on SKILL.md, README.md, scripts/
                content = raw_download(f"{skill_path}/{name}")
                if content is not None:
                    (out_dir / name).write_text(content)
                    files_downloaded += 1

            # Also try scripts subdirectory
            scripts = gh_list(f"{skill_path}/scripts")
            if scripts:
                (out_dir / "scripts").mkdir(exist_ok=True)
                for item in scripts:
                    if item.get("type") != "file":
                        continue
                    name = item["name"]
                    content = raw_download(f"{skill_path}/scripts/{name}")
                    if content is not None:
                        (out_dir / "scripts" / name).write_text(content)
                        files_downloaded += 1

            if files_downloaded > 0:
                fetched[author].append(skill_name)
                total_skills += 1

        print(f"  ✓ downloaded {len(fetched[author])} skills (total so far: {total_skills})")

    # Save summary
    summary = {
        "total_authors_queried": len(authors),
        "authors_with_skills": len(fetched),
        "total_skills_downloaded": total_skills,
        "per_author": {a: len(s) for a, s in fetched.items()},
    }
    Path("results/ioc_fetched_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Summary ===")
    print(f"Authors queried: {len(authors)}")
    print(f"Authors with skills: {len(fetched)}")
    print(f"Total skills downloaded: {total_skills}")


if __name__ == "__main__":
    main()
