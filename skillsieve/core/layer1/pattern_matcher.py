from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from skillsieve.models import SkillPackage

# Resolve the rules file relative to this file so the engine works regardless
# of the current working directory.
_RULES_PATH = Path(__file__).parent.parent.parent / "rules" / "patterns.yaml"


@dataclass
class PatternMatch:
    """A single regex hit found during a pattern scan."""

    category: str
    pattern: str
    file: str
    line: int
    matched_text: str


class PatternMatcher:
    """Load regex rules from patterns.yaml and scan SkillPackage content."""

    def __init__(self, rules_path: Path | None = None) -> None:
        path = Path(rules_path) if rules_path is not None else _RULES_PATH
        raw: dict[str, list[str]] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        # Pre-compile every pattern; keep the original source string for reporting.
        self.rules: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
        for category, patterns in raw.items():
            compiled: list[tuple[str, re.Pattern[str]]] = []
            for pat in patterns:
                compiled.append((pat, re.compile(pat, re.IGNORECASE)))
            self.rules[category] = compiled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, pkg: SkillPackage) -> list[PatternMatch]:
        """Scan SKILL.md body and all script files for malicious patterns.

        Returns a flat list of PatternMatch objects, one per (pattern, line) hit.
        """
        matches: list[PatternMatch] = []

        # Scan the SKILL.md body content
        if pkg.skill_md_content:
            matches.extend(self._scan_text(pkg.skill_md_content, file="SKILL.md"))

        # Scan every script file
        for file_path, content in pkg.script_files.items():
            matches.extend(self._scan_text(content, file=file_path))

        return matches

    def extract_features(self, pkg: SkillPackage) -> dict[str, int]:
        """Return match counts per category plus a 'total' key.

        Example::

            {
                "reverse_shell": 0,
                "credential_theft": 2,
                "data_exfil": 1,
                "obfuscation": 0,
                "prompt_injection": 0,
                "total": 3,
            }
        """
        matches = self.scan(pkg)
        counts: dict[str, int] = {cat: 0 for cat in self.rules}
        for m in matches:
            counts[m.category] = counts.get(m.category, 0) + 1
        counts["total"] = sum(counts[cat] for cat in self.rules)
        return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_text(self, text: str, *, file: str) -> list[PatternMatch]:
        """Scan *text* line-by-line against all compiled rules."""
        matches: list[PatternMatch] = []
        lines = text.splitlines()
        for lineno, line_text in enumerate(lines, start=1):
            for category, patterns in self.rules.items():
                for pat_src, pat_re in patterns:
                    if pat_re.search(line_text):
                        matches.append(
                            PatternMatch(
                                category=category,
                                pattern=pat_src,
                                file=file,
                                line=lineno,
                                matched_text=line_text.strip(),
                            )
                        )
        return matches
