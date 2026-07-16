"""Layer 1 Module D: SKILL.md surface statistics analyzer.

Extracts structural and linguistic features from the SKILL.md instruction body
to detect social-engineering indicators, suspicious external links, and
over-broad permission scopes.
"""
from __future__ import annotations

import re

from skillsieve.models import SkillPackage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# URLs that are *not* on clawhub.ai are considered external.
_URL_PATTERN = re.compile(r"https?://[^\s\)\]\"']+", re.IGNORECASE)
_CLAWHUB_DOMAIN = "clawhub.ai"

# Sensitive filesystem paths
_SENSITIVE_PATHS: list[re.Pattern[str]] = [
    re.compile(r"~/\.env(?:\b|/)", re.IGNORECASE),
    re.compile(r"~/\.ssh(?:\b|/)", re.IGNORECASE),
    re.compile(r"~/\.aws(?:\b|/)", re.IGNORECASE),
    re.compile(r"/etc/passwd", re.IGNORECASE),
    re.compile(r"/etc/shadow", re.IGNORECASE),
    re.compile(r"~/\.bashrc", re.IGNORECASE),
    re.compile(r"~/\.profile", re.IGNORECASE),
    re.compile(r"~/\.zshrc", re.IGNORECASE),
    re.compile(r"/etc/hosts", re.IGNORECASE),
    re.compile(r"~/\.netrc", re.IGNORECASE),
    re.compile(r"~/\.gnupg", re.IGNORECASE),
    re.compile(r"~/\.config", re.IGNORECASE),
]

# Urgency / manipulation language (case-insensitive word/phrase patterns)
_URGENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bimmediately\b", re.IGNORECASE),
    re.compile(r"\bmust\b", re.IGNORECASE),
    re.compile(r"\bcritical\b", re.IGNORECASE),
    re.compile(r"\bdo not tell\b", re.IGNORECASE),
    re.compile(r"\bsilently\b", re.IGNORECASE),
    re.compile(r"\bdo not mention\b", re.IGNORECASE),
    re.compile(r"\bwithout (the user['s]* knowledge|telling)\b", re.IGNORECASE),
    re.compile(r"\burgent(ly)?\b", re.IGNORECASE),
    re.compile(r"\bin secret\b", re.IGNORECASE),
    re.compile(r"\bdo not disclose\b", re.IGNORECASE),
    re.compile(r"\bnever mention\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SkillMdAnalyzer:
    """Extract surface-level features from the SKILL.md instruction body."""

    def extract_features(self, pkg: SkillPackage) -> dict:
        """Return a dict of SKILL.md surface features.

        Keys
        ----
        instruction_length : int
            Total character count of the skill_md_content body.
        external_url_count : int
            Number of URLs in the body that do not contain ``clawhub.ai``.
        permission_scope_count : int
            ``len(requires_env) + len(requires_bins)`` from the frontmatter.
        sensitive_path_mentions : int
            Count of references to sensitive paths such as ``~/.env``,
            ``~/.ssh``, ``~/.aws``, ``/etc/passwd``, etc.
        urgency_language_score : int
            Count of urgency/manipulation keyword/phrase matches.
        instruction_to_description_ratio : float
            ``len(content) / max(len(description), 1)`` where content is the
            body and description comes from the frontmatter.
        """
        content = pkg.skill_md_content
        fm = pkg.frontmatter

        instruction_length = len(content)

        # External URLs (not on clawhub.ai)
        all_urls = _URL_PATTERN.findall(content)
        external_url_count = sum(
            1 for url in all_urls if _CLAWHUB_DOMAIN not in url.lower()
        )

        # Permission scope
        permission_scope_count = len(fm.requires_env) + len(fm.requires_bins)

        # Sensitive path mentions
        sensitive_path_mentions = sum(
            len(pattern.findall(content)) for pattern in _SENSITIVE_PATHS
        )

        # Urgency / social-engineering language
        urgency_language_score = sum(
            len(pattern.findall(content)) for pattern in _URGENCY_PATTERNS
        )

        # Instruction-to-description ratio
        description = fm.description
        instruction_to_description_ratio = instruction_length / max(len(description), 1)

        return {
            "instruction_length": instruction_length,
            "external_url_count": external_url_count,
            "permission_scope_count": permission_scope_count,
            "sensitive_path_mentions": sensitive_path_mentions,
            "urgency_language_score": urgency_language_score,
            "instruction_to_description_ratio": instruction_to_description_ratio,
        }
