"""Layer 1 Module C: Metadata reputation checker with typosquatting detection.

Checks skill frontmatter for suspicious indicators:
- Levenshtein distance to popular skill names (typosquatting)
- Sensitive environment variable requirements
- Dangerous binary requirements
- API-sourced reputation signals (defaulted to sentinel values)
"""
from __future__ import annotations

import Levenshtein

from skillsieve.models import SkillPackage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SENSITIVE_ENV_KEYWORDS: frozenset[str] = frozenset({
    "key", "token", "secret", "password", "credential",
})

_DANGEROUS_BINS: frozenset[str] = frozenset({
    "curl", "wget", "nc", "netcat", "ncat",
    "ssh", "scp", "nmap",
    "python", "python3", "node", "bash",
})


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MetadataChecker:
    """Extract metadata-based security features from a :class:`SkillPackage`.

    Parameters
    ----------
    top_skill_names:
        List of popular / known-good skill names used as reference for
        typosquatting detection.
    """

    def __init__(self, top_skill_names: list[str]) -> None:
        self.top_skill_names: list[str] = list(top_skill_names)

    def extract_features(self, pkg: SkillPackage) -> dict:
        """Return a flat dict of metadata-derived features.

        Keys
        ----
        name_edit_distance : int
            Minimum Levenshtein distance between ``pkg.name`` and any entry
            in ``top_skill_names``.  0 means exact match.
        requires_sensitive_env : bool
            True if any required environment variable name contains one of
            the sensitive keywords (case-insensitive).
        requires_dangerous_bins : bool
            True if any required binary is in the dangerous set.
        num_required_env : int
            Number of entries in ``requires_env``.
        num_required_bins : int
            Number of entries in ``requires_bins``.
        author_age_days : int
            Days since the author account was created (-1 = unknown).
        author_skill_count : int
            Number of skills published by the same author (-1 = unknown).
        download_count : int
            Total download count for this skill (-1 = unknown).
        has_verified_badge : bool
            Whether the author has a verified badge (False = unknown/no).
        """
        fm = pkg.frontmatter
        skill_name = pkg.name

        # Typosquatting: minimum edit distance to any popular skill name
        if self.top_skill_names:
            name_edit_distance = min(
                Levenshtein.distance(skill_name, ref)
                for ref in self.top_skill_names
            )
        else:
            name_edit_distance = -1

        # Sensitive environment variables
        requires_sensitive_env = any(
            any(kw in env_var.lower() for kw in _SENSITIVE_ENV_KEYWORDS)
            for env_var in fm.requires_env
        )

        # Dangerous binary requirements
        requires_dangerous_bins = any(
            bin_name.lower() in _DANGEROUS_BINS
            for bin_name in fm.requires_bins
        )

        # Homoglyph detection: check for non-ASCII characters in skill name
        has_homoglyph = any(ord(c) > 127 for c in skill_name)

        return {
            "name_edit_distance": name_edit_distance,
            "has_homoglyph": has_homoglyph,
            "requires_sensitive_env": requires_sensitive_env,
            "requires_dangerous_bins": requires_dangerous_bins,
            "num_required_env": len(fm.requires_env),
            "num_required_bins": len(fm.requires_bins),
            # API-sourced fields — not yet implemented; use sentinel values
            "author_age_days": -1,
            "author_skill_count": -1,
            "download_count": -1,
            "has_verified_badge": False,
        }
