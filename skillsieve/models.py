from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"


class AttackType(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    CREDENTIAL_THEFT = "credential_theft"
    REMOTE_EXEC = "remote_exec"
    DATA_EXFIL = "data_exfil"
    TYPOSQUATTING = "typosquatting"
    OBFUSCATION = "obfuscation"
    SOCIAL_ENGINEERING = "social_engineering"


class SkillFrontmatter(BaseModel):
    name: str = ""
    description: str = ""
    version: str = ""
    requires_env: list[str] = Field(default_factory=list)
    requires_bins: list[str] = Field(default_factory=list)


class SkillPackage(BaseModel):
    path: Path
    name: str
    frontmatter: SkillFrontmatter = Field(default_factory=SkillFrontmatter)
    skill_md_content: str = ""
    script_files: dict[str, str] = Field(default_factory=dict)
    # Relative paths of files whose symlink target escapes the skill directory.
    # Their content is deliberately NOT read; recorded as a critical Layer 1 signal.
    symlink_escapes: list[str] = Field(default_factory=list)


class LayerResult(BaseModel):
    layer: int
    verdict: Verdict
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ScanResult(BaseModel):
    skill_name: str
    final_verdict: Verdict
    final_confidence: float
    layer_stopped: int
    layer_results: list[LayerResult] = Field(default_factory=list)
    report: Optional[str] = None
