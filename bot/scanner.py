"""Scan logic: clone repo, parse skill, run SkillSieve pipeline."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from skillsieve.parser import parse_skill_package
from skillsieve.adapters.feishu_adapter import parse_feishu_package
from skillsieve.core.layer1.classifier import Layer1Classifier
from skillsieve.models import SkillPackage, Verdict


@dataclass
class ScanResult:
    """Result of a SkillSieve scan."""
    skill_name: str = ""
    verdict: str = "unknown"
    score: float = 0.0
    layer_stopped: int = 0
    evidence: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    l1_verdict: str = ""
    l1_score: float = 0.0
    l2_verdict: str = ""
    l2_score: float = 0.0
    cost: float = 0.0
    n_scripts: int = 0
    error: str = ""


def scan_skill_url(
    url: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ScanResult:
    """Clone a GitHub repo and scan it with SkillSieve.

    Parameters
    ----------
    url : str
        GitHub repository URL (e.g., https://github.com/user/repo)
    progress_callback : callable, optional
        Called with status string at each stage transition.
    """
    result = ScanResult()

    # Validate URL: only allow HTTPS GitHub URLs to prevent SSRF / local file access
    if not re.match(r"^https://github\.com/[\w\-\.]+/[\w\-\.]+", url):
        result.error = "仅支持 GitHub HTTPS URL"
        return result

    # Step 1: Clone
    if progress_callback:
        progress_callback("正在克隆仓库...")

    tmpdir = tempfile.mkdtemp(prefix="skillsieve_")
    try:
        repo_name = url.rstrip("/").split("/")[-1]
        clone_dir = Path(tmpdir) / repo_name

        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"  # no interactive prompts
        env["GIT_CONFIG_NOSYSTEM"] = "1"  # ignore system config
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", "-c", "core.hooksPath=/dev/null",
             url, str(clone_dir)],
            capture_output=True, text=True, timeout=60,
            env=env,
        )
        if proc.returncode != 0:
            result.error = f"Clone failed: {proc.stderr[:200]}"
            return result

        # Security: remove .git directory (hooks, config, objects) immediately
        git_dir = clone_dir / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir, ignore_errors=True)

        # Security: check total size (reject repos > 100MB to prevent disk exhaustion)
        total_size = sum(f.stat().st_size for f in clone_dir.rglob("*") if f.is_file())
        if total_size > 100 * 1024 * 1024:
            result.error = f"仓库过大 ({total_size // 1024 // 1024}MB)，跳过扫描"
            return result

        result.skill_name = repo_name

        # Step 2: Find the skill to scan
        # Check if the repo itself is a skill, or has skills/ subdirectory
        if progress_callback:
            progress_callback("Layer 1 静态分析中...")

        pkg = _find_and_parse_skill(clone_dir)
        if pkg is None:
            result.error = "未找到可扫描的 skill 包（无 SKILL.md 或 README.md）"
            return result

        result.skill_name = pkg.name
        result.n_scripts = len(pkg.script_files)

        # Step 3: Layer 1 scan
        result = _run_layer1(pkg, result)

        # Step 4: If suspicious, run Layer 2
        if result.l1_verdict == "suspicious":
            if progress_callback:
                progress_callback("Layer 2 LLM 语义分析中...")
            result = _run_layer2(pkg, result)

        return result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def scan_skill_text(
    text: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ScanResult:
    """Scan raw skill text content (SKILL.md content pasted by user)."""
    result = ScanResult()

    if progress_callback:
        progress_callback("Layer 1 静态分析中...")

    # Create a temporary skill package from text
    tmpdir = tempfile.mkdtemp(prefix="skillsieve_text_")
    try:
        skill_dir = Path(tmpdir) / "user_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")

        pkg = parse_skill_package(skill_dir)
        result.skill_name = pkg.name or "user_input"
        result.n_scripts = len(pkg.script_files)

        result = _run_layer1(pkg, result)

        if result.l1_verdict == "suspicious":
            if progress_callback:
                progress_callback("Layer 2 LLM 语义分析中...")
            result = _run_layer2(pkg, result)

        return result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _find_and_parse_skill(repo_dir: Path) -> Optional[SkillPackage]:
    """Find the best skill package to scan in a repo."""
    # Priority 1: Has SKILL.md at root
    if (repo_dir / "SKILL.md").exists():
        return parse_feishu_package(repo_dir)

    # Priority 2: Has skills/ subdirectory → scan the first skill
    skills_dir = repo_dir / "skills"
    if skills_dir.is_dir():
        for child in sorted(skills_dir.iterdir()):
            if child.is_dir() and (child / "SKILL.md").exists():
                return parse_feishu_package(child)

    # Priority 3: Has openclaw.plugin.json
    if (repo_dir / "openclaw.plugin.json").exists():
        return parse_feishu_package(repo_dir)

    # Priority 4: Has README.md + source code → standalone
    if (repo_dir / "README.md").exists():
        return parse_feishu_package(repo_dir)

    return None


def _run_layer1(pkg: SkillPackage, result: ScanResult) -> ScanResult:
    """Run Layer 1 static triage."""
    classifier = Layer1Classifier(threshold=0.3)
    l1_result = classifier.classify(pkg)

    result.l1_verdict = l1_result.verdict.value
    result.l1_score = round(l1_result.confidence, 4)
    result.evidence = [e[:200] for e in l1_result.evidence[:10]]

    if l1_result.verdict == Verdict.SAFE:
        result.verdict = "safe"
        result.score = result.l1_score
        result.layer_stopped = 1
        result.cost = 0.0
    else:
        result.verdict = "suspicious"
        result.score = result.l1_score
        result.layer_stopped = 1  # will be updated if L2 runs

    result.details["l1_model"] = l1_result.details.get("model_used", "heuristic")
    result.details["l1_pattern_count"] = l1_result.details.get("pattern_match_count", 0)

    return result


def _run_layer2(pkg: SkillPackage, result: ScanResult) -> ScanResult:
    """Run Layer 2 SSD analysis (if API keys available)."""
    # Check if we have any LLM API key configured
    has_key = any([
        os.environ.get("KIMI_API_KEY"),
        os.environ.get("MINIMAX_API_KEY"),
        os.environ.get("DEEPSEEK_API_KEY"),
    ])

    import logging
    logger = logging.getLogger("skillsieve-bot")

    logger.info(f"L2 check: KIMI_API_KEY={'set' if os.environ.get('KIMI_API_KEY') else 'NOT SET'}")

    if not has_key:
        logger.info("L2 skipped: no API key found")
        result.details["l2_skipped"] = "no_api_key"
        result.layer_stopped = 1
        return result

    logger.info("L2 starting: importing SemanticEngine...")
    try:
        import asyncio
        from skillsieve.core.layer2.semantic_engine import SemanticEngine
        engine = SemanticEngine(profile="kimi")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            l2_result = loop.run_until_complete(
                engine.analyze(pkg, layer1_evidence=result.evidence)
            )
        finally:
            loop.close()

        result.l2_verdict = l2_result.verdict.value
        result.l2_score = round(l2_result.confidence, 4)
        result.layer_stopped = 2
        result.cost = 0.04  # approximate

        # Update final verdict based on L2
        if l2_result.verdict == Verdict.SAFE:
            result.verdict = "safe"
            result.score = result.l2_score
        elif l2_result.verdict == Verdict.MALICIOUS:
            result.verdict = "malicious"
            result.score = result.l2_score
        else:
            result.verdict = "suspicious"
            result.score = result.l2_score

        # Merge evidence
        result.evidence.extend([e[:200] for e in l2_result.evidence[:5]])

    except Exception as e:
        logger.exception(f"L2 failed: {e}")
        result.details["l2_error"] = str(e)[:200]
        result.layer_stopped = 1

    return result
