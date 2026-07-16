"""Layer 2 prompt templates for LLM-based semantic analysis sub-tasks."""
from __future__ import annotations

TASK_NAMES = [
    "intent_alignment",
    "permission_justification",
    "covert_behavior",
    "cross_file_consistency",
]

SYSTEM_PROMPT = (
    "You are a security analyst specializing in AI agent skill packages. "
    "Analyze the provided skill for security threats. Be precise and cite evidence. "
    "Output strict JSON only — no explanation outside the JSON object."
)

_TASK_INSTRUCTIONS: dict[str, dict[str, str]] = {
    "intent_alignment": {
        "title": "Intent Alignment Analysis",
        "steps": (
            "1. What does this skill CLAIM to do based on its name, description, and SKILL.md?\n"
            "2. What does the skill ACTUALLY do based on its scripts and instructions?\n"
            "3. List any discrepancies between claimed purpose and actual behavior.\n"
            "4. Determine if the skill is ALIGNED, PARTIALLY_MISALIGNED, or FULLY_MISALIGNED.\n"
            "5. Assign a risk_score from 0.0 (safe) to 1.0 (highly suspicious).\n"
            "6. Cite specific evidence from the skill files."
        ),
        "output_schema": (
            '{"claimed_purpose": "string", "actual_instructions": ["string", "..."], '
            '"discrepancies": ["string", "..."], '
            '"alignment": "ALIGNED|PARTIALLY_MISALIGNED|FULLY_MISALIGNED", '
            '"risk_score": 0.0, "evidence": ["string", "..."]}'
        ),
    },
    "permission_justification": {
        "title": "Permission Justification Analysis",
        "steps": (
            "1. List all permissions, environment variables, and binaries this skill requests.\n"
            "2. For each permission/resource, determine if it is justified by the stated purpose.\n"
            "3. Flag any permissions that seem excessive or unjustified.\n"
            "4. Assess whether the combination of permissions could enable harmful actions.\n"
            "5. Assign a risk_score from 0.0 (safe) to 1.0 (highly suspicious).\n"
            "6. Cite specific evidence from the skill files."
        ),
        "output_schema": (
            '{"requested_permissions": ["string", "..."], '
            '"justified_permissions": ["string", "..."], '
            '"unjustified_permissions": ["string", "..."], '
            '"risk_score": 0.0, "evidence": ["string", "..."]}'
        ),
    },
    "covert_behavior": {
        "title": "Covert Behavior Analysis",
        "steps": (
            "1. Look for any instructions or code that would operate without user awareness.\n"
            "2. Identify data exfiltration patterns (sending data to external URLs, writing to hidden locations).\n"
            "3. Check for obfuscated code, encoded strings, or dynamic code execution.\n"
            "4. Look for persistence mechanisms (cron jobs, startup scripts, registry modifications).\n"
            "5. Identify any social engineering or deceptive instructions.\n"
            "6. Assign a risk_score from 0.0 (safe) to 1.0 (highly suspicious).\n"
            "7. Cite specific evidence from the skill files."
        ),
        "output_schema": (
            '{"covert_patterns_found": ["string", "..."], '
            '"exfiltration_indicators": ["string", "..."], '
            '"obfuscation_indicators": ["string", "..."], '
            '"persistence_indicators": ["string", "..."], '
            '"risk_score": 0.0, "evidence": ["string", "..."]}'
        ),
    },
    "cross_file_consistency": {
        "title": "Cross-File Consistency Analysis",
        "steps": (
            "1. Compare the metadata in SKILL.md with what the scripts actually implement.\n"
            "2. Check if any scripts perform actions not mentioned in SKILL.md.\n"
            "3. Look for hidden scripts or files that contradict the stated functionality.\n"
            "4. Verify that the version and dependency information is internally consistent.\n"
            "5. Identify any inconsistencies between different script files.\n"
            "6. Assign a risk_score from 0.0 (safe) to 1.0 (highly suspicious).\n"
            "7. Cite specific evidence from the skill files."
        ),
        "output_schema": (
            '{"consistent_elements": ["string", "..."], '
            '"inconsistencies": ["string", "..."], '
            '"undocumented_behaviors": ["string", "..."], '
            '"risk_score": 0.0, "evidence": ["string", "..."]}'
        ),
    },
}


def build_prompt(
    task: str,
    skill_md: str,
    script_contents: dict[str, str],
    layer1_flags: list[str],
) -> str:
    """Build the full prompt for a given analysis task.

    Parameters
    ----------
    task:
        One of the keys in :data:`TASK_NAMES`.
    skill_md:
        Full text content of the skill's SKILL.md file.
    script_contents:
        Mapping of filename to file content for all script files.
    layer1_flags:
        List of evidence strings produced by Layer 1 analysis.

    Returns
    -------
    str
        The assembled prompt string ready to pass to the LLM.
    """
    if task not in _TASK_INSTRUCTIONS:
        raise ValueError(f"Unknown task: {task!r}. Must be one of {TASK_NAMES}")

    instructions = _TASK_INSTRUCTIONS[task]

    parts: list[str] = []

    # --- SKILL.md section (truncate to 4000 chars) ---
    MAX_MD = 4000
    MAX_SCRIPT = 2000
    MAX_TOTAL_SCRIPTS = 8000

    parts.append("=== SKILL.md CONTENT ===")
    md_text = skill_md.strip() if skill_md else "(empty)"
    if len(md_text) > MAX_MD:
        md_text = md_text[:MAX_MD] + "\n... [truncated]"
    parts.append(md_text)
    parts.append("")

    # --- Script files section (truncate each file and total) ---
    if script_contents:
        parts.append("=== SCRIPT FILES ===")
        total_script_chars = 0
        for filename, content in script_contents.items():
            if total_script_chars >= MAX_TOTAL_SCRIPTS:
                parts.append(f"--- {filename} ---")
                parts.append("... [skipped, content budget exceeded]")
                parts.append("")
                continue
            parts.append(f"--- {filename} ---")
            text = content.strip() if content else "(empty)"
            if len(text) > MAX_SCRIPT:
                text = text[:MAX_SCRIPT] + "\n... [truncated]"
            parts.append(text)
            total_script_chars += len(text)
            parts.append("")
    else:
        parts.append("=== SCRIPT FILES ===")
        parts.append("(none)")
        parts.append("")

    # --- Layer 1 flags section ---
    parts.append("=== LAYER 1 FLAGS (heuristic analysis results) ===")
    if layer1_flags:
        for flag in layer1_flags:
            parts.append(f"- {flag}")
    else:
        parts.append("(none)")
    parts.append("")

    # --- Task instructions ---
    parts.append(f"=== TASK: {instructions['title']} ===")
    parts.append(instructions["steps"])
    parts.append("")

    # --- Output schema ---
    parts.append("=== REQUIRED JSON OUTPUT FORMAT ===")
    parts.append(
        "Respond with ONLY a valid JSON object matching this schema (no other text):"
    )
    parts.append(instructions["output_schema"])

    return "\n".join(parts)
