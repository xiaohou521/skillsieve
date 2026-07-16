"""Tests for skillsieve.core.layer2.prompts."""
from __future__ import annotations

import pytest

from skillsieve.core.layer2.prompts import TASK_NAMES, build_prompt


EXPECTED_TASK_NAMES = [
    "intent_alignment",
    "permission_justification",
    "covert_behavior",
    "cross_file_consistency",
]


def test_all_four_tasks_exist():
    """TASK_NAMES must contain exactly the 4 expected task identifiers."""
    assert sorted(TASK_NAMES) == sorted(EXPECTED_TASK_NAMES)
    assert len(TASK_NAMES) == 4


def test_build_prompt_contains_skill_content():
    """build_prompt must include the skill_md text, script contents, L1 flags, and task title."""
    skill_md = "## My Test Skill\nDoes something helpful."
    scripts = {
        "main.py": "import os\nprint('hello')",
        "helper.sh": "#!/bin/bash\necho hi",
    }
    flags = ["pattern_match: exec()", "suspicious_import: subprocess"]

    for task in TASK_NAMES:
        prompt = build_prompt(
            task=task,
            skill_md=skill_md,
            script_contents=scripts,
            layer1_flags=flags,
        )

        # skill_md content present
        assert "My Test Skill" in prompt, f"[{task}] skill_md not found in prompt"

        # script filenames and content present
        assert "main.py" in prompt, f"[{task}] main.py filename not in prompt"
        assert "import os" in prompt, f"[{task}] script content not in prompt"
        assert "helper.sh" in prompt, f"[{task}] helper.sh not in prompt"

        # layer1 flags present
        assert "pattern_match: exec()" in prompt, f"[{task}] L1 flag not in prompt"
        assert "suspicious_import: subprocess" in prompt, f"[{task}] L1 flag not in prompt"

        # task title present (each task has a unique title string)
        assert "Analysis" in prompt, f"[{task}] task title not found in prompt"


def test_build_prompt_has_json_output_instruction():
    """build_prompt must instruct the LLM to output JSON."""
    prompt = build_prompt(
        task="intent_alignment",
        skill_md="Minimal skill",
        script_contents={},
        layer1_flags=[],
    )
    assert "JSON" in prompt
