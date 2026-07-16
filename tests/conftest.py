from pathlib import Path

import pytest

from skillsieve.parser import parse_skill_package

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def benign_pkg():
    return parse_skill_package(FIXTURES_DIR / "benign_skill")


@pytest.fixture
def malicious_pkg():
    return parse_skill_package(FIXTURES_DIR / "malicious_skill")


@pytest.fixture
def obfuscated_pkg():
    return parse_skill_package(FIXTURES_DIR / "obfuscated_skill")
