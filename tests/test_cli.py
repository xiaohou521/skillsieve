from pathlib import Path
from click.testing import CliRunner
from skillsieve.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_scan_benign():
    runner = CliRunner()
    result = runner.invoke(main, ["scan", str(FIXTURES / "benign_skill")])
    assert result.exit_code == 0
    assert "SAFE" in result.output


def test_cli_scan_with_max_layer():
    runner = CliRunner()
    result = runner.invoke(main, ["scan", str(FIXTURES / "malicious_skill"), "--max-layer", "1"])
    assert result.exit_code == 0


def test_cli_scan_nonexistent():
    runner = CliRunner()
    result = runner.invoke(main, ["scan", "/nonexistent/path"])
    assert result.exit_code != 0
