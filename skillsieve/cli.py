"""Click-based CLI for SkillSieve."""
import asyncio
import json
import sys
from pathlib import Path

import click

from skillsieve.core.pipeline import Pipeline
from skillsieve.models import Verdict


@click.group()
@click.version_option()
def main():
    """SkillSieve - Detect malicious AI agent skills."""
    pass


@main.command()
@click.argument("skill_path", type=click.Path(exists=True))
@click.option("--max-layer", type=int, default=3, help="Maximum analysis layer (1-3)")
@click.option("--report", is_flag=True, help="Save detailed report for malicious skills")
@click.option("--json-output", is_flag=True, help="Output result as JSON")
@click.option("--layer2-profile", default="kimi", help="LLM profile for Layer 2 (kimi/minimax/baidu)")
def scan(skill_path, max_layer, report, json_output, layer2_profile):
    """Scan a skill package for security threats."""
    path = Path(skill_path)
    if not (path / "SKILL.md").exists():
        click.echo(f"Error: No SKILL.md found in {path}", err=True)
        sys.exit(1)

    pipeline = Pipeline(max_layer=max_layer, layer2_profile=layer2_profile)
    result = asyncio.run(pipeline.scan(path))

    if json_output:
        click.echo(result.model_dump_json(indent=2))
        return

    verdict_icon = {
        "safe": "SAFE",
        "suspicious": "SUSPICIOUS",
        "malicious": "MALICIOUS",
    }[result.final_verdict.value]
    click.echo(f"{verdict_icon} (Layer {result.layer_stopped}, confidence: {result.final_confidence:.2f})")

    if result.report and report:
        report_path = path / "skillsieve-report.md"
        report_path.write_text(result.report)
        click.echo(f"Report saved to {report_path}")
    elif result.report:
        click.echo(result.report)


@main.command("scan-batch")
@click.argument("skills_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
@click.option("--max-layer", type=int, default=3)
def scan_batch(skills_dir, output, max_layer):
    """Scan all skill packages in a directory."""
    root = Path(skills_dir)
    skill_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]
    if not skill_dirs:
        click.echo("No skill packages found.")
        return

    pipeline = Pipeline(max_layer=max_layer)
    results = []
    with click.progressbar(skill_dirs, label="Scanning") as bar:
        for skill_dir in bar:
            result = asyncio.run(pipeline.scan(skill_dir))
            results.append(result.model_dump())

    malicious_count = sum(1 for r in results if r["final_verdict"] == "malicious")
    click.echo(f"\nScanned {len(results)} skills. {malicious_count} malicious detected.")

    if output:
        Path(output).write_text(json.dumps(results, indent=2, default=str))
        click.echo(f"Results saved to {output}")
