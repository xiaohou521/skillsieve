"""Exp 4: Adversarial Robustness — per-technique detection rates."""
import asyncio
import json
from pathlib import Path

from skillsieve.core.pipeline import Pipeline


async def run_adversarial_eval(
    adversarial_dir: Path,
    max_layer: int = 3,
    output_path: Path | None = None,
) -> dict:
    """Evaluate detection on adversarial samples organized by technique subdirs.

    Expected structure:
        adversarial_dir/
            encoding_000/SKILL.md
            encoding_001/SKILL.md
            cross_file_split_000/SKILL.md
            ...
    """
    pipeline = Pipeline(max_layer=max_layer)

    # Group by technique (prefix before _NNN)
    technique_results: dict[str, dict] = {}

    skill_dirs = sorted(d for d in Path(adversarial_dir).iterdir()
                        if d.is_dir() and (d / "SKILL.md").exists())

    for skill_dir in skill_dirs:
        name = skill_dir.name
        # Extract technique from name (e.g., "encoding_000" -> "encoding")
        parts = name.rsplit("_", 1)
        technique = parts[0] if len(parts) == 2 and parts[1].isdigit() else name

        if technique not in technique_results:
            technique_results[technique] = {"total": 0, "detected": 0, "missed": 0}

        result = await pipeline.scan(skill_dir)
        technique_results[technique]["total"] += 1

        if result.final_verdict.value == "malicious":
            technique_results[technique]["detected"] += 1
        else:
            technique_results[technique]["missed"] += 1

    # Compute detection rates
    for tech, counts in technique_results.items():
        counts["detection_rate"] = round(
            counts["detected"] / counts["total"], 4
        ) if counts["total"] > 0 else 0.0

    output = {"experiment": "adversarial_robustness", "max_layer": max_layer,
              "techniques": technique_results}

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 4: Adversarial Robustness")
    parser.add_argument("adversarial_dir", type=Path)
    parser.add_argument("--max-layer", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/exp4_adversarial.json"))
    args = parser.parse_args()
    result = asyncio.run(run_adversarial_eval(args.adversarial_dir, args.max_layer, args.output))
    for tech, data in result["techniques"].items():
        print(f"{tech}: {data['detected']}/{data['total']} ({data['detection_rate']:.0%})")
