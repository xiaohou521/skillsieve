"""Exp 2: Layer Ablation — measure each layer's contribution."""
import asyncio
import json
from pathlib import Path

from experiments.exp1_main_results import run_experiment


async def run_ablation(
    skills_dir: Path,
    labels_path: Path,
    output_path: Path | None = None,
) -> dict:
    configs = [
        {"name": "L1_only", "max_layer": 1},
        {"name": "L1_L2", "max_layer": 2},
        {"name": "L1_L2_L3", "max_layer": 3},
    ]

    results = {}
    for cfg in configs:
        print(f"Running {cfg['name']}...")
        out = await run_experiment(skills_dir, labels_path, max_layer=cfg["max_layer"])
        results[cfg["name"]] = out["metrics"]

    output = {"experiment": "layer_ablation", "results": results}

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 2: Layer Ablation")
    parser.add_argument("skills_dir", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/exp2_ablation.json"))
    args = parser.parse_args()
    result = asyncio.run(run_ablation(args.skills_dir, args.labels, args.output))
    for name, metrics in result["results"].items():
        print(f"{name}: F1={metrics['f1']}, Recall={metrics['recall']}")
