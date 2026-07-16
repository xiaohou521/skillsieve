"""Exp 5: Efficiency & Cost Analysis."""
import asyncio
import json
import time
from pathlib import Path

from skillsieve.core.pipeline import Pipeline


async def run_efficiency_analysis(
    skills_dir: Path,
    output_path: Path | None = None,
) -> dict:
    """Measure per-skill latency and layer distribution."""
    skill_dirs = sorted(d for d in Path(skills_dir).iterdir()
                        if d.is_dir() and (d / "SKILL.md").exists())

    layer_counts = {1: 0, 2: 0, 3: 0}
    latencies = []
    per_layer_latencies: dict[int, list[float]] = {1: [], 2: [], 3: []}

    # Estimate costs: Layer 1 = $0, Layer 2 = ~$0.01, Layer 3 = ~$0.10
    COST_PER_LAYER = {1: 0.0, 2: 0.01, 3: 0.10}
    total_cost = 0.0

    pipeline = Pipeline(max_layer=3)

    for skill_dir in skill_dirs:
        start = time.monotonic()
        result = await pipeline.scan(skill_dir)
        elapsed_ms = (time.monotonic() - start) * 1000

        layer = result.layer_stopped
        layer_counts[layer] += 1
        latencies.append(elapsed_ms)
        per_layer_latencies[layer].append(elapsed_ms)
        total_cost += COST_PER_LAYER.get(layer, 0)

    total = len(skill_dirs)

    def stats(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
        s = sorted(values)
        return {
            "count": len(s),
            "avg": round(sum(s) / len(s), 1),
            "p50": round(s[len(s) // 2], 1),
            "p95": round(s[int(len(s) * 0.95)], 1) if len(s) >= 20 else round(s[-1], 1),
            "p99": round(s[int(len(s) * 0.99)], 1) if len(s) >= 100 else round(s[-1], 1),
        }

    output = {
        "experiment": "efficiency",
        "total_skills": total,
        "layer_distribution": {
            f"layer_{k}": {"count": v, "pct": round(v / total * 100, 1) if total > 0 else 0}
            for k, v in layer_counts.items()
        },
        "overall_latency": stats(latencies),
        "per_layer_latency": {f"layer_{k}": stats(v) for k, v in per_layer_latencies.items()},
        "cost": {
            "total": round(total_cost, 2),
            "avg_per_skill": round(total_cost / total, 4) if total > 0 else 0,
        },
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 5: Efficiency Analysis")
    parser.add_argument("skills_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/exp5_efficiency.json"))
    args = parser.parse_args()
    result = asyncio.run(run_efficiency_analysis(args.skills_dir, args.output))
    print(json.dumps(result, indent=2))
