"""Exp 3: Sub-task ablation — remove each Layer 2 sub-task and measure impact.

Only runs on the ~151 skills that reach Layer 2 (L1 flags suspicious).
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from skillsieve.core.pipeline import Pipeline
from skillsieve.core.layer2.prompts import TASK_NAMES


async def run_ablation_config(
    skills_dir: Path,
    labeled_path: Path,
    task_names: list[str],
    config_name: str,
    profile: str = "kimi",
) -> dict:
    """Run L1+L2 with a specific subset of Layer 2 tasks."""
    dataset = json.loads(labeled_path.read_text())
    # Only run on skills that L1 flags as suspicious (these actually reach L2)
    from skillsieve.core.layer1.classifier import Layer1Classifier
    from skillsieve.parser import parse_skill_package

    classifier = Layer1Classifier()
    samples = []
    for s in dataset["samples"]:
        if s["label"] not in ("benign", "malicious"):
            continue
        skill_path = skills_dir / s["path"]
        if not skill_path.exists():
            continue
        try:
            pkg = parse_skill_package(skill_path)
            l1 = classifier.classify(pkg)
            if l1.verdict.value != "safe":  # Only suspicious skills reach L2
                samples.append(s)
        except Exception:
            pass

    print(f"  [{config_name}] {len(samples)} suspicious skills (L2-relevant)")

    pipeline = Pipeline(max_layer=2, layer2_profile=profile)

    tp = fp = tn = fn = 0

    # Patch both the source module AND the name already imported into
    # semantic_engine (import-time bindings hold a separate reference).
    with patch("skillsieve.core.layer2.prompts.TASK_NAMES", task_names), \
         patch("skillsieve.core.layer2.semantic_engine.TASK_NAMES", task_names):
        for i, sample in enumerate(samples):
            skill_path = skills_dir / sample["path"]

            try:
                result = await pipeline.scan(skill_path)
                pred_mal = result.final_verdict.value == "malicious"
                truth_mal = sample["label"] == "malicious"

                if pred_mal and truth_mal: tp += 1
                elif pred_mal and not truth_mal: fp += 1
                elif not pred_mal and not truth_mal: tn += 1
                elif not pred_mal and truth_mal: fn += 1

                if (i + 1) % 20 == 0:
                    print(f"  [{config_name}] {i+1}/{len(samples)}")
            except Exception:
                pass

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "config": config_name,
        "tasks": task_names,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


async def main(skills_dir: Path, labeled_path: Path, output_path: Path):
    all_tasks = list(TASK_NAMES)
    print(f"All tasks: {all_tasks}")

    configs = [
        ("Full (A+B+C+D)", all_tasks),
        ("-Task A (Intent)", [t for t in all_tasks if t != "intent_alignment"]),
        ("-Task B (Permission)", [t for t in all_tasks if t != "permission_justification"]),
        ("-Task C (Covert)", [t for t in all_tasks if t != "covert_behavior"]),
        ("-Task D (Cross-file)", [t for t in all_tasks if t != "cross_file_consistency"]),
    ]

    results = []
    for name, tasks in configs:
        print(f"\n=== {name} ({len(tasks)} tasks) ===")
        r = await run_ablation_config(skills_dir, labeled_path, tasks, name)
        results.append(r)
        print(f"  P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")

    # Compute deltas
    full_f1 = results[0]["f1"]
    for r in results:
        r["delta_f1"] = round(r["f1"] - full_f1, 4)

    output = {
        "experiment": "subtask_ablation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print(f"\n=== Summary ===")
    for r in results:
        delta = f"({r['delta_f1']:+.3f})" if r["delta_f1"] != 0 else ""
        print(f"  {r['config']:<25} F1={r['f1']:.3f} {delta}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills-dir", type=Path, default=Path("data/raw/openclaw-skills/skills"))
    parser.add_argument("--labeled", type=Path, default=Path("results/labeled_dataset_final.json"))
    parser.add_argument("--output", type=Path, default=Path("results/exp3_ablation.json"))
    args = parser.parse_args()
    asyncio.run(main(args.skills_dir, args.labeled, args.output))
