"""Evaluate SkillSieve on the labeled dataset — compute P/R/F1 at each layer config."""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from skillsieve.core.pipeline import Pipeline


async def evaluate(
    skills_dir: Path,
    labeled_path: Path,
    output_path: Path,
    max_layer: int = 2,
    profile: str = "kimi",
):
    """Run SkillSieve on labeled dataset and compute metrics."""
    dataset = json.loads(labeled_path.read_text())
    samples = [s for s in dataset["samples"] if s["label"] in ("benign", "malicious")]

    print(f"Evaluating SkillSieve (max_layer={max_layer}) on {len(samples)} labeled skills...")

    pipeline = Pipeline(max_layer=max_layer, layer2_profile=profile)
    predictions = []

    for i, sample in enumerate(samples):
        skill_path = skills_dir / sample["path"]
        if not skill_path.exists():
            continue

        print(f"  [{i+1}/{len(samples)}] {sample['path']}...", end=" ", flush=True)

        try:
            start = time.monotonic()
            result = await pipeline.scan(skill_path)
            elapsed = time.monotonic() - start

            pred = "malicious" if result.final_verdict.value == "malicious" else "benign"
            predictions.append({
                "path": sample["path"],
                "ground_truth": sample["label"],
                "prediction": pred,
                "confidence": result.final_confidence,
                "layer_stopped": result.layer_stopped,
                "latency_s": round(elapsed, 1),
            })
            match = "OK" if pred == sample["label"] else "WRONG"
            print(f"→ {pred} (truth={sample['label']}, {match}, {elapsed:.1f}s)")

        except Exception as e:
            print(f"→ ERROR: {e}")
            predictions.append({
                "path": sample["path"],
                "ground_truth": sample["label"],
                "prediction": "error",
                "error": str(e),
            })

    # Compute metrics
    tp = fp = tn = fn = 0
    for p in predictions:
        if p["prediction"] == "error":
            continue
        pred_mal = p["prediction"] == "malicious"
        truth_mal = p["ground_truth"] == "malicious"
        if pred_mal and truth_mal: tp += 1
        elif pred_mal and not truth_mal: fp += 1
        elif not pred_mal and not truth_mal: tn += 1
        elif not pred_mal and truth_mal: fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / total, 4) if total > 0 else 0,
        "fpr": round(fpr, 4),
    }

    # Layer distribution
    layer_dist = {}
    for p in predictions:
        if "layer_stopped" in p:
            l = p["layer_stopped"]
            layer_dist[l] = layer_dist.get(l, 0) + 1

    output = {
        "method": "skillsieve",
        "max_layer": max_layer,
        "llm_profile": profile,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_size": len(samples),
        "evaluated": len([p for p in predictions if p["prediction"] != "error"]),
        "metrics": metrics,
        "layer_distribution": layer_dist,
        "predictions": predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print(f"\n=== Results (max_layer={max_layer}) ===")
    print(f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} Acc={metrics['accuracy']:.3f} FPR={metrics['fpr']:.3f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Layer distribution: {layer_dist}")
    print(f"Saved to {output_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills-dir", type=Path, default=Path("data/raw/openclaw-skills/skills"))
    parser.add_argument("--labeled", type=Path, default=Path("results/labeled_dataset_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/eval_skillsieve.json"))
    parser.add_argument("--max-layer", type=int, default=2)
    parser.add_argument("--profile", default="kimi")
    args = parser.parse_args()
    asyncio.run(evaluate(args.skills_dir, args.labeled, args.output, args.max_layer, args.profile))
