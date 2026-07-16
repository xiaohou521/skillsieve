"""Exp 1: Main Results — overall performance comparison."""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from skillsieve.core.pipeline import Pipeline


def load_labels(labels_path: Path) -> dict[str, str]:
    """Load ground truth labels from the benchmark JSON.

    Supports two formats:
    - labeled_dataset_final.json (nested): {"samples": [{"path": "author/skill", "label": ...}]}
    - simple dict: {"author/skill": "malicious", ...}
    """
    data = json.loads(labels_path.read_text())
    if "samples" in data:
        return {s["path"]: s["label"] for s in data["samples"]}
    return data


def find_skill_dirs(skills_dir: Path) -> list[tuple[str, Path]]:
    """Find all skill directories (author/skill with SKILL.md).

    Returns list of (relative_key, absolute_path) tuples where
    relative_key is 'author/skill' matching the label format.
    """
    results = []
    skills_root = Path(skills_dir)
    for author_dir in sorted(skills_root.iterdir()):
        if not author_dir.is_dir():
            continue
        # Check if this is itself a skill directory (flat layout)
        if (author_dir / "SKILL.md").exists():
            results.append((author_dir.name, author_dir))
            continue
        # Otherwise, look one level deeper (author/skill layout)
        for skill_dir in sorted(author_dir.iterdir()):
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                key = f"{author_dir.name}/{skill_dir.name}"
                results.append((key, skill_dir))
    return results


def compute_metrics(predictions: dict[str, str], labels: dict[str, str]) -> dict:
    """Compute precision, recall, F1, accuracy, FPR.

    Only evaluates skills present in both predictions and labels.
    """
    tp = fp = tn = fn = 0
    matched = 0
    for key, pred in predictions.items():
        if key not in labels:
            continue
        matched += 1
        truth = labels[key]
        if pred == "malicious" and truth == "malicious":
            tp += 1
        elif pred == "malicious" and truth == "benign":
            fp += 1
        elif pred == "safe" and truth == "benign":
            tn += 1
        elif pred == "safe" and truth == "malicious":
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "matched": matched,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "fpr": round(fpr, 4),
    }


async def run_experiment(
    skills_dir: Path,
    labels_path: Path,
    max_layer: int = 3,
    output_path: Path | None = None,
    layer2_profile: str = "kimi",
) -> dict:
    labels = load_labels(labels_path)
    pipeline = Pipeline(max_layer=max_layer, layer2_profile=layer2_profile)
    predictions = {}
    latencies = []

    skill_entries = find_skill_dirs(skills_dir)
    # Only scan skills that are in the label set
    labeled_entries = [(key, path) for key, path in skill_entries if key in labels]

    print(f"Found {len(skill_entries)} skill dirs, {len(labeled_entries)} in label set")

    errors: dict[str, str] = {}
    for i, (key, skill_dir) in enumerate(labeled_entries):
        print(f"  [{i+1}/{len(labeled_entries)}] {key}...", end=" ", flush=True)
        start = time.monotonic()
        try:
            result = await pipeline.scan(skill_dir)
            elapsed = (time.monotonic() - start) * 1000
            latencies.append(elapsed)
            verdict = "malicious" if result.final_verdict.value == "malicious" else "safe"
            predictions[key] = verdict
            print(f"→ {verdict} ({elapsed:.0f}ms)")
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            err = f"{type(e).__name__}: {e}"[:200]
            errors[key] = err
            # Fall back to safe so the skill is still counted.
            predictions[key] = "safe"
            print(f"✗ ERROR ({elapsed:.0f}ms): {err}")

    metrics = compute_metrics(predictions, labels)
    metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else 0
    metrics["total_skills"] = len(labeled_entries)
    metrics["total_in_label_set"] = len(labels)

    output = {"method": "skillsieve", "max_layer": max_layer, "metrics": metrics,
              "predictions": predictions, "errors": errors}
    metrics["n_errors"] = len(errors)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 1: Main Results")
    parser.add_argument("skills_dir", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--max-layer", type=int, default=3)
    parser.add_argument("--layer2-profile", type=str, default="kimi")
    parser.add_argument("--output", type=Path, default=Path("results/exp1_results.json"))
    args = parser.parse_args()
    result = asyncio.run(run_experiment(args.skills_dir, args.labels, args.max_layer, args.output, args.layer2_profile))
    print(json.dumps(result["metrics"], indent=2))
