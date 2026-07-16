"""Run ClawVet baseline on the labeled dataset for comparison."""
import json
import subprocess
import sys
import time
from pathlib import Path


def scan_with_clawvet(skill_dir: Path) -> dict:
    """Run ClawVet on a single skill directory and return parsed result."""
    try:
        result = subprocess.run(
            ["npx", "clawvet", "scan", str(skill_dir), "--format", "json"],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        risk_score = data.get("riskScore", 0)
        findings = data.get("findingsCount", {})
        critical = findings.get("critical", 0)
        high = findings.get("high", 0)

        # ClawVet verdict: malicious if critical > 0 or high > 0 or riskScore >= 50
        if critical > 0 or high > 0 or risk_score >= 50:
            verdict = "malicious"
        elif risk_score >= 20:
            verdict = "suspicious"
        else:
            verdict = "benign"

        return {
            "verdict": verdict,
            "risk_score": risk_score,
            "risk_grade": data.get("riskGrade", "?"),
            "findings": findings,
        }
    except subprocess.TimeoutExpired:
        return {"verdict": "error", "error": "timeout"}
    except (json.JSONDecodeError, Exception) as e:
        return {"verdict": "error", "error": str(e)}


def run_baseline(
    skills_dir: Path,
    labeled_dataset_path: Path,
    output_path: Path,
):
    """Run ClawVet on all skills in the labeled dataset."""
    dataset = json.loads(labeled_dataset_path.read_text())
    samples = dataset["samples"]

    print(f"Running ClawVet on {len(samples)} skills...")
    results = []

    for i, sample in enumerate(samples):
        skill_path = skills_dir / sample["path"]
        if not skill_path.exists():
            results.append({"path": sample["path"], "clawvet": {"verdict": "error", "error": "not found"}})
            continue

        print(f"  [{i+1}/{len(samples)}] {sample['path']}...", end=" ", flush=True)
        start = time.monotonic()
        clawvet_result = scan_with_clawvet(skill_path)
        elapsed = time.monotonic() - start

        clawvet_result["latency_s"] = round(elapsed, 1)
        results.append({
            "path": sample["path"],
            "ground_truth": sample["label"],
            "clawvet": clawvet_result,
        })
        print(f"→ {clawvet_result['verdict']} (score={clawvet_result.get('risk_score', '?')}, {elapsed:.1f}s)")

    # Compute metrics
    tp = fp = tn = fn = 0
    for r in results:
        truth = r["ground_truth"]
        pred = r["clawvet"]["verdict"]
        if truth == "error" or pred == "error":
            continue
        pred_mal = pred in ("malicious", "suspicious")
        truth_mal = truth == "malicious"
        if pred_mal and truth_mal:
            tp += 1
        elif pred_mal and not truth_mal:
            fp += 1
        elif not pred_mal and not truth_mal:
            tn += 1
        elif not pred_mal and truth_mal:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / (tp + fp + tn + fn), 4) if (tp + fp + tn + fn) > 0 else 0,
    }

    output = {
        "baseline": "clawvet",
        "version": "0.6.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": metrics,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nMetrics: {metrics}")
    print(f"Saved to {output_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills-dir", type=Path, default=Path("data/raw/openclaw-skills/skills"))
    parser.add_argument("--labeled-dataset", type=Path, default=Path("results/labeled_dataset_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline_clawvet.json"))
    args = parser.parse_args()
    run_baseline(args.skills_dir, args.labeled_dataset, args.output)
