"""Run skillscan-security baseline on the labeled dataset."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def scan_with_skillscan(skill_dir: Path) -> dict:
    try:
        result = subprocess.run(
            ["skillscan", "scan", str(skill_dir), "--format", "json"],
            capture_output=True, text=True, timeout=60,
        )
        data = json.loads(result.stdout)
        verdict = data.get("verdict", "allow")
        score = data.get("score", 0)
        findings = data.get("findings", [])
        return {
            "verdict": verdict,
            "score": score,
            "n_findings": len(findings),
            "finding_categories": list({f.get("category", "") for f in findings}),
        }
    except subprocess.TimeoutExpired:
        return {"verdict": "error", "error": "timeout"}
    except (json.JSONDecodeError, Exception) as e:
        return {"verdict": "error", "error": str(e)[:200]}


def run_baseline(skills_dir: Path, labeled_path: Path, output_path: Path):
    dataset = json.loads(labeled_path.read_text())
    labels = {s["path"]: s["label"] for s in dataset["samples"]}

    print(f"Running skillscan-security on {len(labels)} skills...")
    results = []
    tp = fp = tn = fn = errors = 0

    for i, (path, truth) in enumerate(sorted(labels.items())):
        skill_path = skills_dir / path
        if not skill_path.exists():
            continue

        print(f"  [{i+1}/{len(labels)}] {path}...", end=" ", flush=True)
        start = time.monotonic()
        ss_result = scan_with_skillscan(skill_path)
        elapsed = time.monotonic() - start

        # Map verdict: block → malicious, warn → malicious (strict), allow → safe
        v = ss_result["verdict"]
        if v == "error":
            pred = "safe"
            errors += 1
        elif v in ("block", "warn"):
            pred = "malicious"
        else:
            pred = "safe"

        if pred == "malicious" and truth == "malicious": tp += 1
        elif pred == "malicious" and truth == "benign": fp += 1
        elif pred == "safe" and truth == "benign": tn += 1
        elif pred == "safe" and truth == "malicious": fn += 1

        results.append({
            "path": path, "ground_truth": truth, "prediction": pred,
            "skillscan": ss_result, "latency_s": round(elapsed, 2),
        })
        print(f"→ {pred} (score={ss_result.get('score', '?')}, {elapsed:.1f}s)")

    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    f2 = 5*p*r/(4*p+r) if (4*p+r) else 0
    acc = (tp+tn)/(tp+fp+tn+fn) if (tp+fp+tn+fn) else 0
    fpr = fp/(fp+tn) if (fp+tn) else 0
    cost = 100*fn + fp

    output = {
        "baseline": "skillscan-security",
        "version": "0.7.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "f2": round(f2, 4),
            "accuracy": round(acc, 4), "fpr": round(fpr, 4),
            "cost_100": cost, "errors": errors,
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nMetrics: P={p:.3f} R={r:.3f} F1={f1:.3f} F2={f2:.3f} Cost={cost}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn} Errors={errors}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills-dir", type=Path, default=Path("/tmp/openclaw-skills-audit/skills"))
    parser.add_argument("--labeled-dataset", type=Path, default=Path("results/labeled_dataset_final.json"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline_skillscan.json"))
    args = parser.parse_args()
    run_baseline(args.skills_dir, args.labeled_dataset, args.output)
