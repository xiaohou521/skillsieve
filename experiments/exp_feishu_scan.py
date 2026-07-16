"""Experiment: Scan Feishu/Lark ecosystem packages with SkillSieve Layer 1.

This script discovers all feishu ecosystem packages, runs Layer 1 static triage,
and reports results for the Cross-Ecosystem Generalization section.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from skillsieve.adapters.feishu_adapter import discover_feishu_skills
from skillsieve.core.layer1.classifier import Layer1Classifier
from skillsieve.models import Verdict

ECOSYSTEM_DIR = Path("data/feishu_ecosystem")
RESULTS_FILE = Path("results/exp_feishu_l1.json")


def main():
    print("=== Feishu Ecosystem Layer 1 Scan ===\n")

    # Discover packages
    packages = discover_feishu_skills(ECOSYSTEM_DIR)
    print(f"Discovered {len(packages)} packages\n")

    # Initialize Layer 1 classifier (heuristic mode)
    classifier = Layer1Classifier(threshold=0.3)

    results = []
    safe_count = 0
    suspicious_count = 0
    total_time = 0.0

    for pkg in packages:
        t0 = time.time()
        result = classifier.classify(pkg)
        elapsed = (time.time() - t0) * 1000  # ms
        total_time += elapsed

        status = "SAFE" if result.verdict == Verdict.SAFE else "SUSPICIOUS"
        if result.verdict == Verdict.SAFE:
            safe_count += 1
        else:
            suspicious_count += 1

        entry = {
            "name": pkg.name,
            "verdict": result.verdict.value,
            "score": round(result.confidence, 4),
            "evidence": result.evidence[:5],  # top 5
            "time_ms": round(elapsed, 1),
            "n_scripts": len(pkg.script_files),
            "has_skillmd": bool(pkg.skill_md_content.strip()),
        }
        results.append(entry)

        flag = " ***" if status == "SUSPICIOUS" else ""
        print(f"  [{status:10s}] score={result.confidence:.3f}  {elapsed:6.1f}ms  {pkg.name}{flag}")
        if result.evidence:
            for ev in result.evidence[:3]:
                print(f"             {ev[:100]}")

    print(f"\n{'='*60}")
    print(f"Total: {len(packages)} packages")
    print(f"  SAFE:       {safe_count} ({safe_count/len(packages)*100:.1f}%)")
    print(f"  SUSPICIOUS: {suspicious_count} ({suspicious_count/len(packages)*100:.1f}%)")
    print(f"  Avg time:   {total_time/len(packages):.1f} ms/pkg")
    print(f"  Total time: {total_time:.0f} ms")

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "feishu_ecosystem_l1_scan",
        "total_packages": len(packages),
        "safe": safe_count,
        "suspicious": suspicious_count,
        "safe_pct": round(safe_count / len(packages) * 100, 1),
        "suspicious_pct": round(suspicious_count / len(packages) * 100, 1),
        "avg_time_ms": round(total_time / len(packages), 1),
        "total_time_ms": round(total_time, 0),
        "results": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
