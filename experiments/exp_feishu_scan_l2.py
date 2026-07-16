"""Experiment: Run Layer 2 SSD on Feishu packages flagged suspicious by Layer 1.

Reads L1 results from exp_feishu_l1.json, runs L2 on suspicious packages,
and saves combined results to exp_feishu_l2.json.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from skillsieve.adapters.feishu_adapter import discover_feishu_skills
from skillsieve.core.layer1.classifier import Layer1Classifier
from skillsieve.core.layer2.semantic_engine import SemanticEngine
from skillsieve.core.layer2.aggregator import aggregate_subtasks
from skillsieve.models import Verdict

ECOSYSTEM_DIR = Path("data/feishu_ecosystem")
RESULTS_FILE = Path("results/exp_feishu_l2.json")


async def main():
    print("=== Feishu Ecosystem Layer 1 + Layer 2 Scan ===\n")

    packages = discover_feishu_skills(ECOSYSTEM_DIR)
    print(f"Discovered {len(packages)} packages\n")

    classifier = Layer1Classifier(threshold=0.3)
    l2_engine = SemanticEngine(profile="kimi")

    results = []
    l1_safe = 0
    l1_suspicious = 0
    l2_safe = 0
    l2_suspicious = 0

    for pkg in packages:
        # Layer 1
        t0 = time.time()
        l1 = classifier.classify(pkg)
        l1_ms = (time.time() - t0) * 1000

        entry = {
            "name": pkg.name,
            "l1_verdict": l1.verdict.value,
            "l1_score": round(l1.confidence, 4),
            "l1_time_ms": round(l1_ms, 1),
            "l1_evidence": l1.evidence[:5],
        }

        if l1.verdict == Verdict.SAFE:
            l1_safe += 1
            entry["final_verdict"] = "safe"
            print(f"  [L1 SAFE     ] score={l1.confidence:.3f}  {pkg.name}")
        else:
            l1_suspicious += 1
            # Layer 2
            t1 = time.time()
            try:
                l2 = await l2_engine.analyze(pkg, layer1_evidence=l1.evidence)
                l2_ms = (time.time() - t1) * 1000

                entry["l2_verdict"] = l2.verdict.value
                entry["l2_score"] = round(l2.confidence, 4)
                entry["l2_time_ms"] = round(l2_ms, 1)
                entry["l2_evidence"] = l2.evidence[:5]
                entry["l2_subtask_scores"] = l2.details.get("subtask_scores", {})

                if l2.verdict == Verdict.SAFE:
                    l2_safe += 1
                    entry["final_verdict"] = "safe"
                    status = "L2 SAFE"
                else:
                    l2_suspicious += 1
                    entry["final_verdict"] = "suspicious"
                    status = "L2 SUSPICIOUS"

                print(f"  [{status:14s}] L1={l1.confidence:.3f} → L2={l2.confidence:.3f}  {pkg.name}")
                if l2.details.get("subtask_scores"):
                    scores = l2.details["subtask_scores"]
                    print(f"                  subtasks: {', '.join(f'{k}={v:.2f}' for k,v in scores.items())}")

            except Exception as e:
                entry["l2_error"] = str(e)[:200]
                entry["final_verdict"] = "error"
                print(f"  [L2 ERROR    ] {pkg.name}: {str(e)[:80]}")

        results.append(entry)

    # Summary
    print(f"\n{'='*60}")
    print(f"Total: {len(packages)} packages")
    print(f"  L1 SAFE:       {l1_safe}")
    print(f"  L1 SUSPICIOUS: {l1_suspicious}")
    print(f"    → L2 SAFE:       {l2_safe} (L1 FP corrected)")
    print(f"    → L2 SUSPICIOUS: {l2_suspicious} (remain flagged)")

    summary = {
        "experiment": "feishu_ecosystem_l1_l2_scan",
        "total_packages": len(packages),
        "l1_safe": l1_safe,
        "l1_suspicious": l1_suspicious,
        "l2_safe": l2_safe,
        "l2_suspicious": l2_suspicious,
        "l2_correction_rate": round(l2_safe / l1_suspicious, 3) if l1_suspicious else 0,
        "results": results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
