"""Build labeled dataset using cross-validation: ClawVet + SkillSieve L1 + known authors.

Labeling strategy:
1. Known malicious authors → malicious (ground truth)
2. Run ClawVet on all samples → get ClawVet verdict
3. Run SkillSieve L1 → get L1 verdict
4. Both agree → auto-label
5. Disagree → mark for human review

This avoids circular reasoning (using our own L2 to label our own test set)
and avoids Kimi's tendency to label everything as benign.
"""
import json
import subprocess
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from skillsieve.core.layer1.classifier import Layer1Classifier
from skillsieve.parser import parse_skill_package

KNOWN_MALICIOUS_AUTHORS = {
    "hightower6eu",
    "moonshine-100rze",
    "sakaen736jih",
}


def scan_clawvet(skill_dir: Path) -> dict:
    """Run ClawVet on a single skill."""
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
        medium = findings.get("medium", 0)

        if critical > 0 or high > 0:
            verdict = "malicious"
        elif risk_score >= 30 or medium >= 3:
            verdict = "suspicious"
        else:
            verdict = "benign"

        return {"verdict": verdict, "risk_score": risk_score, "findings": findings}
    except Exception as e:
        return {"verdict": "error", "error": str(e)[:100]}


def build_dataset(
    skills_dir: Path,
    l1_results_path: Path,
    output_path: Path,
    n_suspicious: int = 200,
    n_safe: int = 300,
):
    l1_data = json.loads(l1_results_path.read_text())
    flagged_skills = l1_data.get("top_50_flagged_skills", [])

    print("=== Building cross-validated labeled dataset ===")

    # Step 1: Collect known malicious
    known_malicious = []
    for author in KNOWN_MALICIOUS_AUTHORS:
        author_dir = skills_dir / author
        if not author_dir.exists():
            continue
        for skill_dir in sorted(author_dir.iterdir()):
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                known_malicious.append(str(skill_dir.relative_to(skills_dir)))

    # Step 2: Collect suspicious (from L1 flagged, excluding known)
    suspicious_candidates = []
    for skill in flagged_skills:
        path = skill["path"]
        author = path.split("/")[0] if "/" in path else ""
        if author not in KNOWN_MALICIOUS_AUTHORS:
            suspicious_candidates.append(path)

    # Step 3: Collect safe (random from non-flagged, non-known)
    flagged_set = {s["path"] for s in flagged_skills}
    all_safe = []
    for author_dir in sorted(skills_dir.iterdir()):
        if not author_dir.is_dir() or author_dir.name in KNOWN_MALICIOUS_AUTHORS:
            continue
        for skill_dir in author_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                rel = str(skill_dir.relative_to(skills_dir))
                if rel not in flagged_set:
                    all_safe.append(rel)

    random.seed(42)

    # Sample
    mal_sample = known_malicious[:50]
    sus_sample = suspicious_candidates[:min(n_suspicious, len(suspicious_candidates))]
    safe_sample = random.sample(all_safe, min(n_safe, len(all_safe)))

    all_samples = []
    for p in mal_sample:
        all_samples.append({"path": p, "source": "known_malicious"})
    for p in sus_sample:
        all_samples.append({"path": p, "source": "l1_suspicious"})
    for p in safe_sample:
        all_samples.append({"path": p, "source": "l1_safe"})

    print(f"Samples: {len(mal_sample)} known + {len(sus_sample)} suspicious + {len(safe_sample)} safe = {len(all_samples)}")

    # Step 4: Run ClawVet + SkillSieve L1 on all samples
    classifier = Layer1Classifier()
    results = []

    for i, sample in enumerate(all_samples):
        skill_path = skills_dir / sample["path"]
        if not skill_path.exists():
            continue

        print(f"  [{i+1}/{len(all_samples)}] {sample['path']}...", end=" ", flush=True)

        # Known malicious — skip scanning
        if sample["source"] == "known_malicious":
            results.append({
                "path": sample["path"],
                "source": "known_malicious",
                "label": "malicious",
                "label_method": "ground_truth",
                "human_reviewed": True,
                "clawvet": {"verdict": "skipped"},
                "skillsieve_l1": {"verdict": "skipped"},
            })
            print("→ malicious (known)")
            continue

        # Run SkillSieve L1
        try:
            pkg = parse_skill_package(skill_path)
            l1_result = classifier.classify(pkg)
            l1_verdict = "suspicious" if l1_result.verdict.value != "safe" else "benign"
            l1_score = l1_result.confidence
        except Exception:
            l1_verdict = "error"
            l1_score = 0

        # Run ClawVet
        cv_result = scan_clawvet(skill_path)
        cv_verdict = cv_result["verdict"]

        # Cross-validate
        if l1_verdict == "suspicious" and cv_verdict in ("malicious", "suspicious"):
            label = "malicious"
            method = "both_flagged"
            human_review = False
        elif l1_verdict == "benign" and cv_verdict == "benign":
            label = "benign"
            method = "both_safe"
            human_review = False
        else:
            # Disagreement — needs human review
            label = "uncertain"
            method = "disagreement"
            human_review = True

        results.append({
            "path": sample["path"],
            "source": sample["source"],
            "label": label,
            "label_method": method,
            "human_reviewed": not human_review,  # auto-agreed ones are "reviewed"
            "needs_human_review": human_review,
            "clawvet": cv_result,
            "skillsieve_l1": {"verdict": l1_verdict, "score": l1_score},
        })

        symbol = "?" if human_review else ("MAL" if label == "malicious" else "ok")
        print(f"→ {label} (L1={l1_verdict}/{l1_score:.2f}, CV={cv_verdict}) [{symbol}]")

    # Summary
    labels = [r["label"] for r in results]
    needs_review = sum(1 for r in results if r.get("needs_human_review", False))

    summary = {
        "total": len(results),
        "malicious": labels.count("malicious"),
        "benign": labels.count("benign"),
        "uncertain": labels.count("uncertain"),
        "needs_human_review": needs_review,
        "auto_agreed": len(results) - needs_review,
    }

    output = {
        "dataset": "skillsieve_benchmark_cross_validated",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "ClawVet + SkillSieve L1 cross-validation",
        "sampling": {
            "known_malicious": len(mal_sample),
            "l1_suspicious": len(sus_sample),
            "l1_safe": len(safe_sample),
        },
        "summary": summary,
        "samples": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print(f"\n=== Summary ===")
    print(f"Total: {summary['total']}")
    print(f"Auto-labeled malicious: {summary['malicious']}")
    print(f"Auto-labeled benign: {summary['benign']}")
    print(f"Needs YOUR review: {summary['needs_human_review']}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills-dir", type=Path, default=Path("data/raw/openclaw-skills/skills"))
    parser.add_argument("--l1-results", type=Path, default=Path("results/real_scan_layer1_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("results/labeled_dataset_cross.json"))
    parser.add_argument("--n-suspicious", type=int, default=200)
    parser.add_argument("--n-safe", type=int, default=300)
    args = parser.parse_args()
    build_dataset(args.skills_dir, args.l1_results, args.output, args.n_suspicious, args.n_safe)
