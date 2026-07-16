"""Build a 500-sample labeled dataset for evaluation.

Sampling strategy:
- 250 from L1 suspicious (highest risk scores, most likely to contain real malicious)
- 250 from L1 safe (random sample, expected mostly benign)
- Known malicious authors are pre-labeled as malicious

Auto-labeling: Use Kimi 2.5 to classify each skill, then output for human review.
"""
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from skillsieve.adapters.llm_adapter import LLMAdapter
from skillsieve.parser import parse_skill_package

# Known malicious authors (confirmed by VirusTotal, Snyk, security researchers)
KNOWN_MALICIOUS_AUTHORS = {
    "hightower6eu",      # ClawHavoc campaign, 354 skills
    "moonshine-100rze",  # ToxicSkills reference, base64 reverse shells
    "sakaen736jih",      # 206 flagged skills
}

LABEL_PROMPT = """\
You are a security analyst. Analyze this AI agent skill package and classify it.

## SKILL.md
{skill_md}

## Script files
{scripts}

## Task
Classify this skill as BENIGN or MALICIOUS.

A skill is MALICIOUS if it:
- Tricks users into downloading/running executables from untrusted sources
- Steals credentials, API keys, or sensitive files
- Sends data to external servers without clear user benefit
- Contains obfuscated code (base64-encoded commands, hex strings)
- Uses social engineering to bypass security (fake prerequisites, urgency)
- Contains prompt injection to override agent safety

A skill is BENIGN if it:
- Provides legitimate functionality matching its description
- Only accesses resources needed for its stated purpose
- Does not attempt to hide its actions

Respond with JSON only:
{{"label": "benign" or "malicious", "confidence": 0.0-1.0, "reason": "one sentence"}}
"""


async def auto_label_skill(adapter: LLMAdapter, skill_dir: Path) -> dict:
    """Use LLM to auto-label a single skill."""
    pkg = parse_skill_package(skill_dir)

    # Truncate content
    skill_md = pkg.skill_md_content[:3000]
    scripts_text = ""
    total = 0
    for fname, content in pkg.script_files.items():
        if total > 4000:
            break
        text = content[:1500]
        scripts_text += f"--- {fname} ---\n{text}\n\n"
        total += len(text)
    if not scripts_text:
        scripts_text = "(no scripts)"

    prompt = LABEL_PROMPT.format(skill_md=skill_md, scripts=scripts_text)

    try:
        raw = await adapter.complete(prompt, system="You are a security analyst. Output JSON only.")
        parsed = LLMAdapter.parse_json_response(raw)
        return {
            "label": parsed.get("label", "unknown"),
            "confidence": parsed.get("confidence", 0.0),
            "reason": parsed.get("reason", ""),
        }
    except Exception as e:
        return {"label": "error", "confidence": 0.0, "reason": str(e)}


async def build_dataset(
    skills_dir: Path,
    l1_results_path: Path,
    output_path: Path,
    n_suspicious: int = 250,
    n_safe: int = 250,
):
    """Build labeled dataset with stratified sampling + auto-labeling."""
    l1_data = json.loads(l1_results_path.read_text())

    # Collect all skill paths from L1 results
    flagged_skills = l1_data.get("top_50_flagged_skills", [])

    # We need more than top 50 — scan the full results to find all suspicious
    # For now, use top flagged + known malicious + random safe
    print("=== Building 500-sample labeled dataset ===")

    # Step 1: Collect known malicious skills
    known_malicious = []
    for author in KNOWN_MALICIOUS_AUTHORS:
        author_dir = skills_dir / author
        if not author_dir.exists():
            continue
        for skill_dir in sorted(author_dir.iterdir()):
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                rel = str(skill_dir.relative_to(skills_dir))
                known_malicious.append(rel)

    print(f"Known malicious authors: {len(known_malicious)} skills")

    # Step 2: Collect suspicious skills (from L1 flagged, excluding known malicious)
    suspicious_candidates = []
    for skill in flagged_skills:
        path = skill["path"]
        author = path.split("/")[0] if "/" in path else ""
        if author not in KNOWN_MALICIOUS_AUTHORS:
            suspicious_candidates.append(path)

    # Step 3: Collect safe skills (random sample from all skills not in suspicious/known)
    all_skill_dirs = []
    for author_dir in sorted(skills_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        author = author_dir.name
        if author in KNOWN_MALICIOUS_AUTHORS:
            continue
        for skill_dir in author_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                rel = str(skill_dir.relative_to(skills_dir))
                if rel not in [s["path"] for s in flagged_skills]:
                    all_skill_dirs.append(rel)

    random.seed(42)

    # Sample
    # Take up to 50 known malicious
    malicious_sample = known_malicious[:50]
    # Take up to 200 from suspicious (not known malicious)
    suspicious_sample = suspicious_candidates[:min(n_suspicious - len(malicious_sample), len(suspicious_candidates))]
    # Fill remaining suspicious quota from known malicious
    n_suspicious_total = len(malicious_sample) + len(suspicious_sample)
    # Take safe samples
    safe_sample = random.sample(all_skill_dirs, min(n_safe, len(all_skill_dirs)))

    all_samples = []
    for path in malicious_sample:
        all_samples.append({"path": path, "source": "known_malicious", "pre_label": "malicious"})
    for path in suspicious_sample:
        all_samples.append({"path": path, "source": "l1_suspicious", "pre_label": "unknown"})
    for path in safe_sample:
        all_samples.append({"path": path, "source": "l1_safe", "pre_label": "unknown"})

    print(f"Sample composition: {len(malicious_sample)} known malicious + {len(suspicious_sample)} suspicious + {len(safe_sample)} safe = {len(all_samples)} total")

    # Step 4: Auto-label with Kimi
    adapter = LLMAdapter(profile="kimi")
    results = []

    for i, sample in enumerate(all_samples):
        skill_path = skills_dir / sample["path"]
        if not skill_path.exists():
            continue

        print(f"  [{i+1}/{len(all_samples)}] {sample['path']}...", end=" ", flush=True)

        if sample["pre_label"] == "malicious":
            # Known malicious — skip LLM, use ground truth
            results.append({
                "path": sample["path"],
                "source": sample["source"],
                "label": "malicious",
                "confidence": 1.0,
                "reason": "Known malicious author (confirmed by VirusTotal/Snyk)",
                "llm_label": "skipped",
                "human_reviewed": False,
            })
            print("→ malicious (known)")
            continue

        # Auto-label with LLM
        llm_result = await auto_label_skill(adapter, skill_path)

        results.append({
            "path": sample["path"],
            "source": sample["source"],
            "label": llm_result["label"],
            "confidence": llm_result["confidence"],
            "reason": llm_result["reason"],
            "llm_label": llm_result["label"],
            "human_reviewed": False,
        })
        print(f"→ {llm_result['label']} (conf={llm_result['confidence']:.2f})")

        # Rate limiting
        if (i + 1) % 10 == 0:
            await asyncio.sleep(1)

    # Summary
    labels = [r["label"] for r in results]
    summary = {
        "total": len(results),
        "malicious": labels.count("malicious"),
        "benign": labels.count("benign"),
        "unknown": labels.count("unknown"),
        "errors": labels.count("error"),
        "human_reviewed": sum(1 for r in results if r["human_reviewed"]),
    }

    output = {
        "dataset": "skillsieve_benchmark_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sampling": {
            "known_malicious": len(malicious_sample),
            "l1_suspicious": len(suspicious_sample),
            "l1_safe": len(safe_sample),
        },
        "summary": summary,
        "samples": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print(f"\n=== Dataset Summary ===")
    print(f"Total: {summary['total']}")
    print(f"Malicious: {summary['malicious']}")
    print(f"Benign: {summary['benign']}")
    print(f"Unknown/Error: {summary['unknown'] + summary['errors']}")
    print(f"Saved to {output_path}")
    print(f"\nNext: Human review the 'unknown' and low-confidence labels")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build 500-sample labeled dataset")
    parser.add_argument("--skills-dir", type=Path, default=Path("data/raw/openclaw-skills/skills"))
    parser.add_argument("--l1-results", type=Path, default=Path("results/real_scan_layer1_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("results/labeled_dataset_v1.json"))
    parser.add_argument("--n-suspicious", type=int, default=250)
    parser.add_argument("--n-safe", type=int, default=250)
    args = parser.parse_args()
    asyncio.run(build_dataset(args.skills_dir, args.l1_results, args.output, args.n_suspicious, args.n_safe))
