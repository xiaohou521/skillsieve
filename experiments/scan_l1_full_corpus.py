"""Scan the full 49,592-skill corpus with Layer 1 and emit the flagged list.

This reproduces the paper's full-corpus Layer 1 scan and records the per-skill
paths of the approximately 6,871 suspicious packages. Pure CPU, no LLM/GPU.

The scan also refreshes the paper's corpus statistics (flagged count, parse
errors, and timing).

Output: results/l1_full_corpus_scan.json with per-skill {path, l1_score, evidence}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from skillsieve.core.layer1.classifier import Layer1Classifier
from skillsieve.parser import parse_skill_package
from skillsieve.models import Verdict


def main(corpora_dir: Path, out_path: Path):
    clf = Layer1Classifier()
    flagged, errors, total = [], 0, 0
    t0 = time.monotonic()
    # Use `find -name SKILL.md` semantics (do NOT follow symlinks) to match the
    # paper's 49,592 corpus definition. Python rglob follows symlinks and
    # over-counts by ~770. Each SKILL.md's parent dir is one skill package.
    import subprocess
    all_md = subprocess.run(
        ["find", str(corpora), "-name", "SKILL.md"], capture_output=True, text=True, check=True
    ).stdout.strip().split("\n")
    for md_path in all_md:
        if not md_path:
            continue
        sd = Path(md_path).parent
        total += 1
        try:
            pkg = parse_skill_package(sd)
            r = clf.classify(pkg)
        except Exception as e:  # noqa: BLE001 — record, don't crash
            errors += 1
            continue
        if r.verdict != Verdict.SAFE:
            rel = sd.relative_to(corpora)
            flagged.append({
                "path": str(rel),
                "l1_score": round(r.confidence, 4),
                "symlink_escapes": list(pkg.symlink_escapes),
            })
        if total % 5000 == 0:
            rate = total / max(time.monotonic() - t0, 1e-9)
            print(f"[{total}/49592] flagged={len(flagged)} errors={errors} | {rate:.0f}/s", flush=True)

    dt = time.monotonic() - t0
    out = {
        "experiment": "l1_full_corpus_scan",
        "corpus": "dataset/openclaw_skills_archive_20260404/skills",
        "total_scanned": total,
        "n_flagged": len(flagged),
        "n_parse_errors": errors,
        "flag_rate": round(len(flagged) / total, 4),
        "wall_clock_s": round(dt, 1),
        "flagged_skills": flagged,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False))
    print(f"\n=== L1 full corpus scan ===")
    print(f"scanned={total} flagged={len(flagged)} ({len(flagged)/total*100:.2f}%) errors={errors} in {dt/60:.1f} min")
    print(f"论文 Table 4 现写: 6,871 flagged (13.86%), 1,623 unparseable (3.27%)")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    corpora = Path("dataset/openclaw_skills_archive_20260404/skills")
    out = Path("results/l1_full_corpus_scan.json")
    main(corpora, out)
