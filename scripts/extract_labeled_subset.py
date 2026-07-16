"""Extract two skill subsets from the full corpus:

  1. labeled_390/  — all 390 in-benchmark labeled skills (open-source artifact)
  2. /tmp/audit_workspace/ — 180 benigns whose label_method has not been
     human-reviewed (both_safe or cv_error_default_benign), for codex re-audit

Run from project root.
"""
import json
import shutil
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
SRC = PROJ / "dataset" / "openclaw_skills_archive_20260404" / "skills"
LABELED_OUT = PROJ / "dataset" / "labeled_390"
AUDIT_OUT = Path("/tmp/audit_workspace")
META = PROJ / "results" / "labeled_dataset_final.json"

UNAUDITED_METHODS = {"both_safe", "cv_error_default_benign"}


def copy_skill(rel_path: str, dst_root: Path) -> bool:
    src = SRC / rel_path
    if not src.exists():
        return False
    dst = dst_root / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def main():
    with open(META) as f:
        meta = json.load(f)

    labeled = [s for s in meta["samples"] if s.get("evaluated_in_390_benchmark")]
    audit = [
        s
        for s in labeled
        if s["label"] == "benign" and s.get("label_method") in UNAUDITED_METHODS
    ]

    print(f"Labeled (390-benchmark): {len(labeled)}")
    print(f"Audit subset (unaudited benigns): {len(audit)}")

    LABELED_OUT.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)

    n_l_ok = n_l_miss = 0
    for s in labeled:
        if copy_skill(s["path"], LABELED_OUT):
            n_l_ok += 1
        else:
            n_l_miss += 1
            print(f"  [labeled] missing: {s['path']}")
    print(f"\nLabeled copy: {n_l_ok} ok, {n_l_miss} missing")

    n_a_ok = n_a_miss = 0
    for s in audit:
        if copy_skill(s["path"], AUDIT_OUT):
            n_a_ok += 1
        else:
            n_a_miss += 1
            print(f"  [audit] missing: {s['path']}")
    print(f"\nAudit copy: {n_a_ok} ok, {n_a_miss} missing")

    # Save audit manifest
    manifest = AUDIT_OUT / "_manifest.json"
    with open(manifest, "w") as f:
        json.dump(
            [
                {"path": s["path"], "original_label_method": s["label_method"]}
                for s in audit
            ],
            f,
            indent=2,
        )
    print(f"\nAudit manifest: {manifest}")


if __name__ == "__main__":
    main()
