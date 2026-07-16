"""Clean re-run of the ClawVet baseline on the exact 390-skill benchmark.

Why this replaces baseline_clawvet_v2.json (AAAI-27 revision):
  - The old run had 68/399 samples (17%) fail with "Expecting value: line 1
    column 1" — a stdout-parse failure in the harness, NOT ClawVet failing to
    detect. Verified 2026-07-13: the same samples parse cleanly with clawvet
    0.6.0/0.6.3/0.7.5 (identical riskScore). The old errors were environmental
    (npx cold-start banner contaminating stdout on the original machine).
  - The old run was on a stale label set (88 malicious), never re-run on the
    final 390 (56 malicious). The paper's Table 1 ClawVet row (R=0.429) matches
    no stored artifact.

This script pins an explicit clawvet version, scans exactly the 390 evaluated
skills, stores each sample's raw ClawVet JSON, and reports metrics with fully
transparent error accounting (coverage + two policies). Verdict mapping is
byte-identical to the original harness so the only thing that changes is
correctness, not the decision rule.

Usage:
  python experiments/rescan_clawvet_390.py --clawvet-version 0.6.3 \
      --skills-dir dataset/labeled_390 \
      --labels results/labeled_dataset_final.json \
      --out results/baseline_clawvet_390_clean.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path


def map_verdict(data: dict) -> dict:
    """Original harness mapping (run_clawvet_baseline.py lines 17-28), unchanged."""
    risk_score = data.get("riskScore", 0)
    findings = data.get("findingsCount", {}) or {}
    critical = findings.get("critical", 0)
    high = findings.get("high", 0)
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


async def scan_one(version: str, skill_dir: Path, timeout: float) -> dict:
    """Run `npx clawvet@<version> scan <dir> --format json`; capture raw + parsed."""
    proc = await asyncio.create_subprocess_exec(
        "npx", "--yes", f"clawvet@{version}", "scan", str(skill_dir), "--format", "json",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return {"verdict": "error", "error": "timeout", "raw_stdout": ""}
    stdout = out.decode("utf-8", "replace")
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        return {"verdict": "error", "error": f"json_parse: {e}",
                "raw_stdout": stdout[:2000], "raw_stderr": err.decode("utf-8", "replace")[:500]}
    res = map_verdict(data)
    res["raw_json"] = data
    return res


def load_done(ckpt: Path) -> dict:
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["path"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def metrics(records: list[dict], error_policy: str) -> dict:
    """error_policy: 'exclude' (drop errors from denom) or 'fn' (count errors as missed)."""
    tp = fp = tn = fn = err_mal = err_ben = 0
    for r in records:
        truth_mal = r["truth"] == "malicious"
        if r["verdict"] == "error":
            if error_policy == "exclude":
                (err_mal if truth_mal else err_ben)  # noqa: B018 — counted below
                if truth_mal:
                    err_mal += 1
                else:
                    err_ben += 1
                continue
            # policy 'fn': error on a malicious = missed (fn); error on benign = correct-negative (tn)
            if truth_mal:
                fn += 1
            else:
                tn += 1
            continue
        pred_mal = r["verdict"] in ("malicious", "suspicious")
        if pred_mal and truth_mal:
            tp += 1
        elif pred_mal and not truth_mal:
            fp += 1
        elif not pred_mal and not truth_mal:
            tn += 1
        else:
            fn += 1
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    F2 = 5 * P * R / (4 * P + R) if 4 * P + R else 0.0
    FPR = fp / (fp + tn) if fp + tn else 0.0
    return {"policy": error_policy, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "errors_excluded_mal": err_mal, "errors_excluded_ben": err_ben,
            "P": round(P, 3), "R": round(R, 3), "F1": round(F1, 3),
            "F2": round(F2, 3), "FPR": round(FPR, 3)}


async def main(args):
    skills_dir = Path(args.skills_dir)
    labels = json.loads(Path(args.labels).read_text())
    samples = [s for s in labels["samples"] if s.get("evaluated_in_390_benchmark")]
    n_mal = sum(1 for s in samples if s["label"] == "malicious")
    print(f"390 benchmark: {len(samples)} samples ({n_mal} malicious, {len(samples)-n_mal} benign)")
    print(f"ClawVet version pinned: {args.clawvet_version}", flush=True)

    out_path = Path(args.out)
    ckpt = out_path.with_suffix(".checkpoint.jsonl")
    done = load_done(ckpt)
    todo = [s for s in samples if s["path"] not in done]
    print(f"Resuming: {len(done)} done, {len(todo)} to scan.", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    ck_lock = asyncio.Lock()
    counter = {"n": 0}
    t0 = time.monotonic()

    async def do(sample):
        async with sem:
            path = sample["path"]
            skill_path = skills_dir / path
            t1 = time.monotonic()
            if not (skill_path).exists():
                rec = {"path": path, "truth": sample["label"], "verdict": "error", "error": "dir_missing"}
            else:
                res = await scan_one(args.clawvet_version, skill_path, args.timeout)
                rec = {"path": path, "truth": sample["label"], **res, "latency_s": round(time.monotonic() - t1, 1)}
            async with ck_lock:
                with ckpt.open("a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counter["n"] += 1
            if counter["n"] % 20 == 0 or rec["verdict"] == "error":
                rate = counter["n"] / max(time.monotonic() - t0, 1e-9)
                eta = (len(todo) - counter["n"]) / max(rate, 1e-9) / 60
                tag = rec["verdict"] if rec["verdict"] != "error" else f"ERR {rec.get('error','')[:40]}"
                print(f"[{counter['n']}/{len(todo)}] {path[:45]} -> {tag} | {rate*60:.0f}/min ETA {eta:.0f}m", flush=True)

    await asyncio.gather(*[do(s) for s in todo])
    print(f"\nScanned. {time.monotonic()-t0:.0f}s total.", flush=True)

    # Consolidate and report.
    records = list(load_done(ckpt).values())
    n_err = sum(1 for r in records if r["verdict"] == "error")
    m_exclude = metrics(records, "exclude")
    m_fn = metrics(records, "fn")
    out = {
        "baseline": "clawvet",
        "clawvet_version": args.clawvet_version,
        "label_set": "390_benchmark_final",
        "n_samples": len(records),
        "n_errors": n_err,
        "coverage": round((len(records) - n_err) / len(records), 4) if records else 0,
        "metrics_error_excluded": m_exclude,
        "metrics_error_as_fn": m_fn,
        "records": records,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n=== ClawVet on 390 (v{args.clawvet_version}) ===")
    print(f"coverage: {out['coverage']*100:.1f}%  ({n_err} errors of {len(records)})")
    print(f"[errors excluded] P={m_exclude['P']} R={m_exclude['R']} F1={m_exclude['F1']} "
          f"F2={m_exclude['F2']} FPR={m_exclude['FPR']}  (tp={m_exclude['tp']} fp={m_exclude['fp']} "
          f"tn={m_exclude['tn']} fn={m_exclude['fn']})")
    print(f"[errors as FN ] P={m_fn['P']} R={m_fn['R']} F1={m_fn['F1']} "
          f"F2={m_fn['F2']} FPR={m_fn['FPR']}  (tp={m_fn['tp']} fp={m_fn['fp']} "
          f"tn={m_fn['tn']} fn={m_fn['fn']})")
    print(f"\nPaper Table 1 currently claims: P=0.157 R=0.429 F1=0.230 F2=0.318 FPR=0.375")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clawvet-version", default="0.6.3")
    ap.add_argument("--skills-dir", default="dataset/labeled_390")
    ap.add_argument("--labels", default="results/labeled_dataset_final.json")
    ap.add_argument("--out", default="results/baseline_clawvet_390_clean.json")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60.0)
    asyncio.run(main(ap.parse_args()))
