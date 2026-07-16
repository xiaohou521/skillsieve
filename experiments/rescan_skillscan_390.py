"""Clean re-run of the SkillScan baseline on the exact 390-skill benchmark.

Why this replaces baseline_skillscan.json (AAAI-27 revision):
  - The old run was on 391 samples (55 malicious / 337 benign), not the final 390
    (56 / 334). The paper's Table 1 SkillScan row (P=0.154 / F1=0.267) is ~0.003
    off the stored file (P=0.1511 / F1=0.2625) — computed on a slightly different
    N. Same "wrong label set" class of issue as ClawVet.
  - We reproduce with the *claimed* tool version: skillscan-security 0.7.0
    (pinned via pip, verified `skillscan version`). Rules/intel are network-fetched
    by the tool; we run with --no-auto-intel against a locally-populated cache and
    RECORD the exact rules/intel component versions so the row is fully specified
    and reproducible.

Verdict mapping is byte-identical to the original harness
(run_skillscan_baseline.py lines 53-61): block|warn -> malicious, allow -> safe,
error -> safe (counted). Only correctness / N changes, not the decision rule.

Prereq: run `skillscan update` once so the rule cache is complete, then scan
offline. Component versions are captured from `skillscan version`.

Usage:
  python experiments/rescan_skillscan_390.py \
      --skills-dir dataset/labeled_390 \
      --labels results/labeled_dataset_final.json \
      --out results/baseline_skillscan_390_clean.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path


def component_versions() -> dict:
    """Capture `skillscan version` output so the run is fully specified."""
    try:
        out = subprocess.run(["skillscan", "version"], capture_output=True, text=True, timeout=30).stdout
        return {"skillscan_version_raw": out.strip()}
    except Exception as e:  # noqa: BLE001
        return {"skillscan_version_raw": f"(unavailable: {e})"}


def map_prediction(data: dict) -> dict:
    """Original harness mapping (run_skillscan_baseline.py), unchanged."""
    verdict = data.get("verdict", "allow")
    score = data.get("score", 0)
    findings = data.get("findings", []) or []
    if verdict in ("block", "warn"):
        pred = "malicious"
    else:
        pred = "safe"
    return {
        "verdict": verdict,
        "prediction": pred,
        "score": score,
        "n_findings": len(findings),
        "finding_categories": sorted({f.get("category", "") for f in findings}),
    }


async def scan_one(skill_dir: Path, timeout: float, extra_args: list[str]) -> dict:
    proc = await asyncio.create_subprocess_exec(
        "skillscan", "scan", str(skill_dir), "--format", "json", *extra_args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return {"verdict": "error", "prediction": "safe", "error": "timeout"}
    stdout = out.decode("utf-8", "replace")
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        return {"verdict": "error", "prediction": "safe", "error": f"json_parse: {e}",
                "raw_stdout": stdout[:1000], "raw_stderr": err.decode("utf-8", "replace")[:500]}
    return map_prediction(data)


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
    tp = fp = tn = fn = err_mal = err_ben = 0
    for r in records:
        truth_mal = r["truth"] == "malicious"
        if r.get("verdict") == "error":
            if error_policy == "exclude":
                if truth_mal:
                    err_mal += 1
                else:
                    err_ben += 1
                continue
            # policy 'safe': original harness treated errors as pred=safe
            if truth_mal:
                fn += 1
            else:
                tn += 1
            continue
        pred_mal = r["prediction"] == "malicious"
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
            "F2": round(F2, 3), "FPR": round(FPR, 3), "cost_100": 100 * fn + fp}


async def main(args):
    skills_dir = Path(args.skills_dir)
    labels = json.loads(Path(args.labels).read_text())
    samples = [s for s in labels["samples"] if s.get("evaluated_in_390_benchmark")]
    n_mal = sum(1 for s in samples if s["label"] == "malicious")
    comp = component_versions()
    print(f"390 benchmark: {len(samples)} samples ({n_mal} malicious, {len(samples)-n_mal} benign)")
    print(comp["skillscan_version_raw"], flush=True)

    extra = [] if args.auto_intel else ["--no-auto-intel"]
    out_path = Path(args.out)
    ckpt = out_path.with_suffix(".checkpoint.jsonl")
    done = load_done(ckpt)
    todo = [s for s in samples if s["path"] not in done]
    print(f"Resuming: {len(done)} done, {len(todo)} to scan. args={extra}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    ck_lock = asyncio.Lock()
    counter = {"n": 0}
    t0 = time.monotonic()

    async def do(sample):
        async with sem:
            path = sample["path"]
            skill_path = skills_dir / path
            t1 = time.monotonic()
            if not skill_path.exists():
                rec = {"path": path, "truth": sample["label"], "verdict": "error",
                       "prediction": "safe", "error": "dir_missing"}
            else:
                res = await scan_one(skill_path, args.timeout, extra)
                rec = {"path": path, "truth": sample["label"], **res,
                       "latency_s": round(time.monotonic() - t1, 1)}
            async with ck_lock:
                with ckpt.open("a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counter["n"] += 1
            if counter["n"] % 20 == 0 or rec.get("verdict") == "error":
                rate = counter["n"] / max(time.monotonic() - t0, 1e-9)
                eta = (len(todo) - counter["n"]) / max(rate, 1e-9) / 60
                tag = rec.get("prediction") if rec.get("verdict") != "error" else f"ERR {rec.get('error','')[:40]}"
                print(f"[{counter['n']}/{len(todo)}] {path[:42]} -> {tag} | {rate*60:.0f}/min ETA {eta:.0f}m", flush=True)

    await asyncio.gather(*[do(s) for s in todo])
    print(f"\nScanned. {time.monotonic()-t0:.0f}s.", flush=True)

    records = list(load_done(ckpt).values())
    n_err = sum(1 for r in records if r.get("verdict") == "error")
    m_excl = metrics(records, "exclude")
    m_safe = metrics(records, "safe")
    out = {
        "baseline": "skillscan-security",
        "claimed_version": "0.7.0",
        "components": comp,
        "auto_intel": args.auto_intel,
        "label_set": "390_benchmark_final",
        "n_samples": len(records),
        "n_errors": n_err,
        "coverage": round((len(records) - n_err) / len(records), 4) if records else 0,
        "metrics_error_excluded": m_excl,
        "metrics_error_as_safe": m_safe,
        "records": records,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n=== SkillScan on 390 (v0.7.0) ===")
    print(f"coverage: {out['coverage']*100:.1f}%  ({n_err} errors of {len(records)})")
    print(f"[errors as safe (orig policy)] P={m_safe['P']} R={m_safe['R']} F1={m_safe['F1']} "
          f"F2={m_safe['F2']} FPR={m_safe['FPR']}  (tp={m_safe['tp']} fp={m_safe['fp']} tn={m_safe['tn']} fn={m_safe['fn']})")
    print(f"[errors excluded            ] P={m_excl['P']} R={m_excl['R']} F1={m_excl['F1']} "
          f"F2={m_excl['F2']} FPR={m_excl['FPR']}")
    print(f"\nPaper Table 1 currently claims: P=0.154 R=1.000 F1=0.267 F2=0.476 FPR=0.917")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills-dir", default="dataset/labeled_390")
    ap.add_argument("--labels", default="results/labeled_dataset_final.json")
    ap.add_argument("--out", default="results/baseline_skillscan_390_clean.json")
    ap.add_argument("--auto-intel", action="store_true", help="allow network intel refresh (default: offline --no-auto-intel)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=90.0)
    asyncio.run(main(ap.parse_args()))
