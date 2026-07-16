"""Exp B: End-to-end fast-path evaluation on the 390-skill benchmark.

Flow:
  L1 SUSPICIOUS → XGBoost(p) → p ≥ τ_high → skip L2+L3, output MALICIOUS
                              → else        → normal L2 → L3

Compare to current full pipeline (exp1_L3_reaudited.json):
  - final P/R/F1/F2/FPR
  - L2/L3 call reduction
  - disagreement cases
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


TAU_HIGH = 0.95  # more conservative than OOF τ=0.98 to account for 390 distribution shift


def main():
    # Load training set (exclude benchmark paths for fair holdout)
    train = json.load(open("results/features_expanded.json"))
    X_tr0 = np.array(train["X"])
    y_tr0 = np.array(train["y"])
    train_paths = train["paths"]
    names = train["names"]

    # Load benchmark
    test = json.load(open("results/features_400.json"))
    heur = json.load(open("results/exp1_L1_reaudited.json"))
    heur_paths = set(heur["predictions"].keys())
    keep = np.array([p in heur_paths for p in test["paths"]], dtype=bool)
    X_te = np.array(test["X"])[keep]
    y_te = np.array(test["y"])[keep]
    test_paths = [p for p, k in zip(test["paths"], keep) if k]
    test_names = test["names"]
    test_set = set(test_paths)

    # Align feature columns
    tfmaps = [dict(zip(test_names, r)) for r in X_te]
    X_te_a = np.array([[float(fm.get(k, 0.0)) for k in names] for fm in tfmaps])

    # IOC augment (excluding test leakage)
    from skillsieve.core.layer1.classifier import Layer1Classifier
    from skillsieve.parser import parse_skill_package
    classifier = Layer1Classifier()
    skills_dir = Path("/tmp/openclaw-skills-audit/skills")
    new_X = []
    for a in sorted(skills_dir.iterdir()):
        if not a.is_dir():
            continue
        for s in a.iterdir():
            if not (s / "SKILL.md").exists():
                continue
            rel = f"{a.name}/{s.name}"
            if rel in set(train_paths) or rel in test_set:
                continue
            try:
                pkg = parse_skill_package(s)
                vec, n = classifier.assemble_features(pkg)
                fm = dict(zip(n, vec))
                new_X.append([float(fm.get(k, 0.0)) for k in names])
            except Exception:
                pass
    new_X = np.array(new_X) if new_X else np.zeros((0, len(names)))
    new_y = np.ones(len(new_X), dtype=int)

    keep_mask = np.array([p not in test_set for p in train_paths], dtype=bool)
    X_tr = np.vstack([X_tr0[keep_mask], new_X])
    y_tr = np.concatenate([y_tr0[keep_mask], new_y])
    print(f"Training: {len(y_tr)} ({int(y_tr.sum())} mal) "
          f"Test: {len(y_te)} ({int(y_te.sum())} mal)")

    import xgboost as xgb
    # Train and predict
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42},
        xgb.DMatrix(X_tr, y_tr), num_boost_round=200,
    )
    p_xgb = booster.predict(xgb.DMatrix(X_te_a))

    # Load L3 (current full pipeline) verdicts — those are the "fall-through" verdicts
    l3 = json.load(open("results/exp1_L3_reaudited.json"))
    l3_preds = l3["predictions"]  # path -> 'malicious' / 'safe'

    # Ground-truth labels
    labels = json.load(open("results/labeled_dataset_final.json"))
    label_map = {s["path"]: s["label"] for s in labels["samples"]}

    # L1 heuristic verdicts (needed to decide who is "suspicious")
    l1 = heur["predictions"]

    # Build fast-path pipeline:
    # - L1 benign → final = benign (unchanged)
    # - L1 suspicious + XGBoost ≥ τ_high → final = malicious (fast-path)
    # - L1 suspicious + XGBoost < τ_high → final = L3 verdict (fall-through)
    fp_final = {}
    fastpath_accept = []   # XGBoost fast-accept samples
    fallthrough = []       # went through L2+L3
    for i, path in enumerate(test_paths):
        l1_verdict = l1[path]  # 'malicious' or 'safe'
        if l1_verdict != "malicious":
            fp_final[path] = "benign"
            continue
        # L1 suspicious
        if p_xgb[i] >= TAU_HIGH:
            fp_final[path] = "malicious"
            fastpath_accept.append((path, float(p_xgb[i])))
        else:
            l3_v = l3_preds.get(path, "malicious")  # default malicious if missing
            fp_final[path] = "malicious" if l3_v == "malicious" else "benign"
            fallthrough.append(path)

    # Evaluate
    def metrics(pred_map, label_map, paths):
        tp = fp = tn = fn = 0
        for p in paths:
            pr = (pred_map[p] == "malicious")
            tr = (label_map[p] == "malicious")
            if pr and tr: tp += 1
            elif pr and not tr: fp += 1
            elif not pr and not tr: tn += 1
            else: fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec,
                    f1=f1, f2=f2, fpr=fpr)

    # Baseline: full pipeline
    base = metrics(l3_preds, label_map, test_paths)
    # Fast-path
    fast = metrics(fp_final, label_map, test_paths)

    # Count
    n_susp = sum(1 for p in test_paths if l1[p] == "malicious")
    n_fast = len(fastpath_accept)
    n_fall = len(fallthrough)
    reduction = n_fast / n_susp if n_susp else 0

    print(f"\nL1 SUSPICIOUS: {n_susp}")
    print(f"Fast-accept (XGBoost ≥ {TAU_HIGH}): {n_fast} ({100*reduction:.1f}% of SUSPICIOUS)")
    print(f"Fall-through to L2+L3: {n_fall}")

    print(f"\n=== Full pipeline (baseline) ===")
    print(f"  TP={base['tp']} FP={base['fp']} TN={base['tn']} FN={base['fn']}")
    print(f"  P={base['precision']:.4f} R={base['recall']:.4f} F1={base['f1']:.4f} "
          f"F2={base['f2']:.4f} FPR={base['fpr']:.4f}")

    print(f"\n=== Fast-path pipeline (τ_high={TAU_HIGH}) ===")
    print(f"  TP={fast['tp']} FP={fast['fp']} TN={fast['tn']} FN={fast['fn']}")
    print(f"  P={fast['precision']:.4f} R={fast['recall']:.4f} F1={fast['f1']:.4f} "
          f"F2={fast['f2']:.4f} FPR={fast['fpr']:.4f}")

    print(f"\n=== Delta ===")
    print(f"  Δ Recall: {fast['recall'] - base['recall']:+.4f} "
          f"{'⚠ FN introduced' if fast['fn'] > base['fn'] else 'OK (no new FN)'}")
    print(f"  Δ Precision: {fast['precision'] - base['precision']:+.4f}")
    print(f"  Δ F1: {fast['f1'] - base['f1']:+.4f}")

    # Disagreement cases
    disagreements = []
    for p in test_paths:
        if fp_final[p] != l3_preds.get(p, fp_final[p]):
            disagreements.append({
                "path": p,
                "true": label_map[p],
                "full_pipeline": l3_preds.get(p, "(missing)"),
                "fastpath": fp_final[p],
                "xgboost_p": float(p_xgb[test_paths.index(p)]),
            })

    print(f"\nDisagreements between full pipeline and fast-path: {len(disagreements)}")
    for d in disagreements[:10]:
        print(f"  {d['path']}: true={d['true']} full={d['full_pipeline']} "
              f"fast={d['fastpath']} p_xgb={d['xgboost_p']:.3f}")

    # Cases where fast-path introduced FN (if any)
    fp_fn_cases = [d for d in disagreements
                   if d["true"] == "malicious" and d["fastpath"] != "malicious"]
    if fp_fn_cases:
        print(f"\n⚠ Fast-path FN cases: {len(fp_fn_cases)}")
        for d in fp_fn_cases:
            print(f"    {d['path']}: p_xgb={d['xgboost_p']:.3f}")

    out = {
        "tau_high": TAU_HIGH,
        "test_size": len(test_paths),
        "n_suspicious": n_susp,
        "n_fastpath_accept": n_fast,
        "n_fallthrough": n_fall,
        "fastpath_reduction_pct": reduction,
        "baseline": base,
        "fastpath": fast,
        "disagreements": disagreements,
        "fastpath_accepted": fastpath_accept,
    }
    Path("results/xgboost_fastpath_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to results/xgboost_fastpath_eval.json")


if __name__ == "__main__":
    main()
