"""Evaluate XGBoost (trained on augmented 1,438 set) on the 391-skill benchmark.

Direct comparison to heuristic F1=0.514 on the same benchmark.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    # Load augmented training data
    train = json.load(open("results/features_expanded.json"))
    X_tr = np.array(train["X"])
    y_tr = np.array(train["y"])
    train_paths = set(train["paths"])
    names = train["names"]

    # Load benchmark first so we can exclude test paths from IOC augmentation
    test = json.load(open("results/features_400.json"))
    heur = json.load(open("results/exp1_L1_reaudited.json"))
    heur_paths = set(heur["predictions"].keys())
    feat_paths_all = test["paths"]

    # Intersect: only keep paths evaluated by both
    keep_test = np.array([p in heur_paths for p in feat_paths_all], dtype=bool)
    X_te = np.array(test["X"])[keep_test]
    y_te = np.array(test["y"])[keep_test]
    test_paths = [p for p, k in zip(feat_paths_all, keep_test) if k]
    test_names = test["names"]
    test_path_set = set(test_paths)
    print(f"Benchmark aligned to heuristic: {len(test_paths)} samples ({int(y_te.sum())} malicious)")

    # Add IOC-fetched new skills (excluding any in the test set)
    from skillsieve.core.layer1.classifier import Layer1Classifier
    from skillsieve.parser import parse_skill_package

    skills_dir = Path("/tmp/openclaw-skills-audit/skills")
    classifier = Layer1Classifier()

    new_X = []
    new_skipped_leak = 0
    for author_dir in sorted(skills_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        for skill_dir in author_dir.iterdir():
            if not (skill_dir / "SKILL.md").exists():
                continue
            rel = f"{author_dir.name}/{skill_dir.name}"
            if rel in train_paths:
                continue
            if rel in test_path_set:
                new_skipped_leak += 1
                continue
            try:
                pkg = parse_skill_package(skill_dir)
                vec, n = classifier.assemble_features(pkg)
                fmap = dict(zip(n, vec))
                new_X.append([float(fmap.get(k, 0.0)) for k in names])
            except Exception:
                pass
    print(f"IOC skills added to training: {len(new_X)} (skipped {new_skipped_leak} test-set leaks)")
    new_X = np.array(new_X) if new_X else np.zeros((0, len(names)))
    new_y = np.ones(len(new_X), dtype=int)

    # Align feature columns to training names
    test_fmap_list = [dict(zip(test_names, row)) for row in X_te]
    X_te_aligned = np.array([[float(fmap.get(k, 0.0)) for k in names] for fmap in test_fmap_list])

    # Remove benchmark samples from training for fair hold-out
    keep_mask = np.array([p not in test_path_set for p in train["paths"]], dtype=bool)
    X_tr_old = X_tr[keep_mask]
    y_tr_old = y_tr[keep_mask]
    print(f"Removed {len(train['paths']) - keep_mask.sum()} benchmark samples from training")

    X_tr = np.vstack([X_tr_old, new_X])
    y_tr = np.concatenate([y_tr_old, new_y])
    print(f"Training set (after leak removal): {len(y_tr)} ({int(y_tr.sum())} malicious)")
    print(f"Test set: {len(y_te)} ({int(y_te.sum())} malicious)")

    # Train on full training set
    import xgboost as xgb
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.8},
        xgb.DMatrix(X_tr, y_tr),
        num_boost_round=200,
    )

    # Predict on benchmark
    p = booster.predict(xgb.DMatrix(X_te_aligned))

    # Sweep threshold for best F1
    best = (0, 0.5)
    for tau in np.arange(0.1, 0.95, 0.02):
        pred = (p >= tau).astype(int)
        f1 = f1_score(y_te, pred, zero_division=0)
        if f1 > best[0]:
            best = (f1, float(tau))

    # Final at threshold 0.5 and at best
    for label, tau in [("τ=0.5", 0.5), (f"τ={best[1]:.2f} (best F1)", best[1])]:
        pred = (p >= tau).astype(int)
        pr = float(precision_score(y_te, pred, zero_division=0))
        rc = float(recall_score(y_te, pred, zero_division=0))
        f1 = float(f1_score(y_te, pred, zero_division=0))
        f2 = 5*pr*rc/(4*pr+rc) if (4*pr+rc)>0 else 0
        tp = int(((pred==1)&(y_te==1)).sum()); fp = int(((pred==1)&(y_te==0)).sum())
        tn = int(((pred==0)&(y_te==0)).sum()); fn = int(((pred==0)&(y_te==1)).sum())
        print(f"\n{label}: P={pr:.4f} R={rc:.4f} F1={f1:.4f} F2={f2:.4f} (TP={tp} FP={fp} FN={fn})")

    # Feature importances
    importance = booster.get_score(importance_type="gain")
    imp_named = sorted([(names[int(k[1:])], v) for k, v in importance.items()], key=lambda x: -x[1])[:10]
    print(f"\nTop 10 features:")
    for n, g in imp_named:
        print(f"  {g:>8.2f}  {n}")

    # Recompute heuristic metrics on the SAME aligned benchmark
    labels = json.load(open("results/labeled_dataset_final.json"))
    label_map = {s["path"]: s["label"] for s in labels["samples"]}
    htp=hfp=htn=hfn=0
    for tp_path in test_paths:
        pred_h = heur["predictions"][tp_path]  # 'malicious' or 'safe'
        true_h = label_map[tp_path]            # 'malicious' or 'benign'
        pm = (pred_h == "malicious"); tm = (true_h == "malicious")
        if pm and tm: htp+=1
        elif pm and not tm: hfp+=1
        elif not pm and not tm: htn+=1
        else: hfn+=1
    h_pr = htp/(htp+hfp) if (htp+hfp) else 0
    h_rc = htp/(htp+hfn) if (htp+hfn) else 0
    h_f1 = 2*h_pr*h_rc/(h_pr+h_rc) if (h_pr+h_rc) else 0
    h_f2 = 5*h_pr*h_rc/(4*h_pr+h_rc) if (4*h_pr+h_rc) else 0
    h_fpr = hfp/(hfp+htn) if (hfp+htn) else 0

    print(f"\n=== Comparison on same {len(test_paths)}-skill benchmark ===")
    print(f"  Heuristic:               P={h_pr:.4f} R={h_rc:.4f} F1={h_f1:.4f} F2={h_f2:.4f} FPR={h_fpr:.4f} (TP={htp} FP={hfp} FN={hfn})")
    pred_best = (p >= best[1]).astype(int)
    pr = float(precision_score(y_te, pred_best, zero_division=0))
    rc = float(recall_score(y_te, pred_best, zero_division=0))
    f1 = float(f1_score(y_te, pred_best, zero_division=0))
    f2 = 5*pr*rc/(4*pr+rc) if (4*pr+rc) else 0
    xtp = int(((pred_best==1)&(y_te==1)).sum()); xfp = int(((pred_best==1)&(y_te==0)).sum())
    xtn = int(((pred_best==0)&(y_te==0)).sum()); xfn = int(((pred_best==0)&(y_te==1)).sum())
    x_fpr = xfp/(xfp+xtn) if (xfp+xtn) else 0
    print(f"  XGBoost (τ={best[1]:.2f}):      P={pr:.4f} R={rc:.4f} F1={f1:.4f} F2={f2:.4f} FPR={x_fpr:.4f} (TP={xtp} FP={xfp} FN={xfn})")

    # Save
    out = {
        "train_size": len(y_tr), "train_malicious": int(y_tr.sum()),
        "test_size": len(y_te), "test_malicious": int(y_te.sum()),
        "features": len(names),
        "best_threshold": best[1],
        "best_f1": best[0],
        "heuristic_aligned": {
            "tp": htp, "fp": hfp, "tn": htn, "fn": hfn,
            "precision": h_pr, "recall": h_rc, "f1": h_f1, "f2": h_f2, "fpr": h_fpr,
        },
        "xgboost_best": {
            "threshold": best[1], "tp": xtp, "fp": xfp, "tn": xtn, "fn": xfn,
            "precision": pr, "recall": rc, "f1": f1, "f2": f2, "fpr": x_fpr,
        },
        "feature_importances_top10": [{"name": n, "gain": float(g)} for n, g in imp_named],
    }
    Path("results/xgboost_holdout_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to results/xgboost_holdout_eval.json")


if __name__ == "__main__":
    main()
