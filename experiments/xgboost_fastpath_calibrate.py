"""Exp A: Calibrate τ_high and τ_low for the XGBoost fast-path using GroupKFold OOF.

Output:
  τ_high = min τ s.t. precision of predictions with p≥τ on OOF ≥ 0.99
  τ_low  = max τ s.t. NO true malicious has p<τ (i.e. all mal have p>τ_low)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from skillsieve.core.layer1.classifier import Layer1Classifier
from skillsieve.parser import parse_skill_package


def main():
    # Load original training set
    d = json.load(open("results/features_expanded.json"))
    X_old = np.array(d["X"])
    y_old = np.array(d["y"])
    paths_old = d["paths"]
    names = d["names"]
    existing = set(paths_old)

    # Augment with IOC-fetched skills
    skills_dir = Path("/tmp/openclaw-skills-audit/skills")
    classifier = Layer1Classifier()
    new_X, new_paths = [], []
    for author_dir in sorted(skills_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        for skill_dir in author_dir.iterdir():
            if not (skill_dir / "SKILL.md").exists():
                continue
            rel = f"{author_dir.name}/{skill_dir.name}"
            if rel in existing:
                continue
            try:
                pkg = parse_skill_package(skill_dir)
                vec, n = classifier.assemble_features(pkg)
                fmap = dict(zip(n, vec))
                new_X.append([float(fmap.get(k, 0.0)) for k in names])
                new_paths.append(rel)
            except Exception:
                pass
    new_X = np.array(new_X) if new_X else np.zeros((0, len(names)))
    new_y = np.ones(len(new_X), dtype=int)

    X = np.vstack([X_old, new_X])
    y = np.concatenate([y_old, new_y])
    paths = paths_old + new_paths
    authors = [p.split("/")[0] for p in paths]
    print(f"Training set: {len(y)} samples, {int(y.sum())} malicious, "
          f"{len(set(authors))} authors")

    import xgboost as xgb
    gkf = GroupKFold(n_splits=5)
    oof_p = np.full(len(y), np.nan)

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=authors)):
        booster = xgb.train(
            {"objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
             "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42},
            xgb.DMatrix(X[tr], y[tr]),
            num_boost_round=200,
        )
        oof_p[va] = booster.predict(xgb.DMatrix(X[va]))
        print(f"  Fold {fold}: n_val={len(va)}, n_val_mal={int(y[va].sum())}, "
              f"mean p_mal={oof_p[va][y[va]==1].mean():.3f}, "
              f"mean p_ben={oof_p[va][y[va]==0].mean():.3f}")

    assert not np.isnan(oof_p).any()

    # τ_high: minimum threshold where precision@(p≥τ) ≥ 0.99
    taus = np.arange(0.01, 1.00, 0.01)
    tau_high = None
    for t in taus[::-1]:
        pred = oof_p >= t
        if pred.sum() == 0:
            continue
        prec = (pred & (y == 1)).sum() / pred.sum()
        if prec >= 0.99:
            tau_high = float(t)
            # look for lowest τ that still satisfies ≥0.99
            break
    if tau_high is not None:
        for t in np.arange(tau_high - 0.01, 0.01, -0.01):
            pred = oof_p >= t
            if pred.sum() == 0:
                continue
            prec = (pred & (y == 1)).sum() / pred.sum()
            if prec < 0.99:
                break
            tau_high = float(t)

    # τ_low: max τ such that ALL true malicious have p > τ
    mal_mask = (y == 1)
    if mal_mask.any():
        min_mal_p = float(oof_p[mal_mask].min())
        # τ_low must be strictly less than min_mal_p (strict <) for 100% recall
        # Use round-down to be safe
        tau_low = round(min_mal_p - 0.001, 3)
        tau_low = max(tau_low, 0.0)
    else:
        tau_low = 0.0

    # Report metrics at candidate thresholds
    print(f"\n=== Threshold calibration (OOF, {len(y)} samples) ===")
    for t in [tau_low, 0.1, 0.3, 0.5, 0.7, 0.9, tau_high or 0.95]:
        if t is None:
            continue
        pred = (oof_p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f"  τ={t:.3f}: P={prec:.4f} R={rec:.4f} F1={f1:.4f} "
              f"(TP={tp} FP={fp} FN={fn} TN={tn})")

    # Count fast-path hits at chosen thresholds
    n_mal_fast = int(((oof_p >= (tau_high or 0.95)) & (y == 1)).sum())
    n_ben_fast = int(((oof_p <= tau_low) & (y == 0)).sum())
    n_ben_false = int(((oof_p >= (tau_high or 0.95)) & (y == 0)).sum())  # FP of fast-path malicious
    n_mal_false = int(((oof_p <= tau_low) & (y == 1)).sum())             # FN of fast-path benign

    print(f"\n=== Fast-path coverage (OOF) ===")
    print(f"  τ_high = {tau_high}")
    print(f"  τ_low  = {tau_low}")
    print(f"  fast-ACCEPT (p≥τ_high): {n_mal_fast} malicious + {n_ben_false} benign"
          f" (FP of fast-path)")
    print(f"  fast-REJECT (p≤τ_low):  {n_ben_fast} benign + {n_mal_false} malicious"
          f" (FN of fast-path)")
    n_total = len(y)
    n_fast = n_mal_fast + n_ben_false + n_ben_fast + n_mal_false
    print(f"  total fast-path covered: {n_fast} / {n_total} = {100*n_fast/n_total:.1f}%")

    out = {
        "n_samples": int(len(y)),
        "n_malicious": int(y.sum()),
        "tau_high": tau_high,
        "tau_low": tau_low,
        "oof_probabilities": {
            "min_malicious": float(oof_p[mal_mask].min()) if mal_mask.any() else None,
            "max_malicious": float(oof_p[mal_mask].max()) if mal_mask.any() else None,
            "min_benign": float(oof_p[~mal_mask].min()),
            "max_benign": float(oof_p[~mal_mask].max()),
        },
        "fastpath_coverage": {
            "fast_accept_true_mal": n_mal_fast,
            "fast_accept_fp": n_ben_false,
            "fast_reject_true_ben": n_ben_fast,
            "fast_reject_fn": n_mal_false,
            "total_covered": n_fast,
            "total_samples": n_total,
        },
    }
    Path("results/xgboost_fastpath_calibration.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to results/xgboost_fastpath_calibration.json")


if __name__ == "__main__":
    main()
