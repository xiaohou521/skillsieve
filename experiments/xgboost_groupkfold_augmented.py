"""GroupKFold after augmenting the training set with freshly-fetched IOC skills.

Adds the ~37 newly-fetched IOC malicious skills (from 6 additional authors)
to the original 1,401 training set and re-runs GroupKFold by author.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from skillsieve.core.layer1.classifier import Layer1Classifier
from skillsieve.parser import parse_skill_package


def main():
    # Load existing training set
    d = json.load(open("results/features_expanded.json"))
    X_old = np.array(d["X"])
    y_old = np.array(d["y"])
    paths_old = d["paths"]
    names = d["names"]
    existing = set(paths_old)
    print(f"Existing training: {len(y_old)} samples, {int(y_old.sum())} malicious")

    # Fetch new skills from /tmp archive
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
                # Align features to existing names
                feature_map = dict(zip(n, vec))
                row = [float(feature_map.get(name, 0.0)) for name in names]
                new_X.append(row)
                new_paths.append(rel)
            except Exception as e:
                print(f"  parse error {rel}: {str(e)[:80]}")

    print(f"New IOC malicious skills extracted: {len(new_X)}")
    new_X = np.array(new_X)
    new_y = np.ones(len(new_X), dtype=int)  # All from IOC blocklist → label malicious

    # Combine
    X = np.vstack([X_old, new_X])
    y = np.concatenate([y_old, new_y])
    paths = paths_old + new_paths
    authors = [p.split("/")[0] for p in paths]

    n_mal = int(y.sum())
    n_aut = len(set(authors))
    n_mal_aut = len({a for a, lbl in zip(authors, y) if lbl == 1})
    print(f"\nAugmented training: {len(y)} samples, {n_mal} malicious, {n_aut} authors ({n_mal_aut} malicious authors)")
    mal_dist = Counter(a for a, lbl in zip(authors, y) if lbl == 1)
    print(f"\nMalicious distribution (top 10):")
    for a, c in mal_dist.most_common(10):
        pct = 100 * c / n_mal
        print(f"  {a:<30} {c:>4}  ({pct:.1f}%)")

    # XGBoost
    try:
        import xgboost as xgb
    except ImportError:
        print("pip install xgboost"); return

    def train_eval(tr, va, fold):
        booster = xgb.train(
            {"objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
             "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "auc"},
            xgb.DMatrix(X[tr], y[tr]),
            num_boost_round=200,
            evals=[(xgb.DMatrix(X[va], y[va]), "val")],
            verbose_eval=False, early_stopping_rounds=20,
        )
        p = booster.predict(xgb.DMatrix(X[va]))
        pred = (p >= 0.5).astype(int)
        return {
            "fold": fold,
            "n_val": len(va),
            "n_val_mal": int(y[va].sum()),
            "precision": float(precision_score(y[va], pred, zero_division=0)),
            "recall": float(recall_score(y[va], pred, zero_division=0)),
            "f1": float(f1_score(y[va], pred, zero_division=0)),
        }

    print("\n=== Random Stratified 5-fold ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rand = [train_eval(tr, va, i) for i, (tr, va) in enumerate(skf.split(X, y))]
    for r in rand:
        print(f"  Fold {r['fold']}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    rf1 = [r["f1"] for r in rand]
    print(f"  Mean F1: {np.mean(rf1):.4f} ± {np.std(rf1):.4f}")

    print("\n=== GroupKFold by author ===")
    gkf = GroupKFold(n_splits=5)
    grp = [train_eval(tr, va, i) for i, (tr, va) in enumerate(gkf.split(X, y, groups=authors))]
    for r in grp:
        print(f"  Fold {r['fold']}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    gf1 = [r["f1"] for r in grp]
    print(f"  Mean F1: {np.mean(gf1):.4f} ± {np.std(gf1):.4f}")

    print(f"\n=== Delta ===")
    print(f"  Random F1:     {np.mean(rf1):.4f} ± {np.std(rf1):.4f}")
    print(f"  GroupKFold F1: {np.mean(gf1):.4f} ± {np.std(gf1):.4f}")
    print(f"  Drop: {np.mean(rf1) - np.mean(gf1):.4f}")

    out = {
        "n_samples": len(y), "n_malicious": n_mal, "n_authors": n_aut,
        "n_malicious_authors": n_mal_aut,
        "new_skills_added": len(new_X),
        "top_malicious_authors": mal_dist.most_common(20),
        "random_5fold": rand, "random_f1_mean": float(np.mean(rf1)), "random_f1_std": float(np.std(rf1)),
        "groupkfold": grp, "groupkfold_f1_mean": float(np.mean(gf1)), "groupkfold_f1_std": float(np.std(gf1)),
    }
    Path("results/xgboost_groupkfold_augmented.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to results/xgboost_groupkfold_augmented.json")


if __name__ == "__main__":
    main()
