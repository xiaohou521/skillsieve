"""Train the Layer 1 XGBoost classifier on labeled skill data."""
import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve

from skillsieve.core.layer1.classifier import Layer1Classifier
from skillsieve.parser import parse_skill_package

logger = logging.getLogger(__name__)


def collect_features(
    skills_dir: Path,
    labels: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract features from all labeled skills.

    Returns:
        X: feature matrix (n_samples, n_features)
        y: label vector (0=benign, 1=malicious)
        feature_names: list of feature names
        skill_names: list of skill names (aligned with rows)
    """
    classifier = Layer1Classifier()
    X_rows = []
    y_rows = []
    skill_names = []
    feature_names = None

    skill_dirs = sorted(d for d in Path(skills_dir).iterdir()
                        if d.is_dir() and (d / "SKILL.md").exists())

    for skill_dir in skill_dirs:
        name = skill_dir.name
        if name not in labels:
            continue
        try:
            pkg = parse_skill_package(skill_dir)
            vec, names = classifier.assemble_features(pkg)
            if feature_names is None:
                feature_names = names
            X_rows.append(vec)
            y_rows.append(1 if labels[name] == "malicious" else 0)
            skill_names.append(name)
        except Exception as e:
            logger.warning(f"Failed to process {name}: {e}")

    return (
        np.array(X_rows) if X_rows else np.empty((0, 0)),
        np.array(y_rows) if y_rows else np.empty(0),
        feature_names or [],
        skill_names,
    )


def find_optimal_threshold(
    model: xgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    target_recall: float = 0.98,
) -> float:
    """Find threshold that achieves target recall on validation set."""
    dval = xgb.DMatrix(X_val, feature_names=feature_names)
    scores = model.predict(dval)

    precisions, recalls, thresholds = precision_recall_curve(y_val, scores)

    # Find lowest threshold that achieves target recall
    for i, rec in enumerate(recalls):
        if rec >= target_recall and i < len(thresholds):
            return float(thresholds[i])

    return 0.3  # fallback default


def train(
    skills_dir: Path,
    labels_path: Path,
    output_model_path: Path,
    output_metrics_path: Path | None = None,
    val_size: float = 0.2,
    target_recall: float = 0.98,
    xgb_params: dict | None = None,
) -> dict:
    """Full training pipeline.

    Args:
        skills_dir: directory with skill packages
        labels_path: JSON file mapping skill_name -> "benign"|"malicious"
        output_model_path: where to save the trained model
        output_metrics_path: optional JSON file for metrics
        val_size: validation split ratio
        target_recall: target recall for threshold tuning
        xgb_params: XGBoost hyperparameters

    Returns:
        dict with metrics and threshold
    """
    labels = json.loads(labels_path.read_text())

    logger.info(f"Collecting features from {skills_dir}...")
    X, y, feature_names, skill_names = collect_features(skills_dir, labels)

    if len(X) == 0:
        raise ValueError("No valid skills found for training")

    logger.info(f"Dataset: {len(X)} skills, {int(y.sum())} malicious, {len(X) - int(y.sum())} benign")
    logger.info(f"Features: {len(feature_names)}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None,
    )

    # Train
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "seed": 42,
    }
    if xgb_params:
        params.update(xgb_params)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    model = xgb.train(
        params, dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    # Find optimal threshold
    threshold = find_optimal_threshold(model, X_val, y_val, feature_names, target_recall)
    logger.info(f"Optimal threshold for {target_recall:.0%} recall: {threshold:.4f}")

    # Evaluate
    val_scores = model.predict(dval)
    val_preds = (val_scores >= threshold).astype(int)

    report = classification_report(y_val, val_preds, target_names=["benign", "malicious"], output_dict=True)

    # Feature importance
    importance = model.get_score(importance_type="gain")
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    # Save model
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_model_path))

    metrics = {
        "dataset": {"total": len(X), "train": len(X_train), "val": len(X_val),
                     "malicious_ratio": float(y.mean())},
        "threshold": threshold,
        "val_metrics": {
            "precision": report["malicious"]["precision"],
            "recall": report["malicious"]["recall"],
            "f1": report["malicious"]["f1-score"],
            "accuracy": report["accuracy"],
        },
        "top_features": top_features,
        "xgb_params": params,
    }

    if output_metrics_path:
        output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        output_metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train Layer 1 XGBoost classifier")
    parser.add_argument("skills_dir", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output-model", type=Path, default=Path("skillsieve/models/xgboost_layer1.json"))
    parser.add_argument("--output-metrics", type=Path, default=Path("results/training_metrics.json"))
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--target-recall", type=float, default=0.98)
    args = parser.parse_args()

    metrics = train(args.skills_dir, args.labels, args.output_model, args.output_metrics,
                    args.val_size, args.target_recall)
    print(json.dumps(metrics, indent=2))
