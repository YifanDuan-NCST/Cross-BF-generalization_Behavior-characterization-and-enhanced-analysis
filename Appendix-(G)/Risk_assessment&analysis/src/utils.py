"""
Utility Functions for Risk Assessment Module
=============================================
Helper functions for data preprocessing, metrics computation,
model evaluation, and result serialization.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)


def compute_risk_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_scores: Optional[np.ndarray] = None,
) -> Dict:
    """
    Comprehensive risk assessment metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (0=low, 1=medium, 2=high)
    y_pred : np.ndarray
        Predicted labels
    risk_scores : np.ndarray, optional
        Continuous risk scores for AUC computation

    Returns
    -------
    dict with all metrics
    """
    all_labels = [0, 1, 2]  # Ensure all classes are represented
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=all_labels)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=all_labels)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=all_labels)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["Low Risk", "Medium Risk", "High Risk"],
            labels=all_labels,
            zero_division=0,
            output_dict=True,
        ),
    }

    # Per-class metrics
    for cls, name in [(0, "low"), (1, "medium"), (2, "high")]:
        cls_mask = y_true == cls
        if cls_mask.sum() > 0:
            metrics[f"{name}_precision"] = float(
                precision_score(y_true == cls, y_pred == cls, zero_division=0)
            )
            metrics[f"{name}_recall"] = float(
                recall_score(y_true == cls, y_pred == cls, zero_division=0)
            )
            metrics[f"{name}_f1"] = float(
                f1_score(y_true == cls, y_pred == cls, zero_division=0)
            )

    # AUC (one-vs-rest for multiclass)
    if risk_scores is not None:
        try:
            # Binarize labels for OvR AUC
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=[0, 1, 2])
            # Use risk score directly
            auc_scores = []
            for i in range(3):
                if y_bin[:, i].sum() > 0:
                    auc = roc_auc_score(y_bin[:, i], risk_scores)
                    auc_scores.append(auc)
            if auc_scores:
                metrics["auc_macro"] = float(np.mean(auc_scores))
        except Exception:
            pass

    return metrics


def normalize_features(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize features.

    Parameters
    ----------
    data : np.ndarray
        Feature data to normalize
    mean, std : np.ndarray, optional
        Pre-computed statistics (for inference)

    Returns
    -------
    normalized_data, mean, std
    """
    if mean is None:
        mean = np.mean(data, axis=(0, 1), keepdims=True)
    if std is None:
        std = np.std(data, axis=(0, 1), keepdims=True) + eps

    normalized = (data - mean) / std
    return normalized, mean.squeeze(), std.squeeze()


def merge_features(
    f_raw: np.ndarray,
    f_fused: np.ndarray,
    t_hat: np.ndarray,
) -> np.ndarray:
    """
    Concatenate feature sources as specified in Section 5.1.1.3.2.1.

    F_input = Concat(F_raw, F_fused, T_hat)

    Parameters
    ----------
    f_raw : np.ndarray, shape (..., seq_len, n_raw)
    f_fused : np.ndarray, shape (..., seq_len, n_fused)
    t_hat : np.ndarray, shape (..., seq_len, 1)

    Returns
    -------
    merged : np.ndarray, shape (..., seq_len, n_raw + n_fused + 1)
    """
    return np.concatenate([f_raw, f_fused, t_hat], axis=-1)


def risk_score_to_category(score: float, thresholds: Tuple[float, float] = (0.3, 0.7)) -> str:
    """Convert continuous risk score to risk category."""
    if score >= thresholds[1]:
        return "high"
    elif score >= thresholds[0]:
        return "medium"
    else:
        return "low"


def print_evaluation_summary(metrics: Dict):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 60)
    print("📊 Model Evaluation Summary")
    print("=" * 60)

    overall = metrics.get("classification_report", {}).get("macro avg", {})
    print(f"\n  Overall Metrics:")
    print(f"    Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"    Precision: {metrics.get('precision_macro', 'N/A'):.4f}")
    print(f"    Recall:    {metrics.get('recall_macro', 'N/A'):.4f}")
    print(f"    F1-Score:  {metrics.get('f1_macro', 'N/A'):.4f}")
    if "auc_macro" in metrics:
        print(f"    AUC (OvR): {metrics['auc_macro']:.4f}")

    print(f"\n  Per-Class Metrics:")
    cm = metrics.get("confusion_matrix", [])
    if cm:
        print(f"    Confusion Matrix:")
        print(f"              Pred Low  Pred Med  Pred High")
        for i, (name, row) in enumerate(
            zip(["True Low", "True Med", "True High"], cm)
        ):
            print(f"    {name:10s}: {row[0]:8d}  {row[1]:8d}  {row[2]:8d}")

    for cls_name in ["low", "medium", "high"]:
        f1 = metrics.get(f"{cls_name}_f1", "N/A")
        rec = metrics.get(f"{cls_name}_recall", "N/A")
        prec = metrics.get(f"{cls_name}_precision", "N/A")
        if isinstance(f1, float):
            print(
                f"    {cls_name.capitalize():>8s}: "
                f"F1={f1:.4f}, Recall={rec:.4f}, Precision={prec:.4f}"
            )

    print("=" * 60)


def save_json_report(data: Dict, filepath: str):
    """Save any dict as formatted JSON report."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"📄 Report saved: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("Utils Module - Quick Test")
    print("=" * 60)

    # Test metrics computation
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 0, 1, 2, 2, 2, 0, 1, 1, 0])

    metrics = compute_risk_metrics(y_true, y_pred)
    print_evaluation_summary(metrics)

    print("\n✅ Utils test passed!")