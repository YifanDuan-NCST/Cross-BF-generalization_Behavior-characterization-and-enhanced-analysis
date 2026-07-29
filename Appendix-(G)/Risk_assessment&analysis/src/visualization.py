"""
Visualization Module for Risk Assessment
=========================================
Provides visualization tools for risk analysis results:
- Risk score distribution plots
- SHAP feature importance plots
- Confusion matrix visualization
- Risk trend over time
- Rule-based decision tree visualization
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os
from typing import Optional, List, Dict, Tuple

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "font.size": 11})


def plot_risk_score_distribution(
    risk_scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Risk Score Distribution",
):
    """
    Plot histogram of risk scores with optional ground truth overlay.

    Parameters
    ----------
    risk_scores : np.ndarray, shape (N,)
    labels : np.ndarray, shape (N,), optional
        0=low, 1=medium, 2=high
    save_path : str, optional
    title : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    axes[0].hist(risk_scores, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
    axes[0].axvline(0.3, color="orange", linestyle="--", alpha=0.8, label="Medium Risk Threshold")
    axes[0].axvline(0.7, color="red", linestyle="--", alpha=0.8, label="High Risk Threshold")
    axes[0].set_xlabel("Risk Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{title} (Overall)")
    axes[0].legend()
    axes[0].set_xlim(0, 1)

    # By class
    if labels is not None:
        colors = ["green", "orange", "red"]
        class_names = ["Low Risk", "Medium Risk", "High Risk"]
        for cls in range(3):
            mask = labels == cls
            if mask.sum() > 0:
                axes[1].hist(
                    risk_scores[mask],
                    bins=20,
                    alpha=0.5,
                    color=colors[cls],
                    label=class_names[cls],
                    edgecolor="white",
                )
        axes[1].set_xlabel("Risk Score")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"{title} (By True Class)")
        axes[1].legend()
        axes[1].set_xlim(0, 1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved: {save_path}")
    plt.close()


def plot_shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 10,
    save_path: Optional[str] = None,
    title: str = "SHAP Feature Importance",
):
    """
    Horizontal bar plot of SHAP feature importance.

    Parameters
    ----------
    shap_values : np.ndarray, shape (n_features,)
    feature_names : List[str], optional
    top_k : int
    save_path : str, optional
    title : str
    """
    n_features = len(shap_values)
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Sort by absolute SHAP value
    abs_shap = np.abs(shap_values)
    sorted_idx = np.argsort(abs_shap)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.4)))

    colors = ["#d32f2f" if v > 0 else "#1976d2" for v in shap_values[sorted_idx]]
    y_pos = np.arange(len(sorted_idx))

    ax.barh(y_pos, shap_values[sorted_idx], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("SHAP Value (impact on risk score)")
    ax.set_title(title)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)

    # Add value labels
    for i, (pos, val) in enumerate(zip(y_pos, shap_values[sorted_idx])):
        ax.text(
            val + 0.01 * (1 if val >= 0 else -1),
            pos,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ["Low Risk", "Medium Risk", "High Risk"],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
):
    """Plot confusion matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved: {save_path}")
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Training Loss", alpha=0.8)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Training Accuracy", alpha=0.8)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Validation Accuracy", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved: {save_path}")
    plt.close()


def plot_risk_trend(
    risk_scores: np.ndarray,
    threshold_low: float = 0.3,
    threshold_high: float = 0.7,
    save_path: Optional[str] = None,
    title: str = "Risk Score Trend",
):
    """Plot risk score as a time series with risk zone shading."""
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(risk_scores))

    # Risk zones
    ax.axhspan(threshold_high, 1.0, alpha=0.15, color="red", label="High Risk Zone")
    ax.axhspan(threshold_low, threshold_high, alpha=0.15, color="orange", label="Medium Risk Zone")
    ax.axhspan(0, threshold_low, alpha=0.15, color="green", label="Low Risk Zone")

    ax.plot(x, risk_scores, "b-", linewidth=2, alpha=0.8, label="Risk Score")
    ax.fill_between(x, risk_scores, alpha=0.2, color="steelblue")

    ax.axhline(threshold_low, color="orange", linestyle="--", alpha=0.5)
    ax.axhline(threshold_high, color="red", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Risk Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved: {save_path}")
    plt.close()


def plot_shap_summary(
    label_shap_store: np.ndarray,
    class_names: List[str] = ["Low Risk", "Medium Risk", "High Risk"],
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """Plot grouped bar chart comparing SHAP values across classes."""
    n_classes, n_features = label_shap_store.shape

    if feature_names is None:
        feature_names = [f"F{i}" for i in range(n_features)]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_features)
    width = 0.25
    colors = ["#4caf50", "#ff9800", "#f44336"]

    for cls in range(n_classes):
        ax.bar(
            x + cls * width - width,
            label_shap_store[cls],
            width,
            label=class_names[cls],
            color=colors[cls],
            alpha=0.7,
        )

    ax.set_xlabel("Features")
    ax.set_ylabel("Mean SHAP Value")
    ax.set_title("Label-Level SHAP Values Across Risk Classes")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved: {save_path}")
    plt.close()


def generate_all_visualizations(
    risk_scores: np.ndarray,
    labels: np.ndarray,
    shap_values: np.ndarray,
    history: Dict,
    label_shap_store: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "outputs/figures",
):
    """
    Generate all standard visualizations for a complete risk analysis report.

    Parameters
    ----------
    risk_scores : np.ndarray, shape (N,)
    labels : np.ndarray, shape (N,)
    shap_values : np.ndarray, shape (n_features,)
    history : dict with train/val loss and acc
    label_shap_store : np.ndarray, optional, shape (3, n_features)
    feature_names : list, optional
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n📈 Generating visualizations...")

    plot_risk_score_distribution(
        risk_scores, labels,
        save_path=os.path.join(output_dir, "risk_score_distribution.png"),
    )

    plot_shap_feature_importance(
        shap_values, feature_names,
        save_path=os.path.join(output_dir, "shap_feature_importance.png"),
    )

    plot_training_history(
        history,
        save_path=os.path.join(output_dir, "training_history.png"),
    )

    plot_risk_trend(
        risk_scores[:200],  # Show first 200 points
        save_path=os.path.join(output_dir, "risk_trend.png"),
    )

    if label_shap_store is not None:
        plot_shap_summary(
            label_shap_store, feature_names=feature_names,
            save_path=os.path.join(output_dir, "label_shap_summary.png"),
        )

    print(f"✅ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    print("=" * 60)
    print("Visualization Module - Quick Test")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    scores = np.random.beta(2, 5, 1000)
    labels = np.array([0 if s < 0.3 else 1 if s < 0.7 else 2 for s in scores])
    shap_vals = np.random.randn(16)
    history = {
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.25],
        "val_loss": [0.9, 0.7, 0.5, 0.4, 0.35],
        "train_acc": [0.6, 0.7, 0.8, 0.85, 0.88],
        "val_acc": [0.55, 0.65, 0.75, 0.80, 0.82],
    }
    label_shap = np.random.randn(3, 16) * 0.2

    # Test plots
    output_dir = "outputs/figures"
    feature_names = [f"sensor_{i}" for i in range(16)]

    plot_risk_score_distribution(scores, labels, os.path.join(output_dir, "test_dist.png"))
    plot_shap_feature_importance(shap_vals, feature_names, os.path.join(output_dir, "test_shap.png"))
    plot_training_history(history, os.path.join(output_dir, "test_history.png"))
    plot_risk_trend(scores[:200], save_path=os.path.join(output_dir, "test_trend.png"))
    plot_shap_summary(label_shap, feature_names=feature_names, save_path=os.path.join(output_dir, "test_shap_summary.png"))

    print("\n✅ Visualization test complete!")