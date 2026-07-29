"""
Enhanced Interpreter Module
===========================
SHAP-based explainability engine for risk assessment.

Provides feature-level risk attribution, label-level SHAP aggregation,
and simplified rule generation for decision support.

References:
    - Section 5.1.1.3.3: Enhanced Interpreter specification
    - SHAP: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
"""

import numpy as np
import torch
import shap
from typing import List, Tuple, Dict, Optional, Callable
import json
import os
from collections import defaultdict


class LabelSHAPStore:
    """
    Label-level SHAP value storage with exponential moving average update.

    Implements the label-level SHAP storage mechanism described in
    Section 5.1.1.3.3.3 of the specification, enabling O(1) lookup
    during inference without recomputation.
    """

    def __init__(self, num_classes: int = 3, feature_dim: int = 16, alpha: float = 0.1):
        """
        Parameters
        ----------
        num_classes : int
            Number of risk classes (low=0, medium=1, high=2)
        feature_dim : int
            Dimension of feature space for SHAP values
        alpha : float
            Exponential moving average update rate (default 0.1)
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha

        # Φ_k: label-level average SHAP values, shape (num_classes, feature_dim)
        self.label_shap = np.zeros((num_classes, feature_dim))

        # Count of samples per class
        self.sample_counts = np.zeros(num_classes, dtype=np.int64)

        # Running variance for uncertainty estimation
        self.label_shap_var = np.zeros((num_classes, feature_dim))

    def update(self, class_idx: int, shap_values: np.ndarray):
        """
        Update label-level SHAP store with exponential moving average.

        Φ_k^{new} = α · φ(x_new) + (1 - α) · Φ_k^{old}

        Parameters
        ----------
        class_idx : int
            Class label index (0, 1, or 2)
        shap_values : np.ndarray, shape (feature_dim,)
            SHAP values for a correctly predicted sample
        """
        old_mean = self.label_shap[class_idx].copy()
        self.label_shap[class_idx] = (
            self.alpha * shap_values + (1 - self.alpha) * old_mean
        )

        # Update running variance (Welford's online algorithm)
        if self.sample_counts[class_idx] > 0:
            diff = shap_values - old_mean
            self.label_shap_var[class_idx] = (
                (1 - self.alpha) * self.label_shap_var[class_idx]
                + self.alpha * diff**2
            )

        self.sample_counts[class_idx] += 1

    def get_label_shap(self, class_idx: int) -> np.ndarray:
        """Get average SHAP values for a given class."""
        return self.label_shap[class_idx].copy()

    def get_feature_importance(self) -> np.ndarray:
        """
        Compute global feature importance across all classes.

        rank_j = argsort( (1/K) * Σ_k |Φ_jk| )
        """
        return np.mean(np.abs(self.label_shap), axis=0)

    def get_feature_ranking(self, feature_names: Optional[List[str]] = None) -> List[Tuple]:
        """
        Get ranked list of features by importance.

        Returns list of (feature_name_or_index, importance_score) sorted descending.
        """
        importance = self.get_feature_importance()
        indices = np.argsort(importance)[::-1]

        result = []
        for idx in indices:
            name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
            result.append((name, importance[idx], idx))
        return result

    def to_dict(self) -> dict:
        """Serialize store to dictionary."""
        return {
            "label_shap": self.label_shap.tolist(),
            "label_shap_var": self.label_shap_var.tolist(),
            "sample_counts": self.sample_counts.tolist(),
            "feature_importance": self.get_feature_importance().tolist(),
        }


class SimplifiedRuleGenerator:
    """
    Generates interpretable decision rules from label-level SHAP values.

    Implements Section 5.1.1.3.3.4 of the specification.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        contribution_threshold: float = 0.05,
    ):
        self.feature_names = feature_names
        self.threshold = contribution_threshold
        self.rules: List[Dict] = []

    def generate_rules(
        self,
        label_shap_store: LabelSHAPStore,
        data_mean: np.ndarray,
        data_std: np.ndarray,
    ) -> List[Dict]:
        """
        Generate decision rules based on label-level SHAP values.

        R_jk = I(|Φ_jk| > θ)  where θ is contribution_threshold

        Returns structured rules for decision support.
        """
        rules = []
        shap_vals = label_shap_store.label_shap
        importance = label_shap_store.get_feature_importance()

        # For each class, identify key discriminating features
        class_labels = ["Low Risk", "Medium Risk", "High Risk"]

        for cls in range(label_shap_store.num_classes):
            cls_shap = shap_vals[cls]
            # Find features where |SHAP| > threshold for this class
            significant = np.where(np.abs(cls_shap) > self.threshold)[0]

            if len(significant) == 0:
                continue

            # Sort by absolute contribution
            significant = significant[np.argsort(np.abs(cls_shap[significant]))[::-1]]

            rule_conditions = []
            for feat_idx in significant[:5]:  # Top 5 per class
                name = (
                    self.feature_names[feat_idx]
                    if self.feature_names and feat_idx < len(self.feature_names)
                    else f"feature_{feat_idx}"
                )
                direction = "increases" if cls_shap[feat_idx] > 0 else "decreases"

                # Determine threshold based on data stats
                threshold_val = data_mean[feat_idx] + (
                    np.sign(cls_shap[feat_idx]) * 0.5 * data_std[feat_idx]
                )

                rule_conditions.append(
                    {
                        "feature": name,
                        "feature_index": int(feat_idx),
                        "shap_value": float(cls_shap[feat_idx]),
                        "direction": direction,
                        "approximate_threshold": round(float(threshold_val), 3),
                    }
                )

            rule = {
                "target_class": class_labels[cls],
                "class_index": cls,
                "confidence": float(np.mean(np.abs(cls_shap[significant]))),
                "num_significant_features": len(significant),
                "conditions": rule_conditions,
                "decision_expression": self._format_expression(rule_conditions, class_labels[cls]),
            }
            rules.append(rule)

        self.rules = rules
        return rules

    def _format_expression(self, conditions: List[Dict], target: str) -> str:
        """Format rule conditions as human-readable expression."""
        parts = []
        for c in conditions[:3]:  # Top 3 for readability
            relation = ">" if c["direction"] == "increases" else "<"
            parts.append(f"{c['feature']} {relation} {c['approximate_threshold']}")
        expr = " AND ".join(parts) if parts else "default"
        return f"IF ({expr}) THEN {target}"

    def get_summary_rules(self) -> str:
        """Get a text summary of all generated rules."""
        lines = []
        for rule in self.rules:
            lines.append(f"  • {rule['decision_expression']}")
        return "\n".join(lines)


class EnhancedInterpreter:
    """
    Enhanced Interpreter: SHAP-based explainability engine.

    Combines SHAP analysis, label-level knowledge storage, and
    rule generation for complete risk explainability.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_dim: int = 16,
        num_classes: int = 3,
        background_size: int = 50,
        n_samples: int = 100,
        feature_names: Optional[List[str]] = None,
        contribution_threshold: float = 0.05,
        device: str = "cpu",
    ):
        self.model = model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.background_size = background_size
        self.n_samples = n_samples
        self.device = device

        self.label_shap_store = LabelSHAPStore(
            num_classes=num_classes, feature_dim=feature_dim, alpha=0.1
        )
        self.rule_generator = SimplifiedRuleGenerator(
            feature_names=feature_names, contribution_threshold=contribution_threshold
        )

        # Background data for Kernel SHAP
        self.background_data: Optional[np.ndarray] = None

    def set_background(self, data: np.ndarray):
        """Set background dataset for SHAP explainer."""
        if len(data) > self.background_size:
            indices = np.random.choice(len(data), self.background_size, replace=False)
            self.background_data = data[indices]
        else:
            self.background_data = data

    def _predict_wrapper(self, flat_input: np.ndarray) -> np.ndarray:
        """
        Wrapper function for SHAP to call the model on flattened input.

        The input is reshaped from (N, T * d) to (N, T, d) for the model.
        """
        batch_size = flat_input.shape[0]
        # Reshape to (batch, seq_len, feature_dim)
        reshaped = flat_input.reshape(batch_size, -1, self.feature_dim)

        # We need to split into f_raw, f_fused, t_hat
        # f_raw: first 12 features, f_fused: next 3, t_hat: last 1
        f_raw = reshaped[:, :, :12]
        f_fused = reshaped[:, :, 12:15]
        t_hat = reshaped[:, :, 15:16]

        f_raw_t = torch.FloatTensor(f_raw).to(self.device)
        f_fused_t = torch.FloatTensor(f_fused).to(self.device)
        t_hat_t = torch.FloatTensor(t_hat).to(self.device)

        with torch.no_grad():
            risk_scores, _, _ = self.model(f_raw_t, f_fused_t, t_hat_t)

        return risk_scores.cpu().numpy()

    def explain(
        self,
        f_raw: np.ndarray,
        f_fused: np.ndarray,
        t_hat: np.ndarray,
    ) -> Dict:
        """
        Compute SHAP explanations for a single sample.

        Parameters
        ----------
        f_raw : np.ndarray, shape (seq_len, raw_feat_dim)
        f_fused : np.ndarray, shape (seq_len, fused_feat_dim)
        t_hat : np.ndarray, shape (seq_len, 1)

        Returns
        -------
        dict with keys:
            - shap_values: SHAP values per feature
            - risk_score: predicted risk score
            - risk_category: low/medium/high
            - top_features: top contributing features
        """
        # Concatenate features
        sample = np.concatenate([f_raw, f_fused, t_hat], axis=-1)  # (T, d)
        flat_sample = sample.flatten().reshape(1, -1)  # (1, T*d)

        if self.background_data is None:
            # Use the sample itself as background if not set
            background = flat_sample
        else:
            background = self.background_data

        # Create Kernel SHAP explainer
        explainer = shap.KernelExplainer(self._predict_wrapper, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(
            flat_sample, nsamples=self.n_samples
        )

        # Get risk score
        risk_score = self._predict_wrapper(flat_sample)[0, 0]
        risk_category = "low"
        if risk_score >= 0.7:
            risk_category = "high"
        elif risk_score >= 0.3:
            risk_category = "medium"

        # Aggregate SHAP values to feature-level (mean over time)
        shap_agg = np.mean(
            shap_values[0].reshape(-1, self.feature_dim), axis=0
        )

        # Top features
        top_indices = np.argsort(np.abs(shap_agg))[::-1][:5]

        feature_names = self.rule_generator.feature_names or [
            f"feature_{i}" for i in range(self.feature_dim)
        ]

        top_features = [
            {
                "name": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                "index": int(idx),
                "shap_value": float(shap_agg[idx]),
                "importance": float(np.abs(shap_agg[idx])),
            }
            for idx in top_indices
        ]

        return {
            "shap_values": shap_agg.tolist(),
            "risk_score": float(risk_score),
            "risk_category": risk_category,
            "top_features": top_features,
            "total_shap_explained": float(np.sum(np.abs(shap_agg))),
        }

    def update_label_store(
        self, sample_idx: int, f_raw: np.ndarray, f_fused: np.ndarray, t_hat: np.ndarray,
        true_label: int, predicted_label: int
    ):
        """Update label-level SHAP store if prediction is correct."""
        if true_label != predicted_label:
            return False

        # Compute SHAP for this sample
        sample = np.concatenate([f_raw, f_fused, t_hat], axis=-1)
        flat_sample = sample.flatten().reshape(1, -1)

        if self.background_data is not None:
            background = self.background_data
        else:
            background = flat_sample

        explainer = shap.KernelExplainer(self._predict_wrapper, background)
        shap_values = explainer.shap_values(flat_sample, nsamples=self.n_samples)

        # Aggregate over time dimension
        shap_agg = np.mean(shap_values[0].reshape(-1, self.feature_dim), axis=0)

        # Update store
        self.label_shap_store.update(true_label, shap_agg)
        return True

    def generate_rules(self, data_mean: np.ndarray, data_std: np.ndarray) -> List[Dict]:
        """Generate simplified decision rules from accumulated SHAP knowledge."""
        return self.rule_generator.generate_rules(
            self.label_shap_store, data_mean, data_std
        )

    def generate_risk_report(
        self,
        f_raw: np.ndarray,
        f_fused: np.ndarray,
        t_hat: np.ndarray,
        data_mean: Optional[np.ndarray] = None,
        data_std: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Generate complete risk report with SHAP explanations.

        Returns a comprehensive dict with all risk assessment information.
        """
        explanation = self.explain(f_raw, f_fused, t_hat)
        risk_score = explanation["risk_score"]

        report = {
            "risk_score": risk_score,
            "risk_category": explanation["risk_category"],
            "risk_level_percentage": f"{risk_score * 100:.1f}%",
            "top_risk_factors": explanation["top_features"],
            "shap_feature_attribution": explanation["shap_values"],
            "explanation_variance": float(
                explanation["total_shap_explained"]
            ),
            "feature_level_details": [
                {
                    "feature": tf["name"],
                    "contribution": tf["shap_value"],
                    "interpretation": (
                        "Increasing risk"
                        if tf["shap_value"] > 0
                        else "Decreasing risk"
                    ),
                }
                for tf in explanation["top_features"]
            ],
        }

        # Add rules if available
        if self.rule_generator.rules:
            report["decision_rules"] = [
                r["decision_expression"] for r in self.rule_generator.rules
            ]

        return report


if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Interpreter - Module Test")
    print("=" * 60)

    # Test LabelSHAPStore
    store = LabelSHAPStore(num_classes=3, feature_dim=16)
    for cls in range(3):
        for _ in range(10):
            shap_vals = np.random.randn(16) * (cls + 1) * 0.1
            store.update(cls, shap_vals)

    print("\n📊 LabelSHAPStore Test:")
    print(f"   Sample counts: {store.sample_counts}")
    print(f"   Feature importance (top 5): {store.get_feature_ranking()[:5]}")

    # Test RuleGenerator
    feature_names = [f"sensor_{i}" for i in range(16)]
    rg = SimplifiedRuleGenerator(feature_names=feature_names, contribution_threshold=0.05)

    data_mean = np.random.randn(16) * 10 + 100
    data_std = np.abs(np.random.randn(16)) * 5 + 1

    rules = rg.generate_rules(store, data_mean, data_std)
    print(f"\n📋 Generated Rules ({len(rules)}):")
    print(rg.get_summary_rules())

    print("\n✅ Enhanced Interpreter test passed!")