"""
Inference Pipeline
==================
End-to-end inference for risk assessment and analysis.

Pipeline:
    1. Load trained RiskEstimator model
    2. Load/input blast furnace data
    3. Compute risk score
    4. Run SHAP explanation
    5. Generate risk report
    6. (Optional) Update label-level SHAP store
"""

import torch
import numpy as np
import json
import os
from typing import Optional, Dict, List, Tuple


class RiskInferencePipeline:
    """
    Complete inference pipeline for risk assessment.

    Combines RiskEstimator and EnhancedInterpreter for
    end-to-end risk analysis with explainability.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        interpreter=None,
        device: str = "cpu",
        feature_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.interpreter = interpreter
        self.device = device
        self.feature_names = feature_names

    def predict(
        self,
        f_raw: np.ndarray,
        f_fused: np.ndarray,
        t_hat: np.ndarray,
        return_features: bool = False,
    ) -> Dict:
        """
        Run risk prediction on input data.

        Parameters
        ----------
        f_raw : np.ndarray, shape (batch, seq_len, 12) or (seq_len, 12)
        f_fused : np.ndarray, shape (batch, seq_len, 3) or (seq_len, 3)
        t_hat : np.ndarray, shape (batch, seq_len, 1) or (seq_len, 1)
        return_features : bool
            Whether to return intermediate features

        Returns
        -------
        dict with risk scores, categories, and optionally features
        """
        # Handle single sample
        if f_raw.ndim == 2:
            f_raw = f_raw[np.newaxis, ...]
            f_fused = f_fused[np.newaxis, ...]
            t_hat = t_hat[np.newaxis, ...]

        f_raw_t = torch.FloatTensor(f_raw).to(self.device)
        f_fused_t = torch.FloatTensor(f_fused).to(self.device)
        t_hat_t = torch.FloatTensor(t_hat).to(self.device)

        self.model.eval()
        with torch.no_grad():
            risk_scores, h_mlp, h_lstm = self.model(f_raw_t, f_fused_t, t_hat_t)
            categories = self.model.get_risk_category(risk_scores)

        results = {
            "risk_scores": risk_scores.cpu().numpy().flatten().tolist(),
            "risk_categories": categories.cpu().numpy().flatten().tolist(),
            "category_labels": [
                ["Low", "Medium", "High"][c] for c in categories.cpu().numpy().flatten()
            ],
        }

        if return_features:
            results["h_mlp"] = h_mlp.cpu().numpy()
            results["h_lstm"] = h_lstm.cpu().numpy()

        return results

    def analyze(
        self,
        f_raw: np.ndarray,
        f_fused: np.ndarray,
        t_hat: np.ndarray,
        data_mean: Optional[np.ndarray] = None,
        data_std: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Full analysis pipeline: predict + explain.

        Parameters
        ----------
        f_raw : np.ndarray, shape (seq_len, 12)
            Single sample raw features
        f_fused : np.ndarray, shape (seq_len, 3)
            Single sample fused features
        t_hat : np.ndarray, shape (seq_len, 1)
            Single sample calibrated labels
        data_mean, data_std : optional
            Data statistics for rule generation

        Returns
        -------
        dict with complete risk analysis report
        """
        # Ensure single sample (batch=1)
        if f_raw.ndim == 3:
            f_raw = f_raw[0]
            f_fused = f_fused[0]
            t_hat = t_hat[0]

        # Run interpretation
        if self.interpreter is not None:
            report = self.interpreter.generate_risk_report(
                f_raw, f_fused, t_hat,
                data_mean=data_mean,
                data_std=data_std,
            )
        else:
            # Just risk prediction
            pred = self.predict(
                f_raw[np.newaxis, ...], f_fused[np.newaxis, ...], t_hat[np.newaxis, ...]
            )
            report = {
                "risk_score": pred["risk_scores"][0],
                "risk_category": pred["category_labels"][0],
                "top_risk_factors": [],
                "feature_level_details": [],
            }

        return report

    def batch_analyze(
        self,
        f_raw: np.ndarray,
        f_fused: np.ndarray,
        t_hat: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        update_store: bool = False,
    ) -> List[Dict]:
        """
        Run analysis on a batch of samples.

        Parameters
        ----------
        f_raw : np.ndarray, shape (N, seq_len, 12)
        f_fused : np.ndarray, shape (N, seq_len, 3)
        t_hat : np.ndarray, shape (N, seq_len, 1)
        true_labels : np.ndarray, optional, shape (N,)
            True labels for updating SHAP store
        update_store : bool
            Whether to update label-level SHAP store

        Returns
        -------
        list of dicts with per-sample analysis reports
        """
        reports = []
        n_samples = f_raw.shape[0]

        for i in range(n_samples):
            report = self.analyze(f_raw[i], f_fused[i], t_hat[i])
            reports.append(report)

            # Optionally update label SHAP store
            if (
                update_store
                and true_labels is not None
                and self.interpreter is not None
            ):
                pred_label = 0
                if report["risk_score"] >= 0.7:
                    pred_label = 2
                elif report["risk_score"] >= 0.3:
                    pred_label = 1

                self.interpreter.update_label_store(
                    i, f_raw[i], f_fused[i], t_hat[i],
                    true_labels[i], pred_label,
                )

        return reports

    def summarize_batch_results(self, reports: List[Dict]) -> Dict:
        """Generate summary statistics from batch analysis results."""
        scores = [r["risk_score"] for r in reports]
        categories = [r["risk_category"] for r in reports]

        risk_counts = {
            "low": categories.count("low"),
            "medium": categories.count("medium"),
            "high": categories.count("high"),
        }

        # Find top risk samples
        high_risk_indices = [
            i for i, c in enumerate(categories) if c == "high"
        ]

        return {
            "num_samples": len(reports),
            "avg_risk_score": float(np.mean(scores)),
            "std_risk_score": float(np.std(scores)),
            "max_risk_score": float(np.max(scores)),
            "min_risk_score": float(np.min(scores)),
            "risk_distribution": risk_counts,
            "high_risk_count": risk_counts["high"],
            "high_risk_pct": f"{risk_counts['high'] / len(reports) * 100:.1f}%",
        }


def save_risk_report(report: Dict, filepath: str):
    """Save risk report to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📄 Risk report saved to: {filepath}")


def load_model_for_inference(
    model_class,
    checkpoint_path: str,
    config_path: str = "config/model_config.json",
    device: str = "cpu",
):
    """Load trained model for inference."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_full = os.path.join(base_dir, config_path)

    with open(config_full) as f:
        config = json.load(f)

    data_cfg = config["data"]
    model = model_class(
        raw_feat_dim=data_cfg["num_sensors"],
        fused_feat_dim=data_cfg["num_fused_features"],
        calib_feat_dim=1,
        seq_length=data_cfg["sequence_length"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


if __name__ == "__main__":
    print("=" * 60)
    print("Inference Pipeline - Module Test")
    print("=" * 60)

    from risk_estimator import RiskEstimator

    # Create a simple model (untrained, for structure test)
    model = RiskEstimator()

    # Generate dummy input
    batch_size, seq_len = 4, 60
    f_raw = np.random.randn(batch_size, seq_len, 12).astype(np.float32)
    f_fused = np.random.randn(batch_size, seq_len, 3).astype(np.float32)
    t_hat = np.random.rand(batch_size, seq_len, 1).astype(np.float32)

    pipeline = RiskInferencePipeline(model)
    results = pipeline.predict(f_raw, f_fused, t_hat)

    print(f"\n🧪 Batch Prediction:")
    print(f"   Risk scores: {[f'{s:.4f}' for s in results['risk_scores']]}")
    print(f"   Categories: {results['category_labels']}")

    # Single sample analysis
    report = pipeline.analyze(f_raw[0], f_fused[0], t_hat[0])
    print(f"\n📊 Single Sample Analysis:")
    print(f"   Risk score: {report['risk_score']:.4f}")
    print(f"   Risk category: {report['risk_category']}")

    print("\n✅ Inference pipeline test passed!")