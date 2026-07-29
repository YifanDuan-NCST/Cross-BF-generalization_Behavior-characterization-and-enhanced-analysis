"""
Test Suite for Risk Assessment & Analysis Module
=================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json


def test_risk_estimator():
    """Test RiskEstimator forward pass and risk categorization."""
    from src.risk_estimator import RiskEstimator

    model = RiskEstimator()
    batch_size, seq_len = 8, 60
    f_raw = np.random.randn(batch_size, seq_len, 12).astype(np.float32)
    f_fused = np.random.randn(batch_size, seq_len, 3).astype(np.float32)
    t_hat = np.random.rand(batch_size, seq_len, 1).astype(np.float32)

    import torch
    f_raw_t = torch.FloatTensor(f_raw)
    f_fused_t = torch.FloatTensor(f_fused)
    t_hat_t = torch.FloatTensor(t_hat)

    risk_scores, h_mlp, h_lstm = model(f_raw_t, f_fused_t, t_hat_t)

    assert risk_scores.shape == (batch_size, 1), f"Expected (8, 1), got {risk_scores.shape}"
    assert h_mlp.shape == (batch_size, 64), f"Expected (8, 64), got {h_mlp.shape}"
    assert h_lstm.shape == (batch_size, seq_len, 128), f"Expected (8, 60, 128), got {h_lstm.shape}"
    assert risk_scores.min() >= 0.0 and risk_scores.max() <= 1.0, "Risk scores out of [0,1]"

    categories = model.get_risk_category(risk_scores)
    assert categories.shape == (batch_size, 1), f"Expected (8, 1), got {categories.shape}"
    assert set(categories.numpy().flatten().tolist()).issubset({0, 1, 2}), "Invalid categories"

    print("  ✅ test_risk_estimator passed")


def test_lstm_feature_extractor():
    """Test LSTM feature extraction."""
    from src.risk_estimator import LSTMFeatureExtractor

    model = LSTMFeatureExtractor(input_size=16, hidden_size=128, num_layers=2)
    batch_size, seq_len = 8, 60
    x = np.random.randn(batch_size, seq_len, 16).astype(np.float32)

    import torch
    h_seq, h_last = model(torch.FloatTensor(x))

    assert h_seq.shape == (batch_size, seq_len, 128), f"Got {h_seq.shape}"
    assert h_last.shape == (batch_size, 128), f"Got {h_last.shape}"
    print("  ✅ test_lstm_feature_extractor passed")


def test_mlp_risk_head():
    """Test MLP risk prediction head."""
    from src.risk_estimator import MLPRiskHead

    model = MLPRiskHead(input_dim=256, hidden_dims=[128, 64], dropout=0.3)
    batch_size = 8
    x = np.random.randn(batch_size, 256).astype(np.float32)

    import torch
    logits, intermediate = model(torch.FloatTensor(x))

    assert logits.shape == (batch_size, 1), f"Got {logits.shape}"
    assert intermediate.shape == (batch_size, 64), f"Got {intermediate.shape}"
    print("  ✅ test_mlp_risk_head passed")


def test_label_shap_store():
    """Test LabelSHAPStore update and query."""
    from src.enhanced_interpreter import LabelSHAPStore

    store = LabelSHAPStore(num_classes=3, feature_dim=16, alpha=0.1)
    for cls in range(3):
        for _ in range(20):
            shap_vals = np.random.randn(16) * (cls + 1) * 0.1
            store.update(cls, shap_vals)

    assert store.sample_counts.tolist() == [20, 20, 20], f"Got {store.sample_counts}"
    assert store.label_shap.shape == (3, 16), f"Got {store.label_shap.shape}"

    importance = store.get_feature_importance()
    assert importance.shape == (16,), f"Got {importance.shape}"
    assert importance.min() >= 0.0

    ranking = store.get_feature_ranking()
    assert len(ranking) == 16
    # Check descending order
    scores = [r[1] for r in ranking]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Not sorted"

    print("  ✅ test_label_shap_store passed")


def test_simplified_rule_generator():
    """Test rule generation from SHAP values."""
    from src.enhanced_interpreter import LabelSHAPStore, SimplifiedRuleGenerator

    store = LabelSHAPStore(num_classes=3, feature_dim=16)
    for cls in range(3):
        for _ in range(10):
            store.update(cls, np.random.randn(16) * 0.2 * (cls + 1))

    feature_names = [f"sensor_{i}" for i in range(16)]
    rg = SimplifiedRuleGenerator(feature_names=feature_names, contribution_threshold=0.05)

    data_mean = np.random.randn(16) * 10 + 100
    data_std = np.abs(np.random.randn(16)) * 5 + 1

    rules = rg.generate_rules(store, data_mean, data_std)
    assert isinstance(rules, list)
    assert len(rules) <= 3

    summary = rg.get_summary_rules()
    assert isinstance(summary, str)

    print("  ✅ test_simplified_rule_generator passed")


def test_data_generator():
    """Test data generation."""
    from src.data_generator import generate_sensor_data, create_dataset

    X_raw, X_fused, y, furnace_ids, calib = generate_sensor_data(
        n_samples=100, seq_length=60, seed=42
    )
    assert X_raw.shape == (100, 60, 12), f"Got {X_raw.shape}"
    assert X_fused.shape == (100, 60, 3), f"Got {X_fused.shape}"
    assert y.shape == (100,), f"Got {y.shape}"
    assert furnace_ids.shape == (100,), f"Got {furnace_ids.shape}"
    assert calib.shape == (100, 60, 1), f"Got {calib.shape}"
    assert set(y.tolist()).issubset({0, 1, 2})

    dataset = create_dataset(n_samples=100, seq_length=60, seed=42)
    for key in [
        "X_raw_train", "X_raw_val", "X_raw_test",
        "X_fused_train", "X_fused_val", "X_fused_test",
        "y_train", "y_val", "y_test",
    ]:
        assert key in dataset, f"Missing key: {key}"

    print("  ✅ test_data_generator passed")


def test_training_pipeline():
    """Test training pipeline with minimal config."""
    from src.data_generator import create_dataset
    from src.risk_estimator import RiskEstimator
    from src.train import train_model, RiskDataset

    dataset = create_dataset(n_samples=200, seq_length=60, seed=42)

    model = RiskEstimator(
        raw_feat_dim=12, fused_feat_dim=3, calib_feat_dim=1,
        seq_length=60, lstm_hidden=64, lstm_layers=1,
        mlp_dims=[128, 64], mlp_dropout=0.2,
    )

    train_ds = RiskDataset(
        dataset["X_raw_train"], dataset["X_fused_train"],
        dataset["calib_train"], dataset["y_train"],
    )
    val_ds = RiskDataset(
        dataset["X_raw_val"], dataset["X_fused_val"],
        dataset["calib_val"], dataset["y_val"],
    )

    config = {
        "training": {
            "batch_size": 16,
            "learning_rate": 0.001,
            "num_epochs": 2,
            "weight_decay": 1e-5,
            "early_stopping_patience": 10,
            "weighted_loss": True,
        }
    }

    result = train_model(
        model, train_ds, val_ds, config,
        device="cpu", checkpoint_dir="assets/test_checkpoints",
    )

    assert "best_val_loss" in result
    assert "best_val_acc" in result
    assert "history" in result
    assert os.path.exists(result["best_model_path"]), "Model checkpoint not saved"

    print("  ✅ test_training_pipeline passed")


def test_inference_pipeline():
    """Test inference pipeline end-to-end."""
    from src.risk_estimator import RiskEstimator
    from src.inference import RiskInferencePipeline

    model = RiskEstimator()
    pipeline = RiskInferencePipeline(model)

    batch_size, seq_len = 4, 60
    f_raw = np.random.randn(batch_size, seq_len, 12).astype(np.float32)
    f_fused = np.random.randn(batch_size, seq_len, 3).astype(np.float32)
    t_hat = np.random.rand(batch_size, seq_len, 1).astype(np.float32)

    results = pipeline.predict(f_raw, f_fused, t_hat)
    assert len(results["risk_scores"]) == batch_size
    assert len(results["category_labels"]) == batch_size
    assert results["category_labels"][0] in ["Low", "Medium", "High"]

    # Single sample
    report = pipeline.analyze(f_raw[0], f_fused[0], t_hat[0])
    assert "risk_score" in report
    assert "risk_category" in report

    print("  ✅ test_inference_pipeline passed")


def test_utils():
    """Test utility functions."""
    from src.utils import compute_risk_metrics, normalize_features, risk_score_to_category

    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 0, 1, 2, 2, 2, 0, 1, 1, 0])

    metrics = compute_risk_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics

    # Test normalization
    data = np.random.randn(10, 60, 16)
    norm_data, mean, std = normalize_features(data)
    assert norm_data.shape == data.shape
    assert np.abs(norm_data.mean()) < 0.1  # Should be close to 0

    # Test category conversion
    assert risk_score_to_category(0.1) == "low"
    assert risk_score_to_category(0.5) == "medium"
    assert risk_score_to_category(0.9) == "high"

    print("  ✅ test_utils passed")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 60)
    print("🧪 Risk Assessment Module - Complete Test Suite")
    print("=" * 60)

    tests = [
        ("Data Generator", test_data_generator),
        ("Risk Estimator Full", test_risk_estimator),
        ("LSTM Feature Extractor", test_lstm_feature_extractor),
        ("MLP Risk Head", test_mlp_risk_head),
        ("Label SHAP Store", test_label_shap_store),
        ("Rule Generator", test_simplified_rule_generator),
        ("Training Pipeline", test_training_pipeline),
        ("Inference Pipeline", test_inference_pipeline),
        ("Utility Functions", test_utils),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ {name} FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"📊 Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)