#!/usr/bin/env python3
"""
Risk Assessment & Analysis Module - Main Entry Point
=====================================================
Complete end-to-end pipeline:
    1. Generate synthetic blast furnace data
    2. Train RiskEstimator (LSTM + MLP)
    3. Evaluate model performance
    4. Run SHAP explainability analysis
    5. Generate visualizations and risk report
"""

import sys
import os

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
import time
from pathlib import Path


def print_header(text):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def main():
    print_header("🏭 Blast Furnace Risk Assessment & Analysis Pipeline")
    print("  Version 1.0.0")

    # ================================================================
    # Step 1: Generate Data
    # ================================================================
    print_header("Step 1/5: Generating Synthetic Blast Furnace Data")

    from src.data_generator import create_dataset, load_config

    config = load_config()
    data_cfg = config["data"]

    # Use moderate settings for demo
    n_samples = min(data_cfg["num_samples"], 1000)
    n_epochs = min(config["training"]["num_epochs"], 5)

    print(f"  Samples: {n_samples}")
    print(f"  Sequence Length: {data_cfg['sequence_length']}")
    print(f"  Sensors: {data_cfg['num_sensors']}")
    print(f"  Furnaces: {data_cfg['num_furnaces']}")
    print(f"  Epochs: {n_epochs}")

    dataset = create_dataset(
        n_samples=n_samples,
        seq_length=data_cfg["sequence_length"],
        seed=config["training"]["seed"],
    )

    for split_name, key in [("Train", "y_train"), ("Val", "y_val"), ("Test", "y_test")]:
        y = dataset[key]
        unique, counts = np.unique(y, return_counts=True)
        dist = {["Low", "Medium", "High"][u]: int(c) for u, c in zip(unique, counts)}
        print(f"  → {split_name}: {len(y)} samples, distribution: {dist}")

    # ================================================================
    # Step 2: Build and Train Model
    # ================================================================
    print_header("Step 2/5: Building and Training Risk Estimator")

    from src.risk_estimator import RiskEstimator, build_risk_estimator
    from src.train import train_model, RiskDataset

    model = build_risk_estimator()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    train_ds = RiskDataset(
        dataset["X_raw_train"], dataset["X_fused_train"],
        dataset["calib_train"], dataset["y_train"],
    )
    val_ds = RiskDataset(
        dataset["X_raw_val"], dataset["X_fused_val"],
        dataset["calib_val"], dataset["y_val"],
    )
    test_ds = RiskDataset(
        dataset["X_raw_test"], dataset["X_fused_test"],
        dataset["calib_test"], dataset["y_test"],
    )

    train_cfg = config["training"]
    train_cfg["num_epochs"] = n_epochs

    train_result = train_model(
        model, train_ds, val_ds, config,
        device="cpu",
        checkpoint_dir="assets",
    )

    # ================================================================
    # Step 3: Evaluate Model
    # ================================================================
    print_header("Step 3/5: Evaluating Model Performance")

    from torch.utils.data import DataLoader
    from src.utils import compute_risk_metrics, print_evaluation_summary
    import torch

    # Load best model
    checkpoint = torch.load(train_result["best_model_path"], weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for f_raw, f_fused, t_hat, labels in test_loader:
            risk_scores, _, _ = model(f_raw, f_fused, t_hat)
            categories = model.get_risk_category(risk_scores)

            all_scores.extend(risk_scores.numpy().flatten())
            all_preds.extend(categories.numpy().flatten())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    metrics = compute_risk_metrics(all_labels, all_preds, all_scores)
    print_evaluation_summary(metrics)

    # ================================================================
    # Step 4: SHAP Explainability
    # ================================================================
    print_header("Step 4/5: Running SHAP Explainability Analysis")

    from src.enhanced_interpreter import EnhancedInterpreter

    feature_names = [
        "Core_Temp", "Wall_Temp", "Top_Temp",
        "Bottom_Pressure", "Top_Pressure", "Gas_Flow",
        "O2_Injection", "Coal_Injection", "Slag_Viscosity",
        "Iron_Temp", "Dust_Conc", "Vibration",
        "Anomaly_Score", "Gradient_Energy", "Cross_Corr",
        "Calibrated_Label",
    ]

    # Build interpreter
    interpreter = EnhancedInterpreter(
        model=model,
        feature_dim=data_cfg["num_sensors"] + data_cfg["num_fused_features"] + 1,
        num_classes=3,
        background_size=50,
        n_samples=50,
        feature_names=feature_names,
        contribution_threshold=0.05,
    )

    # Set background data
    background = np.concatenate(
        [dataset["X_raw_train"][:50], dataset["X_fused_train"][:50], dataset["calib_train"][:50]],
        axis=-1,
    )
    background_flat = background.reshape(50, -1)
    interpreter.set_background(background_flat)

    # Run SHAP on test samples and update label store
    print("  Computing SHAP values and updating label store...")
    n_shap_samples = min(100, len(dataset["X_raw_test"]))

    for i in range(n_shap_samples):
        interpreter.update_label_store(
            i,
            dataset["X_raw_test"][i],
            dataset["X_fused_test"][i],
            dataset["calib_test"][i],
            dataset["y_test"][i],
            all_preds[i],
        )

    print(f"  Updated label store with {interpreter.label_shap_store.sample_counts} samples")

    # Generate rules
    print("  Generating simplified decision rules...")

    # Compute data stats
    all_raw = np.concatenate([dataset["X_raw_train"], dataset["X_raw_val"]], axis=0)
    all_fused = np.concatenate([dataset["X_fused_train"], dataset["X_fused_val"]], axis=0)
    all_calib = np.concatenate([dataset["calib_train"], dataset["calib_val"]], axis=0)
    all_data = np.concatenate([all_raw, all_fused, all_calib], axis=-1)
    data_mean = np.mean(all_data, axis=(0, 1))
    data_std = np.std(all_data, axis=(0, 1))

    rules = interpreter.generate_rules(data_mean, data_std)
    print(f"\n  📋 Generated {len(rules)} decision rules:")
    print(interpreter.rule_generator.get_summary_rules())

    # Sample explanation
    print("\n  📊 Sample Risk Explanation (Test Sample #0):")
    explanation = interpreter.explain(
        dataset["X_raw_test"][0],
        dataset["X_fused_test"][0],
        dataset["calib_test"][0],
    )
    print(f"     Risk Score: {explanation['risk_score']:.4f}")
    print(f"     Risk Category: {explanation['risk_category'].upper()}")
    print(f"     Top Risk Factors:")
    for tf in explanation["top_features"][:5]:
        arrow = "⬆" if tf["shap_value"] > 0 else "⬇"
        print(f"       {arrow} {tf['name']}: {tf['shap_value']:.4f}")

    # ================================================================
    # Step 5: Generate Visualizations and Reports
    # ================================================================
    print_header("Step 5/5: Generating Visualizations and Risk Report")

    from src.visualization import generate_all_visualizations
    from src.utils import save_json_report

    # Generate all visualizations
    # Compute aggregate SHAP values
    shap_aggregate = interpreter.label_shap_store.get_feature_importance()

    generate_all_visualizations(
        risk_scores=all_scores,
        labels=all_labels,
        shap_values=shap_aggregate,
        history=train_result["history"],
        label_shap_store=interpreter.label_shap_store.label_shap,
        feature_names=feature_names,
        output_dir="outputs/figures",
    )

    # Generate comprehensive risk report
    risk_report = {
        "pipeline_name": "Blast Furnace Risk Assessment & Analysis",
        "version": "1.0.0",
        "summary": {
            "total_samples": data_cfg["num_samples"],
            "sequence_length": data_cfg["sequence_length"],
            "num_sensors": data_cfg["num_sensors"],
            "num_furnaces": data_cfg["num_furnaces"],
        },
        "model_performance": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
        },
        "model_parameters": total_params,
        "training_info": {
            "best_val_loss": train_result["best_val_loss"],
            "best_val_acc": train_result["best_val_acc"],
            "epochs_trained": len(train_result["history"]["train_loss"]),
        },
        "risk_explanation_sample": explanation,
        "decision_rules": [
            {
                "target": r["target_class"],
                "expression": r["decision_expression"],
                "confidence": r["confidence"],
            }
            for r in rules
        ],
        "feature_ranking": interpreter.label_shap_store.get_feature_ranking(feature_names),
    }

    save_json_report(risk_report, "outputs/risk_assessment_report.json")

    # ================================================================
    # Completion
    # ================================================================
    print_header("✅ Pipeline Complete!")

    print(f"\n  📁 Output Directory Structure:")
    print(f"     ├── assets/best_risk_estimator.pt  (trained model)")
    print(f"     ├── outputs/")
    print(f"     │   ├── risk_assessment_report.json")
    print(f"     │   └── figures/")
    print(f"     │       ├── risk_score_distribution.png")
    print(f"     │       ├── shap_feature_importance.png")
    print(f"     │       ├── training_history.png")
    print(f"     │       ├── risk_trend.png")
    print(f"     │       └── label_shap_summary.png")
    print(f"     ├── config/model_config.json")
    print(f"     ├── src/ (module source)")
    print(f"     ├── tests/ (unit tests)")
    print(f"     └── README.md")

    print(f"\n  📊 Test Set Performance:")
    print(f"     • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"     • F1 Score:  {metrics['f1_macro']:.4f}")
    print(f"     • Precision: {metrics['precision_macro']:.4f}")
    print(f"     • Recall:    {metrics['recall_macro']:.4f}")

    print(f"\n  🎯 Risk Distribution on Test Set:")
    risk_dist = {
        "Low": int((all_labels == 0).sum()),
        "Medium": int((all_labels == 1).sum()),
        "High": int((all_labels == 2).sum()),
    }
    print(f"     {risk_dist}")

    print(f"\n  📋 Decision Rules Generated: {len(rules)}")
    print(f"  📈 Visualizations: 5 figures")
    print(f"\n  🚀 Ready for deployment! Run `python main.py` to re-run.")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n  ⏱ Total execution time: {elapsed:.1f}s")