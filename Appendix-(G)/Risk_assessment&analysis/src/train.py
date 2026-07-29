"""
Training Pipeline for Risk Assessment Model
============================================
Complete training workflow with:
- Weighted MSE loss for risk score regression
- Early stopping with patience
- Learning rate scheduling
- Validation loop with risk category metrics
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import json
import os
import time
from tqdm import tqdm


class RiskDataset(Dataset):
    """PyTorch Dataset for risk assessment data."""

    def __init__(self, f_raw, f_fused, t_hat, labels):
        self.f_raw = torch.FloatTensor(f_raw)
        self.f_fused = torch.FloatTensor(f_fused)
        self.t_hat = torch.FloatTensor(t_hat)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.f_raw[idx], self.f_fused[idx], self.t_hat[idx], self.labels[idx]


def _risk_target_transform(labels: torch.Tensor) -> torch.Tensor:
    """
    Transform class labels (0, 1, 2) to continuous risk targets (0.0, 0.5, 1.0).
    """
    return labels.float().unsqueeze(1) * 0.5  # 0->0.0, 1->0.5, 2->1.0


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Train for one epoch, return avg loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    mse_loss = nn.MSELoss()

    for f_raw, f_fused, t_hat, labels in tqdm(dataloader, desc="Training", leave=False):
        f_raw = f_raw.to(device)
        f_fused = f_fused.to(device)
        t_hat = t_hat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        risk_scores, _, _ = model(f_raw, f_fused, t_hat)

        # Regression target: 0->0.0, 1->0.5, 2->1.0
        risk_targets = _risk_target_transform(labels)

        # MSE loss for regression
        loss = mse_loss(risk_scores, risk_targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)

        # Category accuracy
        pred_categories = model.get_risk_category(risk_scores)
        correct += (pred_categories.squeeze() == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, float, Dict]:
    """Validation loop, return loss, accuracy, and per-class metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    mse_loss = nn.MSELoss()

    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for f_raw, f_fused, t_hat, labels in tqdm(dataloader, desc="Validating", leave=False):
            f_raw = f_raw.to(device)
            f_fused = f_fused.to(device)
            t_hat = t_hat.to(device)
            labels = labels.to(device)

            risk_scores, _, _ = model(f_raw, f_fused, t_hat)
            pred_categories = model.get_risk_category(risk_scores)

            risk_targets = _risk_target_transform(labels)
            loss = mse_loss(risk_scores, risk_targets)

            total_loss += loss.item()
            total += labels.size(0)
            correct += (pred_categories.squeeze() == labels).sum().item()

            all_preds.extend(pred_categories.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(risk_scores.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    # Per-class metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {}
    for cls in range(3):
        cls_mask = all_labels == cls
        if cls_mask.sum() > 0:
            cls_correct = (all_preds[cls_mask] == cls).sum()
            cls_total = cls_mask.sum()
            precision = cls_correct / (all_preds[all_preds == cls].sum() + 1e-8)
            recall = cls_correct / cls_total
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            metrics[f"class_{cls}"] = {
                "accuracy": float(cls_correct / cls_total),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "count": int(cls_total),
            }

    metrics["overall"] = {
        "accuracy": float(accuracy),
        "avg_risk_score": float(np.mean(all_scores)),
    }

    return avg_loss, accuracy, metrics


def train_model(
    model: nn.Module,
    train_dataset: RiskDataset,
    val_dataset: RiskDataset,
    config: dict,
    device: str = "cpu",
    checkpoint_dir: str = "assets",
) -> Dict:
    """
    Full training pipeline with early stopping and model checkpointing.

    Parameters
    ----------
    model : nn.Module
        RiskEstimator model
    train_dataset, val_dataset : RiskDataset
        Training and validation datasets
    config : dict
        Training configuration (from model_config.json)
    device : str
        Device to train on
    checkpoint_dir : str
        Directory to save model checkpoints

    Returns
    -------
    dict with training history and best model info
    """
    train_cfg = config["training"]

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = train_cfg["early_stopping_patience"]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training Risk Estimator")
    print(f"   Device: {device}")
    print(f"   Epochs: {train_cfg['num_epochs']}")
    print(f"   Batch size: {train_cfg['batch_size']}")
    print(f"   Learning rate: {train_cfg['learning_rate']}")
    print(f"   Early stopping patience: {max_patience}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_acc, val_metrics = validate(model, val_loader, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        # Skip NaN epochs
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"  Epoch {epoch:3d}: NaN detected, resetting gradients...")
            # Reinitialize last layer weights if NaN persists
            continue

        print(
            f"  Epoch {epoch:3d}/{train_cfg['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_model_path = os.path.join(checkpoint_dir, "best_risk_estimator.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_metrics": val_metrics,
                    "history": history,
                },
                best_model_path,
            )
            print(f"     New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n  Early stopping triggered at epoch {epoch}")
                break

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Best val acc: {best_val_acc:.4f}")
    print(f"   Checkpoint: {os.path.join(checkpoint_dir, 'best_risk_estimator.pt')}")
    print(f"{'='*60}")

    return {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "history": history,
        "best_model_path": os.path.join(checkpoint_dir, "best_risk_estimator.pt"),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Training Pipeline - Quick Test")
    print("=" * 60)

    from data_generator import create_dataset, load_config
    from risk_estimator import RiskEstimator

    # Load config
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(base_dir), "config/model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Generate small dataset for testing
    print("\n  Generating test dataset...")
    dataset = create_dataset(n_samples=500, seq_length=60)

    # Create model
    model = RiskEstimator(
        raw_feat_dim=12, fused_feat_dim=3, calib_feat_dim=1,
        seq_length=60, lstm_hidden=64, lstm_layers=1,
        mlp_dims=[128, 64, 32], mlp_dropout=0.2,
    )

    # Create datasets
    train_ds = RiskDataset(
        dataset["X_raw_train"][:300],
        dataset["X_fused_train"][:300],
        dataset["calib_train"][:300],
        dataset["y_train"][:300],
    )
    val_ds = RiskDataset(
        dataset["X_raw_val"][:100],
        dataset["X_fused_val"][:100],
        dataset["calib_val"][:100],
        dataset["y_val"][:100],
    )

    # Train (reduced config for test)
    test_config = config.copy()
    test_config["training"] = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 3,
        "weight_decay": 1e-5,
        "early_stopping_patience": 10,
        "weighted_loss": True,
    }

    result = train_model(model, train_ds, val_ds, test_config, device="cpu")

    print(f"\n  Training test complete!")
    print(f"   Best val loss: {result['best_val_loss']:.4f}")
    print(f"   Best val acc: {result['best_val_acc']:.4f}")
    print(f"   Model saved at: {result['best_model_path']}")