"""
Risk Assessment & Analysis Module - Synthetic Data Generator
============================================================
Generates synthetic blast furnace sensor data with temporal dependencies
and realistic risk patterns for testing and demonstration.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import os
import json


def load_config(config_path: str = "config/model_config.json") -> dict:
    """Load model configuration from JSON file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, config_path)
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_sensor_data(
    n_samples: int = 5000,
    seq_length: int = 60,
    n_sensors: int = 12,
    n_fused: int = 3,
    n_furnaces: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic blast furnace multi-sensor time series data.

    Parameters
    ----------
    n_samples : int
        Number of sample sequences to generate
    seq_length : int
        Length of each time series sequence
    n_sensors : int
        Number of raw sensor channels
    n_fused : int
        Number of fused feature channels
    n_furnaces : int
        Number of distinct furnace profiles
    seed : int
        Random seed for reproducibility

    Returns
    -------
    X_raw : np.ndarray, shape (n_samples, seq_length, n_sensors)
        Raw sensor features
    X_fused : np.ndarray, shape (n_samples, seq_length, n_fused)
        Fused feature representations
    y_labels : np.ndarray, shape (n_samples,)
        Risk labels (0=low, 1=medium, 2=high)
    furnace_ids : np.ndarray, shape (n_samples,)
        Furnace origin identifiers
    calibrated_preds : np.ndarray, shape (n_samples, seq_length, 1)
        Calibrated predictions from upstream
    """
    rng = np.random.RandomState(seed)

    # ---- Base sensor profiles per furnace ----
    furnace_base = {
        0: {"temp_mean": 1200, "temp_std": 50, "pressure_mean": 3.5, "pressure_std": 0.3},
        1: {"temp_mean": 1150, "temp_std": 45, "pressure_mean": 3.2, "pressure_std": 0.25},
        2: {"temp_mean": 1250, "temp_std": 55, "pressure_mean": 3.8, "pressure_std": 0.35},
    }

    # ---- Sensor name templates ----
    sensor_names = [
        "temperature_core",
        "temperature_wall",
        "temperature_top",
        "pressure_bottom",
        "pressure_top",
        "gas_flow_rate",
        "oxygen_injection",
        "coal_injection",
        "slag_viscosity",
        "iron_temp",
        "dust_concentration",
        "vibration_level",
    ]

    X_raw_list = []
    X_fused_list = []
    y_list = []
    furnace_list = []
    calib_list = []

    for i in range(n_samples):
        furnace_id = rng.randint(0, n_furnaces)
        base = furnace_base[furnace_id]

        # ---- Generate raw sensor sequence ----
        # Use autoregressive process with trend + noise
        seq_raw = np.zeros((seq_length, n_sensors))
        # Temperature sensors (0,1,2): AR(1) with drift
        t0 = rng.normal(base["temp_mean"], base["temp_std"])
        for t in range(seq_length):
            # Core temperature - AR(1) with small drift
            t0 = 0.95 * t0 + 0.05 * base["temp_mean"] + rng.normal(0, 8)
            seq_raw[t, 0] = t0
            # Wall temperature - correlated with core + extra noise
            seq_raw[t, 1] = 0.85 * t0 + rng.normal(0, 15)
            # Top temperature - weaker correlation
            seq_raw[t, 2] = 0.6 * t0 + rng.normal(0, 25)

        # Pressure sensors (3,4): dynamics
        p0 = rng.normal(base["pressure_mean"], base["pressure_std"])
        for t in range(seq_length):
            p0 = 0.9 * p0 + 0.1 * base["pressure_mean"] + rng.normal(0, 0.1)
            seq_raw[t, 3] = p0
            seq_raw[t, 4] = p0 * (0.6 + 0.1 * rng.randn())

        # Gas flow (5): related to pressure
        for t in range(seq_length):
            seq_raw[t, 5] = 150 + 20 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 5)

        # Injection rates (6,7): controlled variables
        for t in range(seq_length):
            seq_raw[t, 6] = 180 + 10 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 3)
            seq_raw[t, 7] = 120 + 15 * np.sin(2 * np.pi * t / 25 + 1) + rng.normal(0, 4)

        # Other sensors (8-11): various dynamics
        for t in range(seq_length):
            seq_raw[t, 8] = 0.5 + 0.2 * np.sin(2 * np.pi * t / 15) + 0.05 * rng.randn()
            seq_raw[t, 9] = 1480 + 20 * np.sin(2 * np.pi * t / 40) + rng.normal(0, 5)
            seq_raw[t, 10] = 30 + 10 * np.sin(2 * np.pi * t / 35) + rng.normal(0, 3)
            seq_raw[t, 11] = 2.0 + 0.5 * np.sin(2 * np.pi * t / 10) + 0.2 * rng.randn()

        # ---- Generate risk label based on sensor patterns ----
        # Initialize fused features
        seq_fused = np.zeros((seq_length, n_fused))
        corr_window = 20

        # Introduce controlled anomalies to create balanced classes
        # ~33% normal -> low risk, ~33% moderate anomaly -> medium, ~33% severe -> high
        anomaly_type = rng.choice(["none", "mild", "severe"], p=[0.33, 0.34, 0.33])

        if anomaly_type == "mild":
            # Add moderate temperature drift and pressure fluctuation
            drift = np.linspace(0, 30, seq_length) * rng.uniform(0.5, 1.0)
            seq_raw[:, 0] += drift
            seq_raw[:, 1] += drift * 0.7
            seq_raw[:, 3] += rng.normal(0, 0.5, seq_length) * np.sin(np.arange(seq_length) / 5)
        elif anomaly_type == "severe":
            # Add strong anomaly: temperature spike + pressure drop
            spike_center = rng.randint(seq_length // 4, 3 * seq_length // 4)
            spike = 80 * np.exp(-0.5 * ((np.arange(seq_length) - spike_center) / 8) ** 2)
            seq_raw[:, 0] += spike
            seq_raw[:, 1] += spike * 0.6
            seq_raw[:, 2] += spike * 0.3
            # Pressure drop
            seq_raw[:, 3] -= 1.0 + 0.5 * np.sin(np.arange(seq_length) / 3)
            # High vibration
            seq_raw[:, 11] += 3.0 + 2.0 * rng.random(seq_length)

        # Recompute fused features after anomaly injection
        z_score_0 = (seq_raw[:, 0] - base["temp_mean"]) / base["temp_std"]
        z_score_3 = (seq_raw[:, 3] - base["pressure_mean"]) / base["pressure_std"]
        seq_fused[:, 0] = 0.6 * np.abs(z_score_0) + 0.4 * np.abs(z_score_3)

        # Gradient energy
        gradients = np.diff(seq_raw[:, :4], axis=0)
        grad_energy = np.sqrt(np.mean(gradients**2, axis=1))
        seq_fused[1:, 1] = grad_energy
        seq_fused[0, 1] = seq_fused[1, 1] if seq_fused.shape[0] > 1 else 0.0

        # Cross-sensor correlation
        for t in range(seq_length):
            if t >= corr_window:
                window = seq_raw[t - corr_window : t, :4]
                corr_matrix = np.corrcoef(window.T)
                seq_fused[t, 2] = 1.0 - np.mean(np.abs(corr_matrix - np.eye(4)))

        # Risk label assignment
        anomaly_score = np.mean(seq_fused[:, 0]) + 0.3 * np.mean(seq_fused[:, 1])
        anomaly_score = np.clip(anomaly_score, 0, 3)

        if anomaly_type == "none" or anomaly_score < 0.8:
            risk_label = 0  # low
        elif anomaly_type == "mild" or anomaly_score < 1.8:
            risk_label = 1  # medium
        else:
            risk_label = 2  # high

        # ---- Generate calibrated predictions (upstream mock) ----
        calib = np.full((seq_length, 1), risk_label / 2.0)  # normalize to [0,1]
        calib += rng.normal(0, 0.05, (seq_length, 1))  # small noise
        calib = np.clip(calib, 0, 1)

        X_raw_list.append(seq_raw)
        X_fused_list.append(seq_fused)
        y_list.append(risk_label)
        furnace_list.append(furnace_id)
        calib_list.append(calib)

    X_raw = np.stack(X_raw_list, axis=0).astype(np.float32)
    X_fused = np.stack(X_fused_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    furnace_ids = np.array(furnace_list, dtype=np.int64)
    calib_preds = np.stack(calib_list, axis=0).astype(np.float32)

    return X_raw, X_fused, y, furnace_ids, calib_preds


def create_dataset(
    n_samples: int = 5000,
    seq_length: int = 60,
    n_sensors: int = 12,
    n_fused: int = 3,
    n_furnaces: int = 3,
    test_split: float = 0.2,
    val_split: float = 0.1,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Create full dataset with train/val/test splits.

    Returns a dictionary with keys:
        X_raw_train, X_raw_val, X_raw_test
        X_fused_train, X_fused_val, X_fused_test
        y_train, y_val, y_test
        furnace_train, furnace_val, furnace_test
        calib_train, calib_val, calib_test
    """
    X_raw, X_fused, y, furnace_ids, calib_preds = generate_sensor_data(
        n_samples=n_samples,
        seq_length=seq_length,
        n_sensors=n_sensors,
        n_fused=n_fused,
        n_furnaces=n_furnaces,
        seed=seed,
    )

    n = n_samples
    indices = np.random.RandomState(seed).permutation(n)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def _split(data):
        return data[train_idx], data[val_idx], data[test_idx]

    return {
        "X_raw_train": _split(X_raw)[0],
        "X_raw_val": _split(X_raw)[1],
        "X_raw_test": _split(X_raw)[2],
        "X_fused_train": _split(X_fused)[0],
        "X_fused_val": _split(X_fused)[1],
        "X_fused_test": _split(X_fused)[2],
        "y_train": _split(y)[0],
        "y_val": _split(y)[1],
        "y_test": _split(y)[2],
        "furnace_train": _split(furnace_ids)[0],
        "furnace_val": _split(furnace_ids)[1],
        "furnace_test": _split(furnace_ids)[2],
        "calib_train": _split(calib_preds)[0],
        "calib_val": _split(calib_preds)[1],
        "calib_test": _split(calib_preds)[2],
    }


def get_data_stats(dataset: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Print statistics of generated dataset."""
    rows = []
    for key, data in dataset.items():
        if isinstance(data, np.ndarray):
            rows.append(
                {
                    "Split": key.split("_")[-1].capitalize(),
                    "Variable": "_".join(key.split("_")[:-1]),
                    "Shape": str(data.shape),
                    "Dtype": str(data.dtype),
                    "Min": f"{data.min():.4f}",
                    "Max": f"{data.max():.4f}",
                    "Mean": f"{data.mean():.4f}",
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 60)
    print("Risk Assessment & Analysis - Data Generator Demo")
    print("=" * 60)

    dataset = create_dataset(n_samples=2000, seq_length=60)
    stats_df = get_data_stats(dataset)
    print("\n📊 Dataset Statistics:")
    print(stats_df.to_string(index=False))

    print(f"\n✅ Training samples: {dataset['y_train'].shape[0]}")
    print(f"✅ Validation samples: {dataset['y_val'].shape[0]}")
    print(f"✅ Test samples: {dataset['y_test'].shape[0]}")

    # Class distribution
    for split_name, arr_key in [("Train", "y_train"), ("Val", "y_val"), ("Test", "y_test")]:
        y_arr = dataset[arr_key]
        unique, counts = np.unique(y_arr, return_counts=True)
        dist = {["Low", "Medium", "High"][u]: c for u, c in zip(unique, counts)}
        print(f"\n  {split_name} class distribution: {dist}")

    print("\n🎯 Data generation complete!")