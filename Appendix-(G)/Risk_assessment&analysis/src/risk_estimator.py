"""
Risk Estimator Module
======================
LSTM-based temporal feature extraction + MLP risk prediction with
temperature scaling. Core component of the Risk Assessment pipeline.

Architecture:
    Input Features (F_raw, F_fused, T_hat)
        → LSTM (temporal modeling)
        → MLP (non-linear risk mapping)
        → Sigmoid (risk score calibration)
        → Temperature Scaling (score refinement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import json
import os


class LSTMFeatureExtractor(nn.Module):
    """
    LSTM-based temporal feature extraction module.

    Captures long-term dependencies in blast furnace time series data
    through forget gate, input gate, and output gate mechanisms.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)

        Returns
        -------
        h_seq : torch.Tensor, shape (batch, seq_len, hidden_size * num_directions)
            Hidden state sequence (all time steps)
        h_last : torch.Tensor, shape (batch, hidden_size * num_directions)
            Final hidden state (last time step, averaged if bidirectional)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        if self.bidirectional:
            # Concatenate forward and backward final states
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        else:
            h_last = h_n[-1]

        return lstm_out, h_last


class MLPRiskHead(nn.Module):
    """
    MLP-based risk prediction head.

    Multi-layer perceptron for non-linear transformation from
    LSTM features to risk score. Includes BatchNorm and Dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256, 128, 64],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Final risk scoring layer
        self.score_head = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Store intermediate feature dimension for interpretation
        self.intermediate_dim = prev_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)

        Returns
        -------
        risk_score : torch.Tensor, shape (batch, 1)
            Risk score in [0, 1]
        intermediate_features : torch.Tensor, shape (batch, intermediate_dim)
            Features from the last hidden layer (for SHAP analysis)
        """
        intermediate_features = self.mlp(x)
        logits = self.score_head(intermediate_features)
        return logits, intermediate_features


class RiskEstimator(nn.Module):
    """
    Complete Risk Estimator: LSTM → MLP → Sigmoid → Temperature Scaling.

    As described in Section 5.1.1.3.2 of the specification.
    """

    def __init__(
        self,
        raw_feat_dim: int = 12,
        fused_feat_dim: int = 3,
        calib_feat_dim: int = 1,
        seq_length: int = 60,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        mlp_dims: list = [512, 256, 128, 64],
        mlp_dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.seq_length = seq_length
        self.raw_feat_dim = raw_feat_dim
        self.fused_feat_dim = fused_feat_dim
        self.calib_feat_dim = calib_feat_dim

        # Total input dimension to LSTM (as per spec: F_input = Concat(F_raw, F_fused, T_hat))
        lstm_input_size = raw_feat_dim + fused_feat_dim + calib_feat_dim

        self.lstm = LSTMFeatureExtractor(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )

        # After LSTM, we flatten the sequence: (batch, seq_len * hidden_size)
        mlp_input_dim = seq_length * lstm_hidden

        self.mlp = MLPRiskHead(
            input_dim=mlp_input_dim,
            hidden_dims=mlp_dims,
            dropout=mlp_dropout,
            use_batch_norm=use_batch_norm,
        )

        # Temperature scaling parameter (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

        # Risk score statistics for min-max scaling
        self.register_buffer("r_min", torch.tensor(0.0))
        self.register_buffer("r_max", torch.tensor(1.0))

    def forward(
        self, f_raw: torch.Tensor, f_fused: torch.Tensor, t_hat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass of the Risk Estimator.

        Parameters
        ----------
        f_raw : torch.Tensor, shape (batch, seq_len, raw_feat_dim)
            Raw sensor features F_raw
        f_fused : torch.Tensor, shape (batch, seq_len, fused_feat_dim)
            Fused features F_fused
        t_hat : torch.Tensor, shape (batch, seq_len, calib_feat_dim)
            Calibrated predictions T_hat

        Returns
        -------
        risk_score : torch.Tensor, shape (batch, 1)
            Scaled risk score r_scaled in [0, 1]
        h_mlp : torch.Tensor, shape (batch, intermediate_dim)
            MLP intermediate features (64-dim as per spec)
        h_lstm : torch.Tensor, shape (batch, seq_len, lstm_hidden)
            LSTM hidden state sequence
        """
        # Concatenate input features (Eq. F_input = Concat(F_raw, F_fused, T_hat))
        x = torch.cat([f_raw, f_fused, t_hat], dim=-1)  # (B, T, d_input)

        # LSTM temporal feature extraction
        h_lstm_seq, h_lstm_last = self.lstm(x)  # (B, T, d_h), (B, d_h)

        # Flatten LSTM output sequence for MLP
        batch_size = x.size(0)
        z = h_lstm_seq.reshape(batch_size, -1)  # (B, T * d_h)

        # MLP risk prediction
        raw_risk, h_mlp = self.mlp(z)  # (B, 1), (B, 64)

        # Temperature scaling
        scaled_risk = torch.sigmoid(raw_risk / self.temperature)

        # Min-max scaling for score refinement
        if self.training:
            # Update running min/max during training
            with torch.no_grad():
                if self.r_min == 0 or self.r_min > scaled_risk.min():
                    self.r_min = scaled_risk.min().detach()
                if self.r_max == 1 or self.r_max < scaled_risk.max():
                    self.r_max = scaled_risk.max().detach()

        r_scaled = (scaled_risk - self.r_min) / (self.r_max - self.r_min + 1e-8)
        r_scaled = torch.clamp(r_scaled, 0.0, 1.0)

        return r_scaled, h_mlp, h_lstm_seq

    def predict_risk(self, f_raw, f_fused, t_hat):
        """Convenience method for inference - returns risk score and features."""
        self.eval()
        with torch.no_grad():
            risk_score, h_mlp, h_lstm = self.forward(f_raw, f_fused, t_hat)
        return risk_score, h_mlp, h_lstm

    def get_risk_category(self, risk_score: torch.Tensor) -> torch.Tensor:
        """
        Classify risk score into discrete categories based on thresholds.

        Returns:
            0 = Low risk  [0, 0.3)
            1 = Medium risk [0.3, 0.7)
            2 = High risk [0.7, 1.0]
        """
        categories = torch.zeros_like(risk_score, dtype=torch.long)
        categories[risk_score >= 0.7] = 2
        categories[(risk_score >= 0.3) & (risk_score < 0.7)] = 1
        return categories


def build_risk_estimator(config_path: str = "config/model_config.json") -> RiskEstimator:
    """Build RiskEstimator from config file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, config_path)
    with open(full_path, "r") as f:
        config = json.load(f)

    cfg = config["risk_estimator"]
    data_cfg = config["data"]

    model = RiskEstimator(
        raw_feat_dim=data_cfg["num_sensors"],
        fused_feat_dim=data_cfg["num_fused_features"],
        calib_feat_dim=1,
        seq_length=data_cfg["sequence_length"],
        lstm_hidden=cfg["lstm"]["hidden_size"],
        lstm_layers=cfg["lstm"]["num_layers"],
        lstm_dropout=cfg["lstm"]["dropout"],
        mlp_dims=cfg["mlp"]["layers"],
        mlp_dropout=cfg["mlp"]["dropout"],
        use_batch_norm=cfg["mlp"]["use_batch_norm"],
    )
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Risk Estimator - Model Architecture Test")
    print("=" * 60)

    # Load config and build model
    model = build_risk_estimator()
    print(f"\n📐 Model Architecture:\n{model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # Forward pass test
    batch_size, seq_len = 16, 60
    f_raw = torch.randn(batch_size, seq_len, 12)
    f_fused = torch.randn(batch_size, seq_len, 3)
    t_hat = torch.rand(batch_size, seq_len, 1)

    risk_score, h_mlp, h_lstm = model(f_raw, f_fused, t_hat)

    print(f"\n🧪 Forward Pass Test:")
    print(f"   Input F_raw:        {tuple(f_raw.shape)}")
    print(f"   Input F_fused:      {tuple(f_fused.shape)}")
    print(f"   Input T_hat:        {tuple(t_hat.shape)}")
    print(f"   Output Risk Score:  {tuple(risk_score.shape)}  [{risk_score.min():.4f}, {risk_score.max():.4f}]")
    print(f"   Output h_mlp:       {tuple(h_mlp.shape)}")
    print(f"   Output h_lstm:      {tuple(h_lstm.shape)}")
    print(f"\n   Risk categories: {model.get_risk_category(risk_score).squeeze().tolist()}")
    print("\n✅ Risk Estimator model test passed!")