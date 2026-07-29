"""
Risk Assessment & Analysis Module
=================================
Complete implementation of blast furnace risk assessment with
LSTM-MLP risk estimation and SHAP-based explainability.

Components:
    - RiskEstimator: LSTM + MLP for risk score prediction
    - EnhancedInterpreter: SHAP-based explainability engine
    - Training & Inference pipelines
    - Visualization utilities
"""

from .risk_estimator import RiskEstimator, LSTMFeatureExtractor, MLPRiskHead, build_risk_estimator
from .enhanced_interpreter import (
    EnhancedInterpreter, LabelSHAPStore, SimplifiedRuleGenerator,
)
from .data_generator import generate_sensor_data, create_dataset
from .train import train_model, RiskDataset
from .inference import RiskInferencePipeline, load_model_for_inference
from .utils import compute_risk_metrics, normalize_features, print_evaluation_summary
from .visualization import generate_all_visualizations

__version__ = "1.0.0"
__all__ = [
    "RiskEstimator",
    "LSTMFeatureExtractor",
    "MLPRiskHead",
    "EnhancedInterpreter",
    "LabelSHAPStore",
    "SimplifiedRuleGenerator",
    "RiskInferencePipeline",
    "train_model",
    "RiskDataset",
    "generate_sensor_data",
    "create_dataset",
    "compute_risk_metrics",
    "normalize_features",
    "print_evaluation_summary",
    "generate_all_visualizations",
    "load_model_for_inference",
    "build_risk_estimator",
]