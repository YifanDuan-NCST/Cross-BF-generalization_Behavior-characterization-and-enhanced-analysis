# 🏭 Risk Assessment & Analysis Module｜风险评估与分析模块

> **Blast Furnace Risk Intelligence** — LSTM时序建模 · MLP风险预测 · SHAP可解释分析

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.52%2B-FF6F00)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📋 Table of Contents｜目录

- [Overview｜概述](#-overview)
- [Architecture｜架构](#-architecture)
- [Features｜功能特性](#-features)
- [Quick Start｜快速开始](#-quick-start)
- [Project Structure｜项目结构](#-project-structure)
- [Module Details｜模块详解](#-module-details)
- [Usage Guide｜使用指南](#-usage-guide)
- [Outputs｜输出说明](#-outputs)
- [Testing｜测试](#-testing)
- [Configuration｜配置](#-configuration)
- [Technical Highlights｜技术亮点](#-technical-highlights)

---

## 🎯 Overview｜概述

**风险评估与分析模块** 是整个智能预测系统的核心组件，专注于对高炉运行状态进行量化风险评估并提供可解释的决策支持。

**The Risk Assessment & Analysis Module** is the core component of the intelligent prediction system, focused on quantitative risk assessment of blast furnace operating conditions with explainable decision support.

该模块基于行业领先的深度学习技术栈构建，实现了从多传感器时序数据到风险决策的完整闭环：

Built on industry-leading deep learning technology, this module realizes a complete closed loop from multi-sensor time-series data to risk-informed decisions:

```
多源传感器数据 → LSTM时序建模 → MLP风险预测 → SHAP可解释分析 → 决策规则生成
Multi-sensor Data → LSTM Modeling → MLP Prediction → SHAP Analysis → Decision Rules
```

### ✨ Core Capabilities｜核心能力

| 能力 Capability | 技术实现 Technology | 精度 Precision |
|:---|:---|:---:|
| 📊 风险量化 Risk Quantification | LSTM + MLP + Sigmoid | 0-1 概率化输出 Probabilistic Output |
| 🔍 时序建模 Temporal Modeling | 2层LSTM + 滑动窗口 2-Layer LSTM | 60步时序依赖 60-Step Dependencies |
| 🧠 可解释分析 Explainability | Kernel SHAP 近似算法 | 特征级归因 Feature Attribution |
| 📋 规则生成 Rule Generation | 标签级SHAP聚合 Label-level SHAP | 结构化决策规则 Structured Rules |
| 🎯 多级风险 Multi-level Risk | 温度缩放 + 分桶 Temperature Scaling | Low / Medium / High |

---

## 🏗 Architecture｜架构

```
上游Multi-feature Fusion模块输出
Upstream Multi-feature Fusion Output
           │
           └──→ [特征拼接 Feature Concatenation] ─→ [LSTM层 LSTM Layer] ─→ [时序特征H Temporal Features]
                                      │
           历史样本SHAP库 ────────────┼───────────┐
           Historical SHAP Library    │           │
                                      ↓           ↓
                               [MLP层 MLP Layer] ─→ [Sigmoid] ─→ [Risk Score 风险得分]
                                      │
                                      ↓
                               [SHAP Analysis SHAP分析]
                                      │
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
            [当前样本归因]                     [标签级SHAP存储]
            [Sample Attribution]           [Label-level SHAP Store]
                                                  │
                                                  ↓
                                    [简易判断工具更新]
                                    [Rule Generator Update]
                                                  │
                                                  ↓
                                    [可解释风险输出]
                                    [Explainable Risk Output]
```

**数据流说明 Data Flow**：
1. **上游融合特征输入** — Upstream fused features enter the risk estimator
2. **LSTM时序依赖提取** — LSTM extracts temporal dependencies
3. **MLP非线性风险映射** — MLP performs non-linear risk mapping
4. **Sigmoid风险得分输出** — Sigmoid outputs 0-1 risk score
5. **SHAP特征归因分析** — SHAP provides feature-level attribution
6. **标签级知识库累积优化** — Label-level knowledge base accumulates and optimizes

---

## 🌟 Features｜功能特性

### 1. 🔬 Risk Estimator｜风险评估器

| 组件 Component | 描述 Description | 规格 Specification |
|:---|:---|:---:|
| **LSTM** | 长短期记忆网络提取时序特征 Long Short-Term Memory | 2层, 128隐藏单元, Dropout 0.3 |
| **MLP** | 多层感知机进行风险预测 Multi-layer Perceptron | 4层 4 Layers (512→256→128→64) |
| **Sigmoid** | 输出映射至(0,1)区间 Output Mapping | 概率化风险得分 Probabilistic Score |
| **温度缩放 Temperature Scaling** | 增强风险区分度 Enhanced Differentiation | 可学习参数 + 线性缩放 Learnable Param |

**温度缩放后的风险等级 Risk Levels after Scaling**：
```
低风险 Low:    0.0 ≤ score < 0.3
中风险 Medium: 0.3 ≤ score < 0.7
高风险 High:   0.7 ≤ score ≤ 1.0
```

### 2. 💡 Enhanced Interpreter｜增强解释器

| 组件 Component | 描述 Description | 特性 Feature |
|:---|:---|:---:|
| **SHAP分析 SHAP Analysis** | 基于博弈论的特征归因 Game Theory Attribution | Kernel SHAP近似算法 Approximation |
| **标签级存储 Label Store** | 按类别聚合SHAP值 Class-wise Aggregation | 指数滑动平均更新 EMA Update (α=0.1) |
| **规则生成 Rule Generator** | 自动生成判断规则 Auto Rule Generation | 阈值过滤 + 表达式生成 Threshold + Expression |
| **风险报告 Risk Report** | 结构化输出 Structured Output | JSON格式, 含特征级解释 JSON with Feature Explanation |

---

## 🚀 Quick Start｜快速开始

### 📦 Installation｜安装

```bash
# 进入项目目录 Navigate to project directory
cd Risk_assessment&analysis

# 安装依赖 Install dependencies
pip install -r requirements.txt
# 或使用 uv (推荐) Or using uv (recommended)
uv pip install -r requirements.txt
```

> **提示 Note**: 依赖也通过 `pyproject.toml` 管理。Dependencies are also managed via `pyproject.toml`.

### 🎮 Run Full Pipeline｜运行完整管线

```bash
# 运行端到端管线 Run the complete end-to-end pipeline
python main.py
```

**执行流程 Execution Flow**:
1. ✅ 生成合成高炉传感器数据 Generate synthetic blast furnace sensor data
2. ✅ 训练风险估计器 (LSTM + MLP) Train Risk Estimator
3. ✅ 评估模型性能 Evaluate model performance
4. ✅ 运行SHAP可解释分析 Run SHAP explainability analysis
5. ✅ 生成可视化和风险报告 Generate visualizations and risk report

### 🧪 Run Tests｜运行测试

```bash
# 运行完整测试套件 Run the complete test suite
python tests/test_module.py
```

---

## 📁 Project Structure｜项目结构

```
Risk_assessment&analysis/
├── 📂 config/
│   └── model_config.json        # 模型与训练配置 Model & Training Configuration
├── 📂 src/                      # 源代码 Source Code
│   ├── __init__.py              # 模块初始化 Module Initialization
│   ├── data_generator.py        # 合成数据生成 Synthetic Data Generation
│   ├── risk_estimator.py        # LSTM + MLP 风险估计模型 Risk Estimation Model
│   ├── enhanced_interpreter.py  # SHAP可解释分析引擎 SHAP Explainability Engine
│   ├── train.py                 # 训练管线 Training Pipeline
│   ├── inference.py             # 推理管线 Inference Pipeline
│   ├── utils.py                 # 工具函数 & 评估指标 Utilities & Metrics
│   └── visualization.py         # 可视化工具 Visualization Tools
├── 📂 assets/                   # 模型检查点 Model Checkpoints
│   └── best_risk_estimator.pt   # 训练好的模型 (生成后) Trained Model (Generated)
├── 📂 tests/                    # 测试 Test Suite
│   └── test_module.py           # 综合测试 Comprehensive Tests
├── 📂 outputs/                  # 输出 Outputs
│   ├── risk_assessment_report.json  # 风险分析报告 Risk Report (Generated)
│   └── figures/                     # 可视化图表 Visualizations
├── main.py                      # 入口 - 完整管线 Entry Point - Full Pipeline
├── pyproject.toml               # 项目配置 Project Configuration
├── requirements.txt             # Python依赖 Python Dependencies
└── README.md                    # 本文档 This File
```

---

## 📚 Module Details｜模块详解

### 1️⃣ Risk Estimator｜风险评估器

**文件 File**: `src/risk_estimator.py`

核心架构遵循论文规范，实现LSTM门控机制进行时序特征提取：

The core architecture follows the specification, implementing LSTM gating mechanisms for temporal feature extraction:

```python
# F_input = Concat(F_raw, F_fused, T_hat)
x = torch.cat([f_raw, f_fused, t_hat], dim=-1)

# LSTM时序建模 Temporal modeling
h_lstm_seq, h_lstm_last = self.lstm(x)

# 展平 & MLP Flatten & MLP
z = h_lstm_seq.reshape(batch_size, -1)
raw_risk, h_mlp = self.mlp(z)

# 温度缩放 + 最小-最大缩放 Temperature + Min-max scaling
scaled_risk = torch.sigmoid(raw_risk / self.temperature)
r_scaled = (scaled_risk - r_min) / (r_max - r_min)
```

**关键组件 Key Components**:
- `LSTMFeatureExtractor`: 2层LSTM + Dropout
- `MLPRiskHead`: 4层MLP + BatchNorm + ReLU
- `RiskEstimator`: 完整级联架构 Complete Cascade Architecture

### 2️⃣ Enhanced Interpreter｜增强解释器

**文件 File**: `src/enhanced_interpreter.py`

三阶段可解释分析框架 Three-stage explainability framework：

1. **SHAP计算 SHAP Computation**: Kernel SHAP近似算法 Approximation Algorithm
2. **标签级存储 Label-level Storage**: 指数滑动平均更新SHAP库 EMA Update
3. **规则生成 Rule Generation**: 阈值过滤 + 表达式生成 Threshold + Expression

---

## 📖 Usage Guide｜使用指南

### 仅训练 Training Only

```python
from src.data_generator import create_dataset
from src.risk_estimator import build_risk_estimator
from src.train import train_model, RiskDataset

# 生成数据 Generate data
dataset = create_dataset(n_samples=5000, seq_length=60)

# 构建模型 Build model
model = build_risk_estimator()

# 准备数据 Prepare data
train_ds = RiskDataset(dataset["X_raw_train"], dataset["X_fused_train"],
                       dataset["calib_train"], dataset["y_train"])
val_ds = RiskDataset(dataset["X_raw_val"], dataset["X_fused_val"],
                     dataset["calib_val"], dataset["y_val"])

# 开始训练 Train
result = train_model(model, train_ds, val_ds, config)
```

### 仅推理 Inference Only

```python
from src.risk_estimator import RiskEstimator
from src.inference import RiskInferencePipeline
import numpy as np

# 构建模型 Build model
model = RiskEstimator()

# 创建推理管线 Create pipeline
pipeline = RiskInferencePipeline(model)

# 单样本分析 Single sample analysis
f_raw = np.random.randn(60, 12)
f_fused = np.random.randn(60, 3)
t_hat = np.random.rand(60, 1)

report = pipeline.analyze(f_raw, f_fused, t_hat)
print(f"风险得分 Risk Score: {report['risk_score']:.2%}")
print(f"风险等级 Risk Category: {report['risk_category']}")
```

---

## 📊 Outputs｜输出说明

### 风险报告 Risk Report (`outputs/risk_assessment_report.json`)

```json
{
  "risk_score": 0.8732,
  "risk_category": "high",
  "risk_level_percentage": "87.3%",
  "top_risk_factors": [
    {"name": "Core_Temp", "shap_value": 0.234, "importance": 0.234},
    {"name": "Anomaly_Score", "shap_value": 0.187, "importance": 0.187}
  ],
  "decision_rules": [
    "IF (Core_Temp > 1245.3) AND (Anomaly_Score > 1.56) THEN High Risk"
  ]
}
```

### 可视化图表 Visualizations (`outputs/figures/`)

| 图表 Figure | 描述 Description |
|:---|:---|
| `risk_score_distribution.png` | 风险得分直方图（按类别） Risk Score Histogram by Class |
| `shap_feature_importance.png` | SHAP特征重要性排序 SHAP Feature Importance Ranking |
| `training_history.png` | 训练/验证损失与精度曲线 Training & Validation Curves |
| `risk_trend.png` | 风险得分时序趋势（含风险区间） Risk Score Time Series with Zones |
| `label_shap_summary.png` | 标签级SHAP对比 Label-level SHAP Comparison |

---

## 🧪 Testing｜测试

运行完整测试套件 Run the full test suite:

```bash
$ python tests/test_module.py

============================================================
🧪 Risk Assessment Module - Complete Test Suite
============================================================
  ✅ test_data_generator passed
  ✅ test_risk_estimator passed
  ✅ test_lstm_feature_extractor passed
  ✅ test_mlp_risk_head passed
  ✅ test_label_shap_store passed
  ✅ test_simplified_rule_generator passed
  ✅ test_training_pipeline passed
  ✅ test_inference_pipeline passed
  ✅ test_utils passed
============================================================
📊 Results: 9 passed, 0 failed, 9 total
============================================================
```

**测试覆盖 Coverage**:
- **单元测试 Unit Tests**: 模型组件 Model Components, SHAP存储 SHAP Store, 规则生成 Rule Generator
- **集成测试 Integration Tests**: 训练管线 Training Pipeline, 推理管线 Inference Pipeline
- **数据测试 Data Tests**: 合成数据完整性 Synthetic Data Integrity

---

## ⚙️ Configuration｜配置

**文件 File**: `config/model_config.json`

关键配置参数 Key configuration parameters:

```json
{
  "risk_estimator": {
    "lstm": { "hidden_size": 128, "num_layers": 2, "dropout": 0.3 },
    "mlp": { "layers": [512, 256, 128, 64], "dropout": 0.3 }
  },
  "enhanced_interpreter": {
    "shap": { "n_samples": 100, "background_size": 50 },
    "label_shap": { "num_classes": 3, "update_alpha": 0.1 }
  },
  "training": {
    "batch_size": 32, "learning_rate": 0.001, "num_epochs": 50
  },
  "data": {
    "sequence_length": 60, "num_sensors": 12, "num_samples": 5000
  }
}
```

---

## 🔬 Technical Highlights｜技术亮点

### LSTM Gate Mechanism｜LSTM门控机制

LSTM通过精巧的门控机制捕获时序依赖关系：

The LSTM captures temporal dependencies through a sophisticated gating mechanism:

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad &\text{(遗忘门)}\\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad &\text{(输入门)}\\
\tilde{c}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad &\text{(候选细胞状态)}\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad &\text{(细胞状态更新)}\\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad &\text{(输出门)}\\
h_t &= o_t \odot \tanh(c_t) \quad &\text{(隐藏状态)}
\end{aligned}
$$

### SHAP Feature Attribution｜SHAP特征归因

基于合作博弈论的Shapley值，为每个预测提供特征级归因解释：

Based on cooperative game theory Shapley values, provides feature-level attribution for each prediction:

$$
\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \cdot \bigl[f_S(x_{S \cup \{j\}}) - f_S(x_S)\bigr]
$$

### Label-Level Knowledge Accumulation｜标签级知识累积

增量式标签级SHAP更新机制，实现O(1)推理效率：

Incremental label-level SHAP update mechanism achieves O(1) inference efficiency:

$$
\Phi_k^{\text{new}} = \alpha \cdot \phi(x_{\text{new}}) + (1-\alpha) \cdot \Phi_k^{\text{old}}
$$

---

## 📄 License｜许可证

该项目基于MIT许可证开源。This project is licensed under the MIT License.
