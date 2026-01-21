# Physics-Informed Transformers for Wind Power Forecasting 🌬️⚡

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This repository contains the implementation of a **Physics-Informed Transformer** designed to improve short-term wind power forecasting. Unlike traditional "Black Box" deep learning models that blindly learn from data, this project introduces a **Grey-Box** architecture that hybridizes domain-specific aerodynamic knowledge with state-of-the-art attention mechanisms.

The core objective is to mitigate **Grid Instability** by accurately predicting "Ramp Events" (sudden, high-frequency fluctuations in power output) which are notoriously difficult for standard statistical models (ARIMA) or recurrent networks (LSTMs) to capture.

## 💡 Methodology
The architecture is built upon three distinct layers:

### 1. System Identification (The Physics Layer)
Before training, we reverse-engineered the specific aerodynamic properties of the turbine from noisy SCADA data using **Non-Linear Least Squares (NLLS)** regression. We fitted a Generalised Logistic Function to derived the theoretical power curve:

$$P_{theo}(v) = \frac{L}{1 + e^{-k(v - v_0)}}$$

* **$L$ (Capacity):** 1799.09 kW (Identified Rated Power).
* **$v_0$ (Inflection):** 6.95 m/s (Max acceleration point).
* **$k$ (Slope):** 1.08 (Aerodynamic efficiency).

This derived feature acts as a "soft sensor," providing the neural network with a clean physical baseline.

### 2. Time-Series Transformer (The AI Layer)
We utilize a custom **Transformer Encoder** architecture to process time-series data:
* **Self-Attention Mechanism:** Allows the model to capture long-range temporal dependencies (e.g., thermal inertia from 6 hours ago) instantaneously, solving the "vanishing gradient" problem of LSTMs.
* **Cyclic Encoding:** Wind direction is embedded using $(\sin \theta, \cos \theta)$ to preserve topological continuity.

### 3. Custom Ramp Loss (The Optimization Layer)
To prioritize grid stability, we replaced the standard Mean Squared Error (MSE) with a custom **Ramp Loss** function. This function explicitly penalizes errors in the **first derivative** (rate of change):

$$\mathcal{L} = \text{MSE} + \lambda \cdot \frac{1}{N} \sum \left| \frac{\partial y}{\partial t}_{true} - \frac{\partial y}{\partial t}_{pred} \right|$$

This regularization forces the model to be "reactive," ensuring it predicts the *slope* of a power drop, not just the value.

## 📊 Experimental Results
The model was evaluated on a real-world SCADA dataset (Turkey, 2018-2020) containing 118k+ data points.

| Model Architecture | MAE (Normalized) | Improvement | Ramp Detection |
| :--- | :---: | :---: | :---: |
| **Persistence Baseline** | 0.1983 | - | High Lag |
| **Physics-Informed Transformer** | **0.1524** | **+23.16%** | **Near-Zero Lag** |

**Key Finding:** Feature Importance analysis confirmed that the **Theoretical Curve** (Physics) was the 2nd most critical feature for the model, validating the hybrid approach.

## 🛠️ Repository Structure
```bash
├── data/
│   └── Turbine_Data.csv    # Raw SCADA measurements
├── src/
│   ├── dataset.py          # Data pipeline: NLLS regression, cleaning, windowing
│   ├── model.py            # PyTorch implementation of Transformer Encoder
│   ├── loss.py             # Custom Ramp Loss implementation
│   ├── train.py            # Main training loop with Gradient Clipping
│   ├── evaluate.py         # Testing script for MAE metrics
│   ├── explain.py          # Permutation Feature Importance analysis
│   └── visualize.py        # Forecasting plots and Physics Curve validation
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation