## Physics-Informed Transformers for Wind Power Forecasting 🌬️⚡

### 📌 Project Overview

This repository contains the implementation of a Physics-Informed Transformer for short-term wind power forecasting (24-hour horizon). Unlike standard deep learning models, this approach integrates domain knowledge (Betz's Law) directly into the feature engineering process and utilizes a custom Ramp Loss function to better detect critical power changes.

Key Features

Physics-Guided Feature Engineering: Uses a Non-Linear Least Squares (NLLS) fitted power curve as a "virtual sensor" input.

Custom Ramp Loss: A hybrid loss function (MSE + Derivative Penalty) to minimize lag during sudden power ramps.

Uncertainty Quantification: Implements Monte Carlo Dropout to provide confidence intervals for predictions.

Benchmark: Outperforms standard LSTM baselines by ~19.2% in Mean Absolute Error (MAE).

## 📂 Repository Structure
bash
```
├── data/               # Raw and processed datasets (Turbine_Data.csv)
├── src/                # Source code
│   ├── dataset.py      # Custom PyTorch Dataset with physics filtering
│   ├── model.py        # Transformer architecture definition
│   ├── train.py        # Training loop with Ramp Loss
│   └── utils.py        # Helper functions (metrics, plotting)
├── plots/              # Generated figures (Wind Rose, Benchmarks, Loss Curves)
├── checkpoints/        # Saved model weights (.pth)
└── README.md           # Project documentation
```


## 🚀 Getting Started

Prerequisites

Python 3.8+

PyTorch

Pandas, NumPy, Scikit-learn, Matplotlib

Installation
bash
```
git clone [https://github.com/Thedarkiin/physics-informed-wind-transformer.git](https://github.com/Thedarkiin/physics-informed-wind-transformer.git)
cd physics-informed-wind-transformer
pip install -r requirements.txt
```


(Note: Create a requirements.txt with torch, pandas, numpy, scikit-learn, matplotlib if you haven't already.)

## ⚙️ Usage

1. Data Preparation

Ensure Turbine_Data.csv is in the data/ directory. The WindEnergyDataset class handles cleaning (curtailment removal) and feature engineering automatically.

2. Training

Run the training script to train both the Transformer and the LSTM baseline:
bash
```
python src/train.py
```


This will:

Train the models for 10 epochs.

Save the best weights to checkpoints/.

Generate training loss curves in plots/.

3. Evaluation & Plotting

To generate the benchmark plots and metrics:
bash
```
python src/evaluate.py
```


(Assuming you have an evaluation script, or this logic is at the end of train.py)

## 📊 Results

Performance Metrics

| Model | MAE | RMSE | Improvement vs Baseline |
| Persistence | 0.0312 | 0.0421 | - |
| LSTM | 0.0229 | 0.0318 | +26.6% |
| Physics-Transformer | 0.0185 | 0.0264 | +40.7% |

Key Visualizations

1. Benchmark on Ramp Events
The Transformer (Red) anticipates sudden drops in power much better than the LSTM (Blue), which suffers from lag.

2. Uncertainty Quantification
Monte Carlo Dropout provides a 95% confidence interval (Red Band), showing higher uncertainty during volatile regimes.

## 🧠 Methodology Highlights

Physics-Informed Input

We fit a theoretical power curve $P_{theo}(v)$ to the historical data using the logistic function:

$$ P_{theo}(v) = \frac{1799.1}{1 + e^{-1.08(v - 6.95)}} $$

This synthetic feature guides the neural network, acting as a physical prior.

Ramp Loss Function

To penalize "laggy" predictions, we use a custom loss:

$$ \mathcal{L} = \text{MSE} + \lambda \cdot \frac{1}{N} \sum |\nabla y_{true} - \nabla y_{pred}| $$

This forces the model to match the slope of the power signal, not just the value.

## 👥 Author

ASERMOUH YASSIN - Data Science