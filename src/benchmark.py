import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import WindEnergyDataset
from lstm import LSTMModel
from model import TimeSeriesTransformer
import os

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Running Fair Benchmark (Transformer vs LSTM)...")
    
    # --- PATH FIX ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'Turbine_Data.csv')
    lstm_path = os.path.join(script_dir, 'lstm_model.pth')
    trans_path = os.path.join(script_dir, 'best_model.pth')

    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    # 1. Load Data
    ds = WindEnergyDataset(data_path, mode='val')
    
    # 2. Load Models
    # LSTM
    lstm = LSTMModel().to(device)
    if os.path.exists(lstm_path):
        lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    else:
        print("Warning: lstm_model.pth not found. Please run train_lstm.py first.")
        return
    lstm.eval()
    
    # Transformer
    transformer = TimeSeriesTransformer().to(device)
    if os.path.exists(trans_path):
        transformer.load_state_dict(torch.load(trans_path, map_location=device))
    else:
        print("Warning: best_model.pth not found. Please run train.py first.")
        return
    transformer.eval()
    
    # 3. Find a good ramp event
    # Using a try-except block in case index 120 is out of bounds
    try:
        x_sample, y_sample = ds[120] 
    except IndexError:
        x_sample, y_sample = ds[0] # Fallback
        
    x_in = x_sample.unsqueeze(0).to(device)
    
    # 4. Predict
    with torch.no_grad():
        p_trans = transformer(x_in).cpu().numpy()[0, :, 0]
        p_lstm = lstm(x_in).cpu().numpy()[0, :, 0]
        
    y_true = y_sample.numpy()[:, 0]
    
    # 5. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, 'k-', label='Real Power', linewidth=2)
    plt.plot(p_lstm, 'b--', label='LSTM (Baseline)', alpha=0.7)
    plt.plot(p_trans, 'r-', label='Physics-Transformer (Ours)', linewidth=2)
    
    plt.title("Fair Benchmark: Architecture Comparison")
    plt.xlabel("Time Steps (10 min)")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot_path = os.path.join(script_dir, 'plot_benchmark.png')
    plt.savefig(save_plot_path)
    print(f">>> Saved 'plot_benchmark.png' to {save_plot_path}")

if __name__ == "__main__":
    run_benchmark()