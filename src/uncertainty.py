import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import WindEnergyDataset
from model import TimeSeriesTransformer
import os

def estimate_uncertainty():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Running Monte Carlo Dropout Analysis...")
    
    # --- PATH FIX ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'Turbine_Data.csv')
    model_path = os.path.join(script_dir, 'best_model.pth')
    
    # 1. Load Data
    ds = WindEnergyDataset(data_path, mode='val')
    
    # Safe index access
    try:
        x_sample, y_sample = ds[150] 
    except IndexError:
        x_sample, y_sample = ds[50] # Fallback
        
    x_in = x_sample.unsqueeze(0).to(device)
    y_true = y_sample.numpy()[:, 0]
    
    # 2. Load Model
    model = TimeSeriesTransformer().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Error: best_model.pth not found. Run train.py first.")
        return
    
    # 3. Monte Carlo Loop
    model.train() # ENABLE DROPOUT
    
    predictions = []
    n_samples = 100
    
    print(f"Sampling {n_samples} times...")
    with torch.no_grad():
        for i in range(n_samples):
            pred = model(x_in).cpu().numpy()[0, :, 0]
            predictions.append(pred)
            
    predictions = np.array(predictions)
    
    # 4. Stats
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    lower_bound = mean_pred - 1.96 * std_pred
    upper_bound = mean_pred + 1.96 * std_pred
    
    # 5. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, 'k-', linewidth=2, label='Réalité')
    plt.plot(mean_pred, 'r-', linewidth=2, label='Prédiction Moyenne')
    plt.fill_between(range(len(mean_pred)), lower_bound, upper_bound, 
                     color='red', alpha=0.3, label='Incertitude (95%)')
    
    plt.title("Quantification d'Incertitude (Monte Carlo Dropout)")
    plt.xlabel("Temps (10 min)")
    plt.ylabel("Puissance Normalisée")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(script_dir, 'plot_uncertainty.png')
    plt.savefig(save_path)
    print(f">>> Saved 'plot_uncertainty.png' to {save_path}")

if __name__ == "__main__":
    estimate_uncertainty()