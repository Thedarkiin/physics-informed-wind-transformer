import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from dataset import WindEnergyDataset
from model import TimeSeriesTransformer

# --- CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_NAMES = ['ActivePower', 'WindSpeed', 'Theoretical_Curve', 
                 'AmbientTemp', 'Sin_Wind', 'Cos_Wind']

def explain_model():
    print("--- Supervisor: Starting Feature Importance Analysis ---")
    
    # 1. Load Data & Model
    # Use fallback path logic like before
    csv_path = os.path.join(os.path.dirname(__file__), '../data/Turbine_Data[1].csv')
    try:
        ds = WindEnergyDataset(csv_path, mode='test')
    except:
        csv_path = r"C:\Users\aserm\OneDrive\Bureau\wind-turbine\data\Turbine_Data.csv"
        ds = WindEnergyDataset(csv_path, mode='test')

    # We use a smaller subset for explanation speed (first 500 samples)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    
    model = TimeSeriesTransformer(input_dim=6, output_dim=1).to(DEVICE)
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    model.eval()

    # 2. Calculate Baseline Error (Original Performance)
    baseline_error = 0
    criterion = torch.nn.L1Loss() # MAE
    
    batch_X, batch_y = next(iter(loader))
    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
    
    with torch.no_grad():
        original_pred = model(batch_X)
        baseline_loss = criterion(original_pred, batch_y).item()
    
    print(f"Baseline MAE: {baseline_loss:.5f}")

    # 3. Permutation Loop (The "Blindfold" Test)
    # We define Importance as: (Error with Shuffled Feature) - (Baseline Error)
    importances = []
    
    for i, feature_name in enumerate(FEATURE_NAMES):
        # Create a copy of input
        X_perturbed = batch_X.clone()
        
        # Shuffle ONLY column 'i' across the batch
        # This breaks the correlation for that specific feature
        idx = torch.randperm(X_perturbed.size(0))
        X_perturbed[:, :, i] = X_perturbed[idx, :, i]
        
        with torch.no_grad():
            pred_perturbed = model(X_perturbed)
            loss_perturbed = criterion(pred_perturbed, batch_y).item()
        
        # Importance = How much worse did it get?
        importance_score = loss_perturbed - baseline_loss
        importances.append(importance_score)
        print(f"Feature '{feature_name}' Importance: {importance_score:.5f}")

    # 4. Generate the Plot (Research Deliverable)
    plt.figure(figsize=(10, 6))
    # Normalize importances to 0-100% relative scale
    importances = np.array(importances)
    importances = 100 * (importances / importances.sum())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = plt.barh(FEATURE_NAMES, importances, color=colors)
    
    plt.xlabel('Relative Importance Contribution (%)')
    plt.title('Global Feature Importance (Permutation Method)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save
    plt.tight_layout()
    plt.savefig('plot_importance.png')
    print(">> Saved 'plot_importance.png'")

    # 5. Bonus: Error Distribution Plot (Residuals)
    # Shows if errors are "Normal" (Gaussian) or biased
    residuals = (batch_y - original_pred).cpu().numpy().flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=50, color='gray', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Error Distribution (Residuals)')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.savefig('plot_residuals.png')
    print(">> Saved 'plot_residuals.png'")

if __name__ == "__main__":
    explain_model()