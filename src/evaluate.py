import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os

# Import your modules
from dataset import WindEnergyDataset
from model import TimeSeriesTransformer

# --- CONFIG ---
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():
    print(f"--- Supervisor: Starting Research Evaluation on {DEVICE} ---")
    
    # 1. Load the Test Data (The unseen 10%)
    # Using 'test' mode ensures we are evaluating on data the model NEVER saw during training
    csv_path = os.path.join(os.path.dirname(__file__), '../data/Turbine_Data[1].csv')
    try:
        test_dataset = WindEnergyDataset(csv_path, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except:
        # Fallback for direct path
        csv_path = r"C:\Users\aserm\OneDrive\Bureau\wind-turbine\data\Turbine_Data.csv"
        test_dataset = WindEnergyDataset(csv_path, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load the Trained Brain (Your Transformer)
    model = TimeSeriesTransformer(input_dim=6, output_dim=1).to(DEVICE)
    model_path = 'best_model.pth'
    
    if not os.path.exists(model_path):
        print("CRITICAL: 'best_model.pth' not found. Train the model first!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(">>> Model Loaded Successfully.")

    # 3. The Arena (Comparison Loop)
    transformer_errors = []
    baseline_errors = []
    
    print(">>> Comparing Transformer vs. Persistence Baseline...")
    
    with torch.no_grad():
        for x, y_true in test_loader:
            x, y_true = x.to(DEVICE), y_true.to(DEVICE)
            
            # --- COMBATANT 1: The Transformer ---
            y_pred_transformer = model(x)
            
            # --- COMBATANT 2: The Baseline (Persistence) ---
            # Logic: "Tomorrow looks exactly like Today"
            # We take the ActivePower from the input (Index 0) and use it as prediction
            # x shape: [Batch, 144, 6] -> We grab [Batch, 144, 0]
            y_pred_baseline = x[:, :, 0].unsqueeze(-1)
            
            # --- Calculate Mean Absolute Error (MAE) ---
            # MAE is easier to explain in a report than MSE ("We are off by X kilowatts")
            error_transformer = torch.abs(y_pred_transformer - y_true).mean().item()
            error_baseline = torch.abs(y_pred_baseline - y_true).mean().item()
            
            transformer_errors.append(error_transformer)
            baseline_errors.append(error_baseline)

    # 4. The Results (Report Deliverables)
    avg_ai_error = np.mean(transformer_errors)
    avg_base_error = np.mean(baseline_errors)
    
    print("\n" + "="*40)
    print("   RESEARCH RESULTS (PARTIE 2)")
    print("="*40)
    print(f"1. Baseline (Persistence) MAE: {avg_base_error:.4f}")
    print(f"2. AI Model (Transformer) MAE: {avg_ai_error:.4f}")
    print("-" * 40)
    
    if avg_ai_error < avg_base_error:
        improvement = (avg_base_error - avg_ai_error) / avg_base_error * 100
        print(f"SUCCESS: The AI model beat the baseline by {improvement:.2f}%")
        print("Conclusion: The Attention Mechanism successfully learned wind dynamics.")
    else:
        print("FAILURE: The AI model is worse than guessing.")
        print("Recommendation: Train for more epochs or check data scaling.")
    print("="*40)

if __name__ == "__main__":
    evaluate()