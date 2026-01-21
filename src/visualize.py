import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from dataset import WindEnergyDataset
from model import TimeSeriesTransformer

# --- CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_results():
    print("--- Generating Research Plots ---")
    
    # 1. Plot Learning Curve (Evidence of Training)
    try:
        log_path = os.path.join(os.path.dirname(__file__), 'training_log.csv')
        df_log = pd.read_csv(log_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_log['train_loss'], label='Training Loss')
        plt.plot(df_log['val_loss'], label='Validation Loss', linestyle='--')
        plt.title('Model Convergence (Ramp Loss)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('plot_learning_curve.png')
        print(">> Saved 'plot_learning_curve.png'")
    except:
        print("!! Could not find training_log.csv. Run training first.")

    # 2. Plot Prediction vs Reality (The "Money Shot")
    # We take ONE sample from the Test set and visualize the 24h forecast
    csv_path = os.path.join(os.path.dirname(__file__), '../data/Turbine_Data.csv')
    try:
        ds = WindEnergyDataset(csv_path, mode='test')
        model = TimeSeriesTransformer(input_dim=6, output_dim=1).to(DEVICE)
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
        model.eval()
        
        # Grab a random sample (e.g., Index 50)
        x, y_true = ds[50]
        x = x.unsqueeze(0).to(DEVICE) # Add batch dim
        
        with torch.no_grad():
            y_pred = model(x)
            
        # Convert to CPU numpy for plotting
        y_true = y_true.squeeze().cpu().numpy()
        y_pred = y_pred.squeeze().cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual Power', color='black', linewidth=2)
        plt.plot(y_pred, label='AI Prediction', color='red', linestyle='--')
        plt.title('24-Hour Forecast Horizon (Sample)')
        plt.ylabel('Normalized Power (0-1)')
        plt.xlabel('Time Steps (10 min intervals)')
        plt.legend()
        plt.savefig('plot_forecast.png')
        print(">> Saved 'plot_forecast.png'")
        
    except Exception as e:
        print(f"!! Could not generate forecast plot: {e}")

if __name__ == "__main__":
    visualize_results()