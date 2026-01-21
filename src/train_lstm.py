import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WindEnergyDataset
from lstm import LSTMModel
import os

def train_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Training LSTM Baseline (Strict Comparison: 10 Epochs) on {device}...")
    
    # --- PATH FIX: Automatically find the data folder ---
    # Get the directory where THIS script is located (src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root, then down into data
    data_path = os.path.join(script_dir, '..', 'data', 'Turbine_Data.csv')
    
    # Verify file exists
    if not os.path.exists(data_path):
        print(f"ERROR: Could not find data at: {data_path}")
        print("Please check that 'Turbine_Data.csv' is inside the 'data' folder.")
        return

    # 1. Load Data
    try:
        train_ds = WindEnergyDataset(data_path, mode='train')
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # 2. Model Setup
    model = LSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 3. Training Loop (MATCHING TRANSFORMER EPOCHS = 10)
    EPOCHS = 10 
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"LSTM Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f}")
    
    # Save in the same folder as the script
    save_path = os.path.join(script_dir, 'lstm_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f">>> Saved 'lstm_model.pth' to {save_path}")

if __name__ == "__main__":
    train_lstm()