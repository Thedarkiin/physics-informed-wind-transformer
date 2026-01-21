import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time
import os

# --- IMPORT YOUR MODULES ---
from dataset import WindEnergyDataset
from model import TimeSeriesTransformer
from loss import RampLoss

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10 

def train():
    # 1. Setup Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps') # Mac
    else:
        device = torch.device('cpu')
        
    print(f"--- Supervisor: Starting Training on {device} ---")

    # 2. Path Configuration (Updated for User)
    # OPTION A: Relative Path (Best for sharing code)
    # Assumes structure: /wind-turbine/data/Turbine_Data.csv
    csv_path = os.path.join(os.path.dirname(__file__), '../data/Turbine_Data.csv')
    
    # OPTION B: Your Absolute Path (Uncomment if Option A fails on your machine)
    # csv_path = r"C:\Users\aserm\OneDrive\Bureau\wind-turbine\data\Turbine_Data.csv"
    
    print(f"Looking for data at: {csv_path}")

    # 3. Load Data
    try:
        print("Initializing Datasets (This takes ~15 seconds)...")
        # Mode 'train' gets first 70%, 'val' gets next 20%
        train_dataset = WindEnergyDataset(csv_path, mode='train')
        val_dataset = WindEnergyDataset(csv_path, mode='val')
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Data Loaded Successfully. Training Samples: {len(train_dataset)}")
    except Exception as e:
        print(f"\nCRITICAL ERROR: Data not found.\n1. Check if 'Turbine_Data.csv' is in the 'data' folder.\n2. Error details: {e}")
        return

    # 4. Initialize Components
    model = TimeSeriesTransformer(input_dim=6, output_dim=1).to(device)
    criterion = RampLoss(ramp_weight=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    print("--- Beginning Training Loop ---")

    # 5. Training Loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # [Training]
        model.train()
        total_train_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            
            # Gradient Clipping (Prevents model explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # [Validation]
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # [Logging]
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Time: {epoch_time:.1f}s")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("  >>> Best Model Saved")

    # 6. Save Logs
    pd.DataFrame(history).to_csv('training_log.csv', index=False)
    print("--- Done. Send 'training_log.csv' and 'best_model.pth' back to Supervisor. ---")

if __name__ == "__main__":
    train()