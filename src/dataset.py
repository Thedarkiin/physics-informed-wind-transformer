import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class WindEnergyDataset(Dataset):
    def __init__(self, csv_path, seq_len=144, pred_len=144, mode='train'):
        """
        Research-Grade Data Loader adapted for Turbine_Data[1].csv
        
        Methodology:
        1. Physics Filtering: Removes sensor errors where Wind > 3m/s but Power <= 0.
        2. Empirical Reconstruction: Generates 'Theoretical_Curve' using parameters 
           derived from Non-Linear Least Squares regression on this specific dataset.
        3. Cyclic Encoding: Transforms Wind Direction into Sin/Cos components.
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # --- PHASE 1: DATA INGESTION ---
        try:
            df = pd.read_csv(csv_path)
            # Rename index column to 'Date'
            df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            raise ValueError(f"Failed to load data. Ensure path is correct: {e}")

        # --- PHASE 2: TIER 1 IMPUTATION (PHYSICS FILTER) ---
        # We remove rows where physics is violated (High Wind, Zero Power)
        # This prevents the model from learning "broken turbine" behavior.
        mask_curtailment = (df['ActivePower'] <= 0) & (df['WindSpeed'] > 3.0)
        df = df[~mask_curtailment].copy()
        
        # --- PHASE 3: TIER 2 IMPUTATION (CONTINUITY) ---
        # Linear interpolation for continuous weather variables
        df.interpolate(method='linear', limit_direction='both', inplace=True)
        
        # --- PHASE 4: FEATURE ENGINEERING (EMPIRICALLY VERIFIED) ---
        # We reconstruct the missing 'Theoretical_Power_Curve'.
        # Parameters verified via check_math.py:
        # P_max = 1799.09 kW (Capacity)
        # v_center = 6.95 m/s (Inflection point)
        # k = 1.08 (Aerodynamic slope)
        def reconstruct_curve(v):
            return 1799.09 / (1 + np.exp(-(v - 6.95) / 1.08))
        
        df['Theoretical_Curve'] = df['WindSpeed'].apply(reconstruct_curve)
        
        # Cyclic Encoding for Direction
        df['wd_sin'] = np.sin(np.deg2rad(df['WindDirection']))
        df['wd_cos'] = np.cos(np.deg2rad(df['WindDirection']))
        
        # --- PHASE 5: FEATURE SELECTION & SCALING ---
        # We select 6 features: Power, Wind, Theory, Temp, Sin_Dir, Cos_Dir
        self.feature_cols = ['ActivePower', 'WindSpeed', 'Theoretical_Curve', 
                             'AmbientTemperatue', 'wd_sin', 'wd_cos']
        
        # Scaling to [0, 1] for Neural Network stability
        self.scaler = MinMaxScaler()
        data_values = self.scaler.fit_transform(df[self.feature_cols].values)
        
        # --- PHASE 6: SPLITTING ---
        # Time-Series Split (70% Train, 20% Val, 10% Test)
        n = len(data_values)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)
        
        if mode == 'train':
            self.data = torch.FloatTensor(data_values[:train_end])
        elif mode == 'val':
            self.data = torch.FloatTensor(data_values[train_end:val_end])
        elif mode == 'test':
            self.data = torch.FloatTensor(data_values[val_end:])
            
        print(f"[{mode.upper()}] Dataset loaded. Shape: {self.data.shape}")

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        # Sliding Window Protocol
        # Input: Past 24h [144 steps]
        x = self.data[idx : idx + self.seq_len]
        
        # Target: Future 24h [144 steps]
        # We predict ONLY ActivePower (Index 0 is ActivePower)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, 0]
        
        return x, y.unsqueeze(-1)