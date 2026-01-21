import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- SETTINGS ---
plt.style.use('seaborn-v0_8-whitegrid')
csv_path = 'Turbine_Data.csv' # Make sure this matches your file name

def run_eda():
    print(">>> Generating Scientific Graphs...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Fallback for your specific path
        df = pd.read_csv(r"C:\Users\aserm\OneDrive\Bureau\wind-turbine\data\Turbine_Data.csv")
    
    # Clean just for visualization
    df['Date'] = pd.to_datetime(df['Unnamed: 0'])
    df = df.dropna()
    
    # --- GRAPH 1: The "Messy" Physics (Raw Power Curve) ---
    # Shows why the problem is hard: Real data is noisy!
    plt.figure(figsize=(10, 6))
    plt.scatter(df['WindSpeed'], df['ActivePower'], alpha=0.05, s=1, color='black', label='Raw SCADA Data')
    
    # Overlay our Theoretical Curve
    v = np.linspace(0, 25, 100)
    p_theo = 1799.09 / (1 + np.exp(-(v - 6.95) / 1.08))
    plt.plot(v, p_theo, color='red', linewidth=3, label='NLLS Physics Model')
    
    plt.title('System Identification: Raw Data vs. Derived Physics Curve')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Active Power (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_power_curve.png', dpi=300)
    print("1. Saved 'plot_power_curve.png'")

    # --- GRAPH 2: Correlation Heatmap ---
    # Scientifically justifies why we chose our features
    plt.figure(figsize=(8, 6))
    cols = ['ActivePower', 'WindSpeed', 'WindDirection', 'AmbientTemperatue']
    corr = df[cols].corr()
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plot_correlation.png', dpi=300)
    print("2. Saved 'plot_correlation.png'")

    # --- GRAPH 3: Wind Direction Distribution (Polar Plot) ---
    # Shows the "Prevailing Wind"
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # Histogram of wind direction
    ax.hist(np.deg2rad(df['WindDirection']), bins=36, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax.set_theta_zero_location('N') # North on top
    ax.set_theta_direction(-1)      # Clockwise
    plt.title("Wind Direction Distribution (Rose)", va='bottom')
    plt.tight_layout()
    plt.savefig('plot_wind_rose.png', dpi=300)
    print("3. Saved 'plot_wind_rose.png'")

if __name__ == "__main__":
    run_eda()