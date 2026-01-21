import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 1. Load your specific file
df = pd.read_csv('Turbine_Data.csv') # Make sure path is right
df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# 2. Filter: Only look at valid operation (Power > 0 and Wind < 25)
# This removes the "curtailment" noise so we get the true physics curve
clean_data = df[(df['ActivePower'] > 0) & (df['WindSpeed'] < 25)].dropna()

# 3. Define the Sigmoid Function (The "S" shape)
def logistic_curve(wind_speed, p_max, v_center, steepness):
    return p_max / (1 + np.exp(-(wind_speed - v_center) / steepness))

# 4. Fit the curve (Ask Scipy to find the parameters)
# Initial guess: Max=2000, Center=8m/s, Steepness=1
p_opt, _ = curve_fit(logistic_curve, 
                     clean_data['WindSpeed'], 
                     clean_data['ActivePower'], 
                     p0=[2000, 8, 1])

print(f"--- VERIFIED PARAMETERS ---")
print(f"P_max (Max Power): {p_opt[0]:.2f} kW")     # Should be approx 1813
print(f"v_center (Midpoint): {p_opt[1]:.2f} m/s")  # Should be approx 7.0
print(f"Steepness (k): {p_opt[2]:.2f}")            # Should be approx 1.09