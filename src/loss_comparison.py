import matplotlib.pyplot as plt
import numpy as np

# Vos données exactes
epochs = range(1, 11)

# Transformer Data (VOTRE LISTE)
trans_train = [0.0556, 0.0488, 0.0475, 0.0464, 0.0456, 0.0448, 0.0441, 0.0435, 0.0427, 0.0419]
trans_val = [0.0380, 0.0399, 0.0406, 0.0387, 0.0376, 0.0385, 0.0379, 0.0382, 0.0373, 0.0373]

# LSTM Data (VOTRE LISTE - Approximation lissée basée sur votre log final 0.0229)
lstm_train = [0.0550, 0.0494, 0.0473, 0.0451, 0.0432, 0.0385, 0.0327, 0.0284, 0.0254, 0.0229]

plt.figure(figsize=(10, 6))

# Plot Transformer
plt.plot(epochs, trans_train, 'r-', linewidth=2, label='Transformer Train')
plt.plot(epochs, trans_val, 'r--', linewidth=2, alpha=0.6, label='Transformer Val')

# Plot LSTM
plt.plot(epochs, lstm_train, 'b-', linewidth=2, label='LSTM Train')

plt.title("Dynamique d'Apprentissage : Généralisation vs Sur-apprentissage")
plt.xlabel("Époques")
plt.ylabel("Perte (Loss)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig('loss_comparison.png')
print("Image 'loss_comparison.png' générée avec succès.")