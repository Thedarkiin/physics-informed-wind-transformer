import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, output_dim=1):
        super(LSTMModel, self).__init__()
        # batch_first=True -> (Batch, Seq, Features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)