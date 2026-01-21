import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=2, output_dim=1, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        # 1. Feature Embedding (6 inputs -> 64 latent dims)
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding (Injects Time Order)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder (The Attention Mechanism)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Decoder (Project back to Power Output)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src shape: [Batch, 144, 6]
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        prediction = self.decoder(output)
        return prediction

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)