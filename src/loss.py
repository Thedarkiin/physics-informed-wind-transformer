import torch
import torch.nn as nn

class RampLoss(nn.Module):
    def __init__(self, ramp_weight=0.1):
        super(RampLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ramp_weight = ramp_weight

    def forward(self, y_pred, y_true):
        # 1. Standard Error (MSE)
        mse_loss = self.mse(y_pred, y_true)
        
        # 2. Ramp/Trend Error (Derivative)
        # We compare the 'slope' of prediction vs reality
        true_slope = y_true[:, 1:] - y_true[:, :-1]
        pred_slope = y_pred[:, 1:] - y_pred[:, :-1]
        
        ramp_loss = torch.mean(torch.abs(true_slope - pred_slope))
        
        return mse_loss + (self.ramp_weight * ramp_loss)