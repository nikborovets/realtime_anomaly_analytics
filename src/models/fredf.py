import torch
import torch.nn as nn

class FreDFLoss(nn.Module):
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y_true):
        # time-domain loss
        l_time = self.mse(y_hat, y_true)
        # frequency-domain loss
        F_hat = torch.fft.rfft(y_hat, dim=-2)
        F_true = torch.fft.rfft(y_true, dim=-2)
        l_freq = torch.mean(torch.abs(F_hat - F_true))
        return self.alpha * l_freq + (1 - self.alpha) * l_time
