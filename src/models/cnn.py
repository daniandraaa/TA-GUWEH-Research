"""CNN dan positional encoding identik dengan baseline historis."""
import math
import torch
from torch import nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=12, channels=[32, 64, 128, 256], kernels=[7, 5, 3, 3]):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for ch, k in zip(channels, kernels):
            layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ])
            prev_ch = ch
        self.features = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x):
        return self.features(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])
