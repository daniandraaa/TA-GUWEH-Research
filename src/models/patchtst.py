"""Arsitektur baseline yang dipakai bersama pada baseline dan tuning."""
from torch import nn
from .cnn import CNNFeatureExtractor, PositionalEncoding

class CNNPatchTST(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['cnn_channels'][-1]
        self.patch_size = config['patch_size']
        self.cnn = CNNFeatureExtractor(config['n_leads'], config['cnn_channels'], config['cnn_kernels'])
        self.patch_proj = nn.Linear(self.patch_size * d_model, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=config['transformer_dropout'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=config['transformer_heads'],
            dim_feedforward=d_model * 4, dropout=config['transformer_dropout'], batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['transformer_layers'])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True),
            nn.Dropout(config['fc_dropout']),
            nn.Linear(d_model // 2, config['n_classes']))

    def forward(self, x):
        features = self.cnn(x)
        features = features.permute(0, 2, 1)
        B, S, D = features.shape
        n_patches = S // self.patch_size
        features = features[:, :n_patches * self.patch_size, :]
        patches = features.reshape(B, n_patches, self.patch_size * D)
        patches = self.patch_proj(patches)
        patches = self.pos_enc(patches)
        patches = self.transformer(patches)
        pooled = patches.mean(dim=1)
        return self.classifier(pooled)
