"""Arsitektur baseline yang dipakai bersama pada baseline dan tuning."""
from torch import nn
from .cnn import CNNFeatureExtractor, PositionalEncoding

class CNNHierarchicalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['cnn_channels'][-1]
        self.cnn = CNNFeatureExtractor(config['n_leads'], config['cnn_channels'], config['cnn_kernels'])
        self.pos_enc_1 = PositionalEncoding(d_model, dropout=config['transformer_dropout'])
        enc_layer_1 = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=config['transformer_heads'],
            dim_feedforward=d_model * 4, dropout=config['transformer_dropout'], batch_first=True)
        self.transformer_level1 = nn.TransformerEncoder(enc_layer_1, num_layers=1)
        self.downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True))
        self.pos_enc_2 = PositionalEncoding(d_model, dropout=config['transformer_dropout'])
        enc_layer_2 = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=config['transformer_heads'],
            dim_feedforward=d_model * 4, dropout=config['transformer_dropout'], batch_first=True)
        self.transformer_level2 = nn.TransformerEncoder(enc_layer_2, num_layers=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True),
            nn.Dropout(config['fc_dropout']),
            nn.Linear(d_model // 2, config['n_classes']))

    def forward(self, x):
        features = self.cnn(x)
        seq = features.permute(0, 2, 1)
        seq = self.pos_enc_1(seq)
        local_out = self.transformer_level1(seq)
        ds = local_out.permute(0, 2, 1)
        ds = self.downsample(ds)
        ds = ds.permute(0, 2, 1)
        ds = self.pos_enc_2(ds)
        global_out = self.transformer_level2(ds)
        pooled = global_out.mean(dim=1)
        return self.classifier(pooled)
