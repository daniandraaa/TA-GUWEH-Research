"""Arsitektur conditional GAN dari eksperimen sebelumnya."""
import torch
from torch import nn
from torch.nn import functional as F

class Generator1D(nn.Module):
    def __init__(self, latent_dim=100, embed_dim=50, num_classes=3, out_channels=12, seq_length=5000):
        super().__init__()
        self.seq_length = seq_length
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        
        self.init_size = 312
        self.l1 = nn.Sequential(nn.Linear(latent_dim + embed_dim, 256 * self.init_size))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], -1)
        out = self.l1(x)
        out = out.view(out.shape[0], 256, self.init_size)
        out = self.conv_blocks(out)
        
        # Resize to exactly seq_length (5000)
        out = F.interpolate(out, size=self.seq_length, mode='linear', align_corners=False)
        return out

class Discriminator1D(nn.Module):
    def __init__(self, num_classes=3, embed_dim=50, in_channels=12, seq_length=5000):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        
        self.model = nn.Sequential(
            nn.Conv1d(in_channels + embed_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = self.label_emb(labels)
        c = c.unsqueeze(2).expand(-1, -1, img.size(2))
        x = torch.cat([img, c], 1)
        out = self.model(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
