"""Compact conditional GAN, SN projection critic, exact algebraic limb leads."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


def twelve_leads(independent):
    i, ii = independent[:, 0], independent[:, 1]
    limb = torch.stack((i, ii, ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2), dim=1)
    return torch.cat((limb, independent[:, 2:]), dim=1)


class Generator(nn.Module):
    def __init__(self, latent=64, embedding=16, base=64, length=5000):
        super().__init__()
        if length % 8 or base % 16: raise ValueError('Length/base incompatible with generator')
        self.latent, self.base, self.initial = latent, base, length // 8
        self.embedding = nn.Embedding(3, embedding)
        self.start = nn.Linear(latent + embedding, base * self.initial)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(a, b, 11, padding=5),
            nn.GroupNorm(4, b), nn.ReLU()) for a, b in ((base, base // 2), (base // 2, base // 4), (base // 4, base // 4))])
        self.output = nn.Conv1d(base // 4, 8, 11, padding=5)

    def forward(self, noise, labels):
        value = self.start(torch.cat((noise, self.embedding(labels)), dim=1)).reshape(-1, self.base, self.initial)
        for block in self.blocks:
            value = block(F.interpolate(value, scale_factor=2, mode='linear', align_corners=False))
        return twelve_leads(self.output(value))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (12, 16, 32, 64, 128)
        self.features = nn.Sequential(*[nn.Sequential(
            spectral_norm(nn.Conv1d(a, b, k, stride=4, padding=k // 2)), nn.LeakyReLU(.2))
            for a, b, k in zip(channels[:-1], channels[1:], (15, 9, 7, 5))])
        self.score = spectral_norm(nn.Linear(128, 1))
        self.embedding = spectral_norm(nn.Embedding(3, 128))

    def forward(self, value, labels):
        features = self.features(value).mean(dim=-1)
        return self.score(features).squeeze(1) + (features * self.embedding(labels)).sum(dim=1)


def moment_features(value):
    """Differentiable amplitude, temporal roughness and inter-lead correlations."""
    rms = (value.square().mean(-1) + 1e-8).sqrt()
    pieces = [rms.log()]
    for lag in (1, 4, 16, 64):
        derivative = ((value[..., lag:] - value[..., :-lag]).square().mean(-1) + 1e-8).sqrt()
        pieces.append((derivative / rms).clamp_min(1e-6).log())
    centered = value - value.mean(-1, keepdim=True)
    sd = (centered.square().mean(-1) + 1e-8).sqrt()
    corr = torch.bmm(centered, centered.transpose(1, 2)) / value.shape[-1] / (sd.unsqueeze(2) * sd.unsqueeze(1))
    indices = torch.triu_indices(12, 12, offset=1, device=value.device)
    pieces.append(corr[:, indices[0], indices[1]])
    return torch.cat(pieces, dim=1)


def moment_loss(real, fake, labels):
    reference, generated = moment_features(real).detach(), moment_features(fake)
    terms = []
    for c in range(3):
        mask = labels == c
        if mask.any():
            a, b = reference[mask], generated[mask]
            terms.append(F.smooth_l1_loss(b.mean(0), a.mean(0))
                + F.smooth_l1_loss(b.std(0, unbiased=False), a.std(0, unbiased=False)))
    return torch.stack(terms).mean()
