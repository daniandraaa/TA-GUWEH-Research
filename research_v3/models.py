"""V3 patch correction; all other architectures reuse frozen V2 definitions."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models import build_model as legacy_build_model
from src.models.patchtst import CNNPatchTST


def complete_patches(features, patch_size):
    """Pad the final partial patch and return valid-length pooling weights."""
    if features.ndim != 3 or features.shape[1] == 0 or patch_size < 1:
        raise ValueError('Expected nonempty [batch,time,features] and positive patch size')
    b, length, dimension = features.shape
    count = math.ceil(length / patch_size)
    padding = count * patch_size - length
    padded = F.pad(features, (0, 0, 0, padding)) if padding else features
    weights = torch.ones(count, dtype=features.dtype, device=features.device)
    weights[-1] = (patch_size - padding) / patch_size
    return padded.reshape(b, count, patch_size * dimension), weights


class CompleteCNNPatchTransformer(CNNPatchTST):
    """CNN + patch tokens: inspired by PatchTST, not its channel-independent model."""
    def forward(self, inputs):
        features = self.cnn(inputs).permute(0, 2, 1)
        patches, weights = complete_patches(features, self.patch_size)
        patches = self.transformer(self.pos_enc(self.patch_proj(patches)))
        pooled = (patches * weights[None, :, None]).sum(1) / weights.sum()
        return self.classifier(pooled)


def build_model(name, config):
    return CompleteCNNPatchTransformer(config) if name == 'patchtst' else legacy_build_model(name, config)
