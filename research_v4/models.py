"""Compact hybrid hypotheses and probability MIL; not paper reproductions."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from research_v3.models import build_model as build_legacy, complete_patches


def position(length, dimension, device, dtype):
    times = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequencies = torch.exp(torch.arange(0, dimension, 2, device=device, dtype=dtype)
                            * (-math.log(10000.0) / dimension))
    result = torch.zeros(length, dimension, device=device, dtype=dtype)
    result[:, 0::2], result[:, 1::2] = torch.sin(times * frequencies), torch.cos(times * frequencies)
    return result


def transformer(dimension, heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(dimension, heads, dimension * 2, dropout,
                                      activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dimension),
                                 enable_nested_tensor=False)


class LegacyFeatures(nn.Module):
    """Expose the exact V3 pre-classifier features without changing its logits."""
    def __init__(self, name, config):
        super().__init__()
        self.name, self.base = name, build_legacy(name, config)
        self.dimension = config['cnn_channels'][-1]

    def forward(self, inputs):
        m = self.base
        features = m.cnn(inputs).permute(0, 2, 1)
        if self.name == 'encoder':
            pooled = m.transformer(m.pos_enc(features)).mean(1)
        elif self.name == 'patchtst':
            patches, weights = complete_patches(features, m.patch_size)
            tokens = m.transformer(m.pos_enc(m.patch_proj(patches)))
            pooled = (tokens * weights[None, :, None]).sum(1) / weights.sum()
        elif self.name == 'hierarchical':
            local = m.transformer_level1(m.pos_enc_1(features))
            down = m.downsample(local.permute(0, 2, 1)).permute(0, 2, 1)
            pooled = m.transformer_level2(m.pos_enc_2(down)).mean(1)
        else:
            raise ValueError(self.name)
        return pooled, m.classifier(pooled)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.shortcut = nn.Conv1d(incoming, outgoing, 1, stride, bias=False)
        self.main = nn.Sequential(nn.Conv1d(incoming, outgoing, 7, stride, padding=3, bias=False),
            nn.GroupNorm(8, outgoing), nn.GELU(),
            nn.Conv1d(outgoing, outgoing, 3, padding=1, bias=False), nn.GroupNorm(8, outgoing))

    def forward(self, x):
        return F.gelu(self.main(x) + self.shortcut(x))


class ResidualMultiscale(nn.Module):
    def __init__(self, config, classes):
        super().__init__()
        d = self.dimension = config['dimension']
        self.stem = nn.ModuleList([nn.Sequential(nn.Conv1d(12, 16, k, 4, k // 2, bias=False),
                                  nn.GroupNorm(4, 16), nn.GELU()) for k in (7, 15, 31)])
        self.cnn = nn.Sequential(ResidualBlock(48, d, 2), ResidualBlock(d, d, 2),
                                 ResidualBlock(d, d, 2))
        self.transformer = transformer(d, config['heads'], config['layers'], config['dropout'])
        self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Dropout(config['dropout']), nn.Linear(d, classes))

    def forward(self, inputs):
        tokens = self.cnn(torch.cat([branch(inputs) for branch in self.stem], 1)).transpose(1, 2)
        tokens = tokens + position(tokens.shape[1], self.dimension, tokens.device, tokens.dtype)
        pooled = self.transformer(tokens).mean(1)
        return pooled, self.classifier(pooled)


class LeadTime(nn.Module):
    """Shared lead CNN, temporal attention, then attention between lead identities."""
    def __init__(self, config, classes):
        super().__init__()
        d = self.dimension = config['dimension']
        self.bins = config['temporal_bins']
        self.cnn = nn.Sequential(nn.Conv1d(1, 16, 15, 4, 7, bias=False), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv1d(16, 32, 7, 4, 3, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, d, 5, 2, 2, bias=False), nn.GroupNorm(8, d), nn.GELU(),
            nn.AdaptiveAvgPool1d(self.bins))
        self.lead_embedding = nn.Parameter(torch.randn(1, 12, 1, d) * .02)
        self.temporal = transformer(d, config['heads'], 1, config['dropout'])
        self.spatial = transformer(d, config['heads'], 1, config['dropout'])
        self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Dropout(config['dropout']), nn.Linear(d, classes))

    def forward(self, inputs):
        b, leads, length = inputs.shape
        if leads != 12:
            raise ValueError('The fixed lead order requires 12 leads')
        z = self.cnn(inputs.reshape(b * leads, 1, length)).transpose(1, 2)
        z = z + position(self.bins, self.dimension, z.device, z.dtype)
        z = self.temporal(z).reshape(b, leads, self.bins, self.dimension) + self.lead_embedding
        z = self.spatial(z.permute(0, 2, 1, 3).reshape(b * self.bins, leads, self.dimension))
        pooled = z.reshape(b, self.bins, leads, self.dimension).mean((1, 2))
        return pooled, self.classifier(pooled)


class PatientModel(nn.Module):
    """Weights from embeddings combine segment probabilities, not ECG annotations."""
    def __init__(self, backbone, pooling='mean', classes=3):
        super().__init__()
        if pooling not in {'mean', 'attention'}:
            raise ValueError(pooling)
        self.backbone, self.pooling, self.classes = backbone, pooling, classes
        d = backbone.dimension
        self.gate = (nn.ModuleList([nn.Linear(d, 64), nn.Linear(d, 64), nn.Linear(64, 1)])
                     if pooling == 'attention' else None)

    def segment_outputs(self, inputs):
        features, logits = self.backbone(inputs)
        scores = (self.gate[2](torch.tanh(self.gate[0](features)) * torch.sigmoid(self.gate[1](features)))
                  .squeeze(-1) if self.gate is not None else torch.zeros(len(inputs), device=inputs.device))
        return F.log_softmax(logits, -1), scores

    def pool(self, log_probabilities, scores, mask):
        if not bool(mask.any(1).all()):
            raise ValueError('A patient must have at least one valid segment')
        scores = scores.masked_fill(~mask, -torch.inf)
        log_weights = F.log_softmax(scores, 1)
        result = torch.logsumexp(log_probabilities + log_weights[..., None], 1)
        return result, log_weights.exp()

    def forward(self, inputs, mask=None):
        if inputs.ndim == 3:
            return self.segment_outputs(inputs)[0]
        b, k, leads, length = inputs.shape
        if mask is None:
            mask = torch.ones(b, k, dtype=torch.bool, device=inputs.device)
        flat_mask = mask.reshape(-1)
        logp, scores = self.segment_outputs(inputs.reshape(b * k, leads, length)[flat_mask])
        full_logp = torch.zeros(b * k, self.classes, device=inputs.device, dtype=logp.dtype)
        full_scores = torch.zeros(b * k, device=inputs.device, dtype=scores.dtype)
        full_logp[flat_mask], full_scores[flat_mask] = logp, scores
        return self.pool(full_logp.reshape(b, k, self.classes), full_scores.reshape(b, k), mask)[0]

    @torch.no_grad()
    def all_segments(self, inputs, chunk=16):
        """Encode in chunks; one global pooling operation over ALL patient segments."""
        if len(inputs) == 0 or chunk < 1:
            raise ValueError('Nonempty segments and positive chunk required')
        outputs = [self.segment_outputs(block) for block in inputs.split(chunk)]
        logp, scores = [torch.cat([item[i] for item in outputs]) for i in (0, 1)]
        mask = torch.ones(1, len(inputs), device=inputs.device, dtype=torch.bool)
        result, weights = self.pool(logp[None], scores[None], mask)
        return result[0].exp(), weights[0]


def build_model(name, pooling, inherited_config, new_config):
    if name in {'encoder', 'patchtst', 'hierarchical'}:
        backbone = LegacyFeatures(name, inherited_config)
    elif name == 'residual_multiscale':
        backbone = ResidualMultiscale(new_config, inherited_config['n_classes'])
    elif name == 'lead_time':
        backbone = LeadTime(new_config, inherited_config['n_classes'])
    else:
        raise ValueError(name)
    return PatientModel(backbone, pooling, inherited_config['n_classes'])
