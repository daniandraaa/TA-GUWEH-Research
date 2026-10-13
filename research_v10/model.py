"""Unmasked ECG-FM forward using pinned official numerical modules."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import ast
import torch
from torch import nn
from torch.nn import functional as F
from ._official import ConvFeatureExtraction, ConvPositionalEncoding, LayerNorm, TransformerEncoder


def config(model_cfg):
    values = {'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072,
        'encoder_attention_heads': 12, 'layer_norm_first': False, 'saliency': False,
        'dropout': .1, 'attention_dropout': .1, 'activation_dropout': 0., 'encoder_layerdrop': 0.,
        'dropout_input': .1, 'dropout_features': .1, 'conv_feature_layers': '[(256, 2, 2)] * 4',
        'in_d': 12, 'extractor_mode': 'default', 'conv_bias': False, 'conv_pos': 128,
        'conv_pos_groups': 16, 'feature_grad_mult': 0., 'all_gather': False}
    values.update({k: v for k, v in model_cfg.items() if k in values})
    if values['extractor_mode'] != 'default' or values['in_d'] != 12:
        raise ValueError('Only the verified default 12-lead convolutional encoder is supported')
    # Fixed public definition; never evaluate arbitrary configuration expressions.
    if values['conv_feature_layers'] != '[(256, 2, 2)] * 4': raise ValueError('Unexpected convolutional configuration')
    return SimpleNamespace(**values)


class ECGFMEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.feature_grad_mult = 0.
        self.input_quantizer = None
        self.encoder = TransformerEncoder(cfg)
        self.feature_extractor = ConvFeatureExtraction([(256, 2, 2)] * 4, 12, 0., cfg.extractor_mode, cfg.conv_bias)
        self.layer_norm = LayerNorm(256)
        self.post_extract_proj = nn.Linear(256, cfg.encoder_embed_dim)
        self.conv_pos = ConvPositionalEncoding(cfg)

    def embeddings(self, values):
        x = self.layer_norm(self.feature_extractor(values).transpose(1, 2))
        x = self.dropout_input(self.post_extract_proj(x))
        return x + self.conv_pos(x)

    def forward(self, values):
        return self.encoder(self.embeddings(values))['x']

    def until_last(self, values):
        if self.training: raise RuntimeError('Upstream encoding must stay frozen/eval')
        x = self.embeddings(values)
        if not self.cfg.layer_norm_first: x = self.encoder.layer_norm(x)
        x = x.transpose(0, 1)
        for layer in self.encoder.layers[:-1]: x, _ = layer(x)
        return x.transpose(0, 1)

    def finish(self, tokens):
        x, _ = self.encoder.layers[-1](tokens.transpose(0, 1))
        x = x.transpose(0, 1)
        if self.cfg.layer_norm_first: x = self.encoder.layer_norm(x)
        return x


def canonical_encoder_state(state):
    result = {}; omitted = []
    allowed_unused = ('quantizer.', 'project_q.', 'final_proj.', 'target_glu.')
    for name, tensor in state.items():
        if name == 'mask_emb' or name.startswith(allowed_unused): omitted.append(name); continue
        if name.startswith(('input_quantizer.', 'project_inp.')): raise RuntimeError('Quantized encoder input is unsupported')
        target = name.replace('conv_pos.pos_conv.0.weight_g', 'conv_pos.pos_conv.0.parametrizations.weight.original0')
        target = target.replace('conv_pos.pos_conv.0.weight_v', 'conv_pos.pos_conv.0.parametrizations.weight.original1')
        if target in result or not torch.is_tensor(tensor): raise RuntimeError('Unexpected/non-tensor encoder state')
        result[target] = tensor
    return result, omitted


def load_encoder(path, cfg, random_seed=None):
    if random_seed is not None: torch.manual_seed(random_seed)
    model = ECGFMEncoder(config(cfg))
    if random_seed is None:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        raw = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state, unused = canonical_encoder_state(raw)
        model.load_state_dict(state, strict=True)
        del checkpoint, raw, state
    model.requires_grad_(False); model.eval()
    return model


class TailClassifier(nn.Module):
    def __init__(self, encoder, mean, scale):
        super().__init__(); self.cfg = encoder.cfg
        self.tail = deepcopy(encoder.encoder.layers[-1])
        self.tail.requires_grad_(True)
        self.finish_norm = deepcopy(encoder.encoder.layer_norm)
        self.finish_norm.requires_grad_(False)
        self.register_buffer('feature_mean', torch.as_tensor(mean).clone())
        self.register_buffer('feature_scale', torch.as_tensor(scale).clone())
        self.head = nn.Linear(encoder.cfg.encoder_embed_dim, 3)

    def patient_embedding(self, tokens, chunk=4):
        sums = []
        for block in tokens.split(chunk):
            x, _ = self.tail(block.transpose(0, 1)); x = x.transpose(0, 1)
            if self.cfg.layer_norm_first: x = self.finish_norm(x)
            sums.append(x.mean(1).sum(0))
        return torch.stack(sums).sum(0) / len(tokens)

    def forward(self, tokens):
        feature = self.patient_embedding(tokens)
        return self.head((feature - self.feature_mean) / self.feature_scale)


def official_unmasked_forward(model, values):
    """Run the author's unchanged forward body on shared official modules.

    Avoids a second gigabyte-scale model or altering the historical environment.
    Tests the port's assembly; not a claim of parity with every fairseq backend.
    """
    path = Path(__file__).parent / '_official/reference_wav2vec2.py'
    parsed = ast.parse(path.read_text())
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Wav2Vec2Model')
    method = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'))
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['forward'](model, values, padding_mask=None, mask=False, features_only=True)['x']
