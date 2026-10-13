"""Unpaired class-distribution objective; no waveform pairs or clinical labels added.

FFT/statistics are explicitly computed on CPU with autograd intact, including
when the generator is on MPS. This is an engineered spectral-distribution loss,
not a reproduction of paired multi-resolution STFT losses used in vocoders.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from research_v6.quality import BANDS, weighted_quantile

FAMILIES = ('amplitude', 'roughness', 'correlation', 'spectrum')


def empirical_quantiles(values, levels):
    ordered = values.sort(dim=0).values
    # Same midpoint-CDF definition as the patient-weighted fit target bank.
    position = (levels.to(values) * len(values) - .5).clamp(0, len(values) - 1)
    low, high = position.floor().long(), position.ceil().long()
    fraction = (position - low).unsqueeze(1)
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def relative_welch(value, size, fs=2000):
    if value.device.type != 'cpu' or size > value.shape[-1] or size % 2:
        raise ValueError('Spectral statistics require explicit CPU and even supported windows')
    segments = value.unfold(-1, size, size // 2)
    segments = segments - segments.mean(-1, keepdim=True)
    window = torch.hann_window(size, dtype=value.dtype, device='cpu')
    spectrum = torch.fft.rfft(segments * window, dim=-1)
    power = (spectrum.real.square() + spectrum.imag.square()).mean(-2)
    factor = torch.ones(power.shape[-1], dtype=value.dtype); factor[1:-1] = 2
    power = power * factor
    frequency = torch.fft.rfftfreq(size, 1 / fs)
    denominator = power[..., frequency >= .5].sum(-1).clamp_min(1e-20)
    bands = []
    for lo, hi in BANDS:
        mask = (frequency >= lo) & ((frequency <= hi) if hi == fs / 2 else (frequency < hi))
        bands.append(power[..., mask].sum(-1) / denominator)
    return torch.stack(bands, dim=-1)


def statistics(values, settings):
    # .to(cpu) retains the differentiation path; never detach generated signals.
    value = values.to('cpu')
    if value.ndim != 3 or value.shape[1] != 12 or not torch.isfinite(value).all():
        raise ValueError('Fidelity statistics require finite twelve-lead waveforms')
    rms = (value.square().mean(-1) + 1e-8).sqrt()
    ordered = value.sort(dim=-1).values
    def percentile(q):
        position = q * (value.shape[-1] - 1)
        low = int(position); high = min(low + 1, value.shape[-1] - 1)
        return ordered[..., low] * (1 - (position - low)) + ordered[..., high] * (position - low)
    span = (percentile(.95) - percentile(.05)).clamp_min(1e-6)
    output = {'amplitude': torch.cat((rms.log(), span.log()), dim=1)}
    roughness = []
    for lag in (1, 4, 16, 64):
        if lag >= value.shape[-1]: raise ValueError('Need more than64 samples')
        delta = ((value[..., lag:] - value[..., :-lag]).square().mean(-1) + 1e-8).sqrt()
        roughness.append((delta / rms).clamp_min(1e-6).log())
    output['roughness'] = torch.cat(roughness, dim=1)
    centered = value - value.mean(-1, keepdim=True)
    sd = (centered.square().mean(-1) + 1e-8).sqrt()
    corr = torch.bmm(centered, centered.transpose(1, 2)) / value.shape[-1] / (sd.unsqueeze(2) * sd.unsqueeze(1))
    indices = torch.triu_indices(12, 12, offset=1)
    output['correlation'] = corr[:, indices[0], indices[1]]
    spectral = [(relative_welch(value, size) + settings['power_floor']).log().flatten(1)
        for size in settings['spectral_window_samples']]
    output['spectrum'] = torch.cat(spectral, dim=1)
    return output


@torch.no_grad()
def build_bank(X, rows, scale, settings):
    arrays = {key: [] for key in FAMILIES}
    for offset in range(0, len(rows), 16):
        values = np.array(X[rows.segment_index.iloc[offset:offset + 16].to_numpy()], dtype=np.float32, copy=True) / scale
        output = statistics(torch.from_numpy(values), settings)
        for key in FAMILIES: arrays[key].append(output[key].numpy())
    combined = {key: np.concatenate(values) for key, values in arrays.items()}
    bank = {}
    for key, values in combined.items():
        quantiles, scales = [], []
        for c in range(3):
            mask = rows.label.to_numpy() == c
            if not mask.any(): raise ValueError('Fit target bank must contain all classes')
            weights = rows.weight.to_numpy()[mask]
            a = values[mask]
            q = np.array([[weighted_quantile(a[:, i], weights, level) for i in range(a.shape[1])]
                for level in settings['quantiles']], dtype=np.float32)
            iqr = np.array([weighted_quantile(a[:, i], weights, .75) - weighted_quantile(a[:, i], weights, .25)
                for i in range(a.shape[1])], dtype=np.float32)
            quantiles.append(q); scales.append(np.maximum(iqr, settings['iqr_floor'][key]))
        bank[f'{key}_quantiles'] = np.stack(quantiles)
        bank[f'{key}_scale'] = np.stack(scales)
    return bank


class FidelityLoss(nn.Module):
    def __init__(self, bank, settings):
        super().__init__()
        self.settings = settings
        self.register_buffer('levels', torch.tensor(settings['quantiles'], dtype=torch.float32))
        for key in FAMILIES:
            for suffix in ('quantiles', 'scale'):
                name = f'{key}_{suffix}'
                self.register_buffer(name, torch.as_tensor(bank[name]).float().clone())
        for key in FAMILIES:
            target, scale = getattr(self, f'{key}_quantiles'), getattr(self, f'{key}_scale')
            if (target.ndim != 3 or target.shape[:2] != (3, len(self.levels))
                    or scale.shape != (3, target.shape[2]) or not torch.isfinite(target).all()
                    or not torch.isfinite(scale).all() or not (scale > 0).all()):
                raise ValueError('Finite three-class fit targets and positive feature scales required')

    def forward(self, fake, labels):
        if self.levels.device.type != 'cpu': raise ValueError('Keep fidelity loss and target bank explicitly on CPU')
        features = statistics(fake, self.settings)
        labels = labels.to('cpu')
        losses = {}
        for key in FAMILIES:
            terms = []
            for c in range(3):
                selected = features[key][labels == c]
                if not len(selected): continue
                candidate = empirical_quantiles(selected, self.levels)
                target = getattr(self, f'{key}_quantiles')[c]
                scale = getattr(self, f'{key}_scale')[c]
                terms.append(F.smooth_l1_loss((candidate - target) / scale, torch.zeros_like(candidate)))
            if not terms: raise ValueError('Empty minibatch')
            losses[key] = torch.stack(terms).mean()
        total = sum(self.settings['weights'][key] * losses[key] for key in FAMILIES)
        return total.to(fake.device), losses
