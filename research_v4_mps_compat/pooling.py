"""Use the same adaptive bin boundaries as ATen without MPS adaptive pooling."""
import torch
from torch import nn


class ExactAdaptiveAvgPool1d(nn.Module):
    """Exact bin geometry; reductions may differ by floating-point roundoff.

    No padding, cropping, resampling, parameters, or buffers are introduced.
    Slice means remain on the input device and support autograd on CPU/MPS.
    """
    def __init__(self, output_size):
        super().__init__()
        if not isinstance(output_size, int) or output_size < 1:
            raise ValueError('A positive integer output size is required')
        self.output_size = output_size

    def forward(self, inputs):
        if inputs.ndim not in (2, 3) or inputs.shape[-1] < 1:
            raise ValueError('Expected [channels, time] or [batch, channels, time]')
        length, bins = inputs.shape[-1], self.output_size
        return torch.stack([
            inputs[..., (i * length) // bins:((i + 1) * length + bins - 1) // bins].mean(-1)
            for i in range(bins)
        ], dim=-1)

    def extra_repr(self):
        return f'output_size={self.output_size}'
