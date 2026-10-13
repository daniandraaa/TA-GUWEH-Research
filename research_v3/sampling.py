"""Patient-uniform sampling in expectation, using training-fold data only."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def patient_weights(groups):
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups):
        raise ValueError('Expected nonempty patient IDs')
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def patient_class_weights(segments, n_classes):
    if segments.groupby('hospital_id').label.nunique().max() != 1:
        raise ValueError('A patient has inconsistent labels')
    patients = segments.drop_duplicates('hospital_id')
    counts = np.bincount(patients.label.to_numpy(), minlength=n_classes)
    if len(counts) != n_classes or np.any(counts == 0):
        raise ValueError('Every training class must be present')
    values = 1.0 / counts
    return torch.tensor(values / values.sum() * n_classes, dtype=torch.float32)


def training_loader(dataset, segments, batch_size, seed, balanced, num_samples=None):
    generator = torch.Generator().manual_seed(seed)
    weights = patient_weights(segments.hospital_id) if balanced else np.ones(len(dataset))
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset) if num_samples is None else num_samples,
                                    replacement=True, generator=generator)
    # Shared generator is essential: fit_classifier snapshots/restores it for
    # both DataLoader and WeightedRandomSampler at completed epoch boundaries.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        sampler=sampler, generator=generator, num_workers=0)
    return loader


def contribution_table(segments):
    table = segments.groupby(['hospital_id', 'label']).size().reset_index(name='n_segments')
    table['uniform_segment_probability'] = table.n_segments / table.n_segments.sum()
    table['patient_sampler_probability'] = 1.0 / len(table)
    return table
