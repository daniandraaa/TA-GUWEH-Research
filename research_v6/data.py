"""Global scalar preserves inter-lead amplitudes; all reads restricted to inner IDs."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.data import load_pool


def load_development(study):
    X, table = load_pool(study.source, 'development')
    labels = {r['hospital_id']: r['label'] for r in study.records}
    if X.shape != (len(table), 12, 5000) or set(table.hospital_id) != set(labels):
        raise RuntimeError('Native development cache changed')
    if any(int(r.label) != labels[r.hospital_id] for r in table.itertuples()):
        raise RuntimeError('Segment labels differ from frozen anatomical labels')
    return X, table


def reference_rows(table, ids, maximum):
    rows = []
    for hid in sorted(ids):
        indices = np.flatnonzero(table.hospital_id.eq(hid).to_numpy())
        if not len(indices):
            raise ValueError(f'No windows for patient {hid}')
        chosen = np.unique(np.linspace(0, len(indices) - 1, min(maximum, len(indices))).round().astype(int))
        for index in indices[chosen]:
            rows.append({'hospital_id': hid, 'segment_index': int(index),
                'label': int(table.iloc[index].label), 'weight': 1.0 / len(chosen)})
    return pd.DataFrame(rows)


def global_scale(X, rows):
    rms = []
    for _, group in rows.groupby('hospital_id', sort=True):
        values = np.asarray(X[group.segment_index.to_numpy()], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite GAN fit reference signal')
        rms.append(float(np.sqrt(np.mean(values * values))))
    scale = float(np.median(rms))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid GAN fit-only amplitude scale')
    return scale


class BalancedPatients(Dataset):
    def __init__(self, X, table, ids, scale, draws, seed):
        self.X, self.scale, self.draws = X, float(scale), int(draws)
        self.groups = {c: [] for c in range(3)}
        for hid in sorted(ids):
            indices = np.flatnonzero(table.hospital_id.eq(hid).to_numpy())
            if not len(indices): raise ValueError('Empty fit patient')
            self.groups[int(table.iloc[indices[0]].label)].append((hid, indices))
        if any(not g for g in self.groups.values()): raise ValueError('Missing GAN class')
        self.n_patients = len(ids)
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n_patients * self.draws

    def __getitem__(self, index):
        c = int(torch.randint(3, (), generator=self.generator))
        group = self.groups[c]
        hid, indices = group[int(torch.randint(len(group), (), generator=self.generator))]
        row = int(indices[int(torch.randint(len(indices), (), generator=self.generator))])
        value = np.array(self.X[row], dtype=np.float32, copy=True) / self.scale
        if not np.isfinite(value).all(): raise RuntimeError('Nonfinite GAN training signal')
        return torch.from_numpy(value), c, hid
