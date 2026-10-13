"""Bag pasien V4 + augmentasi GAN: satu dari empat window diganti window sintetis kelas sama."""
import numpy as np
import pandas as pd
import torch
from research_v4.data import PatientBags, collate_bags
from src.data import load_pool as load_cache


def load_pool(study, pool='development'):
    if pool not in {'development', 'test'}:
        raise ValueError(pool)
    if pool == 'test':
        from .experiment import require_test_unlocked
        require_test_unlocked(study)
    X, table = load_cache(study.source, pool)
    expected = {r['hospital_id']: r['label'] for r in study.records if r['pool'] == pool}
    if X.shape != (len(table), 12, study.protocol['segment_length']) or set(table.hospital_id) != set(expected):
        raise RuntimeError('Cache sinyal/kohort berubah')
    if any(int(r.label) != expected[r.hospital_id] for r in table.itertuples()):
        raise RuntimeError('Label segmen berbeda dari label anatomi yang dibekukan')
    if not (table.valid_samples == study.protocol['segment_length']).all():
        raise RuntimeError('Window harus lengkap 2,5 detik')
    return X, table


def load_synthetic(folder):
    values = np.load(folder / 'synthetic.npy', mmap_mode='r')
    labels = pd.read_csv(folder / 'synthetic_labels.csv').label.to_numpy()
    if len(values) != len(labels) or set(labels.tolist()) != {0, 1, 2}:
        raise RuntimeError('Sampel sintetis tidak lengkap untuk tiga kelas')
    return values, labels


class GanBags(PatientBags):
    """Satu generator RNG (crop, slot, sampel) agar resume epoch V4 tetap eksak."""

    def __init__(self, X, table, ids, scaler, synthetic, synthetic_labels, size=4, sample_seed=None):
        super().__init__(X, table, ids, scaler, size, sample_seed)
        self.size, self.synthetic = size, synthetic
        self.groups = {c: np.flatnonzero(np.asarray(synthetic_labels) == c) for c in range(3)}
        if any(not len(g) for g in self.groups.values()):
            raise ValueError('Setiap kelas memerlukan window sintetis')
        self.replacements = 0

    def __getitem__(self, item):
        value, label, hid = super().__getitem__(item)
        # Bag pendek (<4 window) tidak diubah, sama seperti V8.
        if len(value) == self.size:
            slot = int(torch.randint(self.size, (), generator=self.generator))
            group = self.groups[label]
            index = int(group[int(torch.randint(len(group), (), generator=self.generator))])
            fake = (np.array(self.synthetic[index], dtype=np.float32, copy=True) - self.mean) / self.scale
            if not np.isfinite(fake).all():
                raise ValueError('Window sintetis tidak finite')
            value[slot] = torch.from_numpy(fake)
            self.replacements += 1
        return value, label, hid


__all__ = ['GanBags', 'PatientBags', 'collate_bags', 'load_pool', 'load_synthetic']
