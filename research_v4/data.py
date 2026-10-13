"""Patient bags and training-only scaler, reusing native 2000-Hz window cache."""
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from src.artifacts import canonical_hash, lock, read_json, sha256_file, write_json
from src.data import ECGDataset, load_pool as original_load_pool
from src.preprocessing import fit_training_scaler


def load_pool(run, source, cfg, pool='development'):
    if pool not in {'development', 'test'}:
        raise ValueError(pool)
    if pool == 'test':
        from .common import require_final_complete
        require_final_complete(run, cfg)
    X, segments = original_load_pool(source, pool)
    expected = {r['hospital_id']: r['label'] for r in read_json(run / 'data/patients.json') if r['pool'] == pool}
    if X.shape != (len(segments), 12, cfg['segment_length']) or set(segments.hospital_id) != set(expected):
        raise ValueError('Shared cache shape/cohort mismatch')
    if any(int(r.label) != expected[r.hospital_id] for r in segments.itertuples()):
        raise ValueError('Shared cache labels changed')
    if not (segments.valid_samples == cfg['segment_length']).all():
        raise ValueError('V4 native-window protocol requires complete 2.5-second windows')
    return X, segments


def scaler_for(run, scope, X, segments, indices, ids):
    actual = sorted(set(segments.iloc[indices].hospital_id))
    if actual != sorted(ids) or len(set(indices)) != len(indices):
        raise ValueError('Scaler training indices/patient provenance mismatch')
    directory = run / 'scalers' / scope
    spec = {'training_patient_ids': actual, 'indices_hash': canonical_hash(np.asarray(indices).tolist()),
            'n_segments': len(indices), 'development_cache': read_json(run / 'protocol.json')['development_cache']}
    with lock(directory / '.lock'):
        path, marker = directory / 'scaler.joblib', directory / 'provenance.json'
        if marker.exists():
            saved = read_json(marker)
            if saved['spec'] != spec or saved['sha256'] != sha256_file(path):
                raise RuntimeError('Scaler identity changed')
            return joblib.load(path)
        scaler = fit_training_scaler(X, indices)
        temporary = directory / 'scaler.tmp.joblib'
        joblib.dump(scaler, temporary)
        temporary.replace(path)
        write_json(marker, {'spec': spec, 'sha256': sha256_file(path),
                            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()})
        return scaler


class PatientBags(Dataset):
    def __init__(self, X, segments, ids, scaler, sample_size=None, sample_seed=None):
        self.X, self.sample_size = X, sample_size
        # Separate from dropout/initialization: matched arms get identical crops.
        self.generator = torch.Generator().manual_seed(torch.initial_seed() if sample_seed is None else sample_seed)
        self.ids = sorted(ids)
        self.mean = np.asarray(scaler.mean_, dtype=np.float32)[:, None]
        self.scale = np.asarray(scaler.scale_, dtype=np.float32)[:, None]
        self.indices, self.labels = [], []
        for hid in self.ids:
            rows = np.flatnonzero(segments.hospital_id.to_numpy() == hid)
            values = segments.iloc[rows].label.unique()
            if len(values) != 1:
                raise ValueError('Patient must have one label and nonempty segments')
            self.indices.append(rows)
            self.labels.append(int(values[0]))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        indices = self.indices[item]
        if self.sample_size is not None and len(indices) > self.sample_size:
            indices = indices[torch.randperm(len(indices), generator=self.generator)[:self.sample_size].numpy()]
        value = (np.array(self.X[indices], dtype=np.float32, copy=True) - self.mean) / self.scale
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite scaled waveform')
        return torch.from_numpy(value), self.labels[item], self.ids[item]


def collate_bags(items):
    maximum = max(len(x) for x, _, _ in items)
    values = torch.zeros(len(items), maximum, *items[0][0].shape[1:])
    mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for i, (x, _, _) in enumerate(items):
        values[i, :len(x)], mask[i, :len(x)] = x, True
    return values, torch.tensor([y for _, y, _ in items]), mask, [hid for _, _, hid in items]


class SegmentsWithIdentity(ECGDataset):
    def __init__(self, X, segments, indices, scaler):
        super().__init__(X, segments.label, indices, scaler)
        self.ids = segments.hospital_id.to_numpy()

    def __getitem__(self, item):
        x, y = super().__getitem__(item)
        return x, y, str(self.ids[self.indices[item]])
