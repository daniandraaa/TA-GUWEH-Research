"""Explicit native 2.5-s adapter; patient pooling uses every real window."""
import numpy as np
from scipy.interpolate import interp1d

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def adapt(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (12, 5000) or not np.isfinite(array).all():
        raise ValueError('Finite [windows,12,5000] native 2000-Hz signals required')
    # Same interpolation coordinates and standardization formula as ecg-transform 0.1.3.
    # This retains 2.5 s: it does not stretch/concatenate/pad into a 5-s recording.
    coordinates = np.linspace(0, 1249, 5000)
    out = interp1d(coordinates, array, kind='linear', axis=-1)(np.arange(1250))
    centered = out - out.mean(-1, keepdims=True)
    out = centered / (centered.std(-1, keepdims=True) + 1e-8)
    if not np.isfinite(out).all(): raise RuntimeError('Nonfinite adapted signal')
    return out.astype(np.float32)


def patient_indices(table, ids):
    groups = []
    for hid in sorted(ids):
        indices = np.flatnonzero(table.hospital_id.to_numpy() == hid)
        if not len(indices) or table.iloc[indices].label.nunique() != 1:
            raise RuntimeError('Missing/inconsistent patient windows')
        groups.append((hid, int(table.iloc[indices[0]].label), indices))
    return groups


def fit_feature_scale(features, train_indices):
    x = np.asarray(features[train_indices], dtype=np.float64)
    if not len(x) or not np.isfinite(x).all(): raise RuntimeError('Invalid training feature vectors')
    mean, std = x.mean(0), x.std(0)
    scale = np.where(std > 1e-8, std, 1.)
    return mean.astype(np.float32), scale.astype(np.float32)
