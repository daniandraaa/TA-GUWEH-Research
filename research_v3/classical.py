"""Small-data ExtraTrees comparator; no anatomical coordinates are inferred."""
import itertools
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import ExtraTreesClassifier
from src.artifacts import canonical_hash, read_json, write_json
from .data import load_pool, representation
from .sampling import patient_weights, patient_class_weights


def scenarios(cfg):
    spec = cfg['classical']
    return [{'id': f'T{i:02}', 'max_features': float(mf), 'min_samples_leaf': int(leaf)}
            for i, (mf, leaf) in enumerate(itertools.product(spec['max_features'], spec['min_samples_leaf']), 1)]


def waveform_features(values, fs):
    """Features per lead; no target-dependent fitting, beat diagnosis, or HRV."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 3 or x.shape[1] != 12 or not np.isfinite(x).all():
        raise ValueError('Expected finite [segments,12,time]')
    freq, psd = welch(x, fs=fs, nperseg=min(512, x.shape[-1]), axis=-1)
    total = psd.sum(axis=-1) + 1e-12
    bands = [psd[..., (freq >= lo) & (freq < hi)].sum(axis=-1) / total
             for lo, hi in [(0, 5), (5, 15), (15, 40), (40, 100)]]
    parts = [x.mean(-1), x.std(-1), np.sqrt((x ** 2).mean(-1)),
             np.quantile(x, .05, axis=-1), np.quantile(x, .95, axis=-1),
             np.abs(x).max(-1), np.abs(np.diff(x, axis=-1)).mean(-1), *bands]
    return np.stack(parts, axis=-1).reshape(len(x), -1).astype('float32')


def features(run, source, arm, config, cfg, pool='development'):
    directory = run / 'features' / representation(arm) / pool
    path = directory / 'X.npy'
    identity = canonical_hash({'experiment': read_json(run / 'protocol.json')['experiment_hash'],
                               'version': cfg['classical']['feature_version'], 'pool': pool})
    if path.exists() and (directory / 'manifest.json').exists():
        if read_json(directory / 'manifest.json')['identity'] != identity:
            raise RuntimeError('Feature cache identity changed')
        return np.load(path, mmap_mode='r')
    X, _ = load_pool(run, source, arm, pool)
    directory.mkdir(parents=True, exist_ok=True)
    values = np.concatenate([waveform_features(X[i:i + 32], config['sampling_rate'])
                             for i in range(0, len(X), 32)])
    temporary = directory / 'X.tmp.npy'
    np.save(temporary, values)
    temporary.replace(path)
    write_json(directory / 'manifest.json', {'identity': identity, 'n_features': values.shape[1],
                'description': '11 descriptive waveform statistics/bandpower fractions per lead; no clinical interpretation'})
    return np.load(path, mmap_mode='r')


def build_tree(cfg, scenario, seed):
    return ExtraTreesClassifier(n_estimators=cfg['classical']['n_estimators'],
                               max_depth=cfg['classical']['max_depth'],
                               max_features=scenario['max_features'],
                               min_samples_leaf=scenario['min_samples_leaf'],
                               random_state=seed, n_jobs=cfg['classical']['n_jobs'])


def fit_tree(values, train_segments, cfg, scenario, seed):
    model = build_tree(cfg, scenario, seed)
    class_weight = patient_class_weights(train_segments, 3).numpy()
    weights = patient_weights(train_segments.hospital_id) * class_weight[train_segments.label.to_numpy()]
    model.fit(values, train_segments.label, sample_weight=weights)
    if not np.array_equal(model.classes_, np.arange(3)):
        raise ValueError('Training classes incomplete')
    return model
