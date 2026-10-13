"""Explicit 2000 Hz preprocessing; no resampling or automatic beat selection."""
import numpy as np
import pandas as pd
import pywt
from scipy import signal
from src.artifacts import ROOT, lock, read_json, sha256_file, write_json
from src.data import LEADS, segment_windows
from .common import disk_check

CLASS_SUBLOCATIONS = [
    {'LCC', 'RCC', 'AMC', 'LCC-RCC Ommisure'},
    {'LC', 'AC', 'RC'},
    {'PosteriorSeptal', 'AnteriorSeptal', 'FreeWall', 'RVOTOther'},
]


def relabel(records, names):
    out = []
    for r in records:
        row = dict(r)
        expected = next((i for i, subs in enumerate(CLASS_SUBLOCATIONS) if r['sublocation'] in subs), -1)
        if r['label'] != expected:
            raise ValueError('Membership changed: audit labels before training')
        row['class_name_historical'] = r['class_name']
        row['class_name'] = names[expected] if expected >= 0 else None
        out.append(row)
    return out


def sure_threshold(coeff):
    sigma = np.median(np.abs(coeff)) / .6745
    if len(coeff) == 0 or sigma <= np.finfo(float).eps:
        return 0.0
    a = np.sort(np.abs(coeff / sigma)) ** 2
    n = len(a)
    risk = (n - 2 * np.arange(1, n + 1) + np.cumsum(a) + np.arange(n - 1, -1, -1) * a) / n
    return float(np.sqrt(a[np.argmin(risk)]) * sigma)


def denoise(raw, fs, spec):
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12 or not np.isfinite(values).all():
        raise ValueError('Expected finite [time,12] signal')
    if fs <= 2 * spec['highpass_hz']:
        raise ValueError('Invalid sampling rate/cutoff')
    wavelet = pywt.Wavelet(spec['wavelet'])
    if pywt.dwt_max_level(len(values), wavelet.dec_len) < spec['level']:
        raise ValueError('Recording too short for declared wavelet level')
    sos = signal.butter(spec['highpass_order'], spec['highpass_hz'], btype='highpass', fs=fs, output='sos')
    filtered = signal.sosfiltfilt(sos, values, axis=0)
    out = np.empty_like(values)
    for lead in range(12):
        coeffs = pywt.wavedec(filtered[:, lead], wavelet, level=spec['level'])
        clean = [coeffs[0]]
        for coeff in coeffs[1:]:
            threshold = sure_threshold(coeff)
            clean.append(coeff if threshold == 0 else pywt.threshold(coeff, threshold, mode='soft'))
        out[:, lead] = pywt.waverec(clean, wavelet)[:len(values)]
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite output')
    return out.astype(np.float32)


def require_final_complete(run, cfg, config):
    if not (run / 'selection.json').exists():
        raise RuntimeError('Seleksi CV belum lengkap; test terkunci.')
    missing = [f'{model}/seed_{seed}' for model in config['models'] for seed in cfg['final_seeds']
               if not (run / 'final' / model / f'seed_{seed}' / 'metrics.json').exists()]
    if missing:
        raise RuntimeError('Test terkunci; selesaikan semua final: ' + ', '.join(missing))


def prepare_pool(run, source, cfg, config, pool='development'):
    if pool not in {'development', 'test'}:
        raise ValueError(pool)
    if pool == 'test':
        require_final_complete(run, cfg, config)
    if pool == 'development':
        original = read_json(source / 'data/patients.json')
        dev = {r['hospital_id'] for r in original if r['pool'] == 'development'}
        folds = read_json(source / 'folds/manifest.json')
        if sorted(h for f in folds for h in f['validation_ids']) != sorted(dev):
            raise ValueError('Invalid validation coverage')
        for f in folds:
            a, b = set(f['train_ids']), set(f['validation_ids'])
            if a & b or a | b != dev:
                raise ValueError('Invalid fold membership')
        write_json(run / 'folds/manifest.json', folds)
    directory = run / 'data' / pool
    with lock(directory / '.lock'):
        if (directory / 'manifest.json').exists():
            if not (directory / 'X.npy').exists() or not (directory / 'segments.csv').exists():
                raise RuntimeError('Incomplete pool')
            return directory
        records = relabel(read_json(source / 'data/patients.json'), config['class_names'])
        write_json(run / 'data/patients.json', records)
        patients = [r for r in records if r['pool'] == pool]
        rows, hashes = [], {}
        length, overlap = cfg['segment_length'], cfg['overlap']
        for r in patients:
            path = ROOT / cfg['preprocessing']['input'] / f"{r['hospital_id']}.csv"
            n = len(pd.read_csv(path, usecols=['II']))
            if n <= 0:
                raise ValueError(f'Empty recording: {path.name}')
            hashes[path.name] = sha256_file(path)
            starts = range(0, n - length + 1, int(length * (1 - overlap))) if n >= length else [0]
            rows.extend({'hospital_id': r['hospital_id'], 'label': r['label'], 'start_sample': s,
                         'valid_samples': min(n, length)} for s in starts)
        needed = len(rows) * 12 * length * 4
        disk_check(run, cfg, needed)
        tmp = directory / 'X.tmp.npy'
        X = np.lib.format.open_memmap(tmp, mode='w+', dtype='float32', shape=(len(rows), 12, length))
        offset = 0
        for index, r in enumerate(patients, 1):
            path = ROOT / cfg['preprocessing']['input'] / f"{r['hospital_id']}.csv"
            if sha256_file(path) != hashes[path.name]:
                raise RuntimeError('Input changed during preprocessing')
            raw = pd.read_csv(path, usecols=LEADS)[LEADS].to_numpy()
            clean = denoise(raw, cfg['sampling_rate'], cfg['preprocessing'])
            for _, _, block in segment_windows(clean, length, overlap):
                X[offset] = block
                offset += 1
            print(f'PREPROCESS {pool} {index}/{len(patients)}', flush=True)
        X.flush()
        del X
        if offset != len(rows):
            raise RuntimeError('Segment count mismatch')
        tmp.replace(directory / 'X.npy')
        pd.DataFrame(rows).to_csv(directory / 'segments.csv', index=False)
        write_json(directory / 'manifest.json', {'pool': pool, 'patients': patients, 'signal_sha256': hashes,
                   'sampling_rate': cfg['sampling_rate'], 'preprocessing': cfg['preprocessing'],
                   'n_patients': len(patients), 'n_segments': len(rows)})
    return directory
