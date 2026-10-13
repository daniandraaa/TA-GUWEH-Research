"""Numerical quality screening and nearest-neighbour diagnostics, not ECG diagnosis."""
from collections import defaultdict
import hashlib
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import wasserstein_distance
from src.data import LEADS

BANDS = ((.5, 5), (5, 15), (15, 40), (40, 100), (100, 250), (250, 500), (500, 1000))


def features(values, fs=2000):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (12, 5000) or not np.isfinite(values).all():
        raise ValueError('Quality features require finite native 12x5000 windows')
    result, families = {}, {}
    rms = np.sqrt(np.mean(values * values, axis=-1))
    centered = values - values.mean(-1, keepdims=True)
    sd = np.sqrt(np.mean(centered * centered, axis=-1))
    def add(family, name, array):
        result[name], families[name] = array, family
    for lead, name in enumerate(LEADS):
        add('amplitude', f'rms_{name}', rms[:, lead])
        add('amplitude', f'range90_{name}', np.quantile(values[:, lead], .95, axis=-1) - np.quantile(values[:, lead], .05, axis=-1))
        for lag in (1, 4, 16, 64):
            d = np.sqrt(np.mean((values[:, lead, lag:] - values[:, lead, :-lag]) ** 2, axis=-1))
            add('roughness', f'lag{lag}_over_rms_{name}', d / np.maximum(rms[:, lead], 1e-12))
    frequency, psd = welch(values, fs=fs, nperseg=2000, noverlap=1000, axis=-1)
    total = np.maximum(psd[..., frequency >= .5].sum(-1), 1e-20)
    for lo, hi in BANDS:
        band = (frequency >= lo) & ((frequency <= hi) if hi == fs / 2 else (frequency < hi))
        relative = psd[..., band].sum(-1) / total
        for lead, name in enumerate(LEADS):
            add('spectrum', f'power_{lo:g}_{hi:g}_{name}', relative[:, lead])
    corr = np.einsum('nlt,nkt->nlk', centered, centered) / values.shape[-1]
    corr /= np.maximum(sd[:, :, None] * sd[:, None, :], 1e-20)
    for a in range(12):
        for b in range(a + 1, 12):
            add('correlation', f'corr_{LEADS[a]}_{LEADS[b]}', corr[:, a, b])
    limb_den = np.maximum(np.sqrt(np.mean(centered[:, :6] ** 2, axis=(1, 2))), 1e-12)
    relations = (centered[:, 1] - centered[:, 0] - centered[:, 2],
        centered[:, 3] + (centered[:, 0] + centered[:, 1]) / 2,
        centered[:, 4] - centered[:, 0] + centered[:, 1] / 2,
        centered[:, 5] - centered[:, 1] + centered[:, 0] / 2)
    for name, value in zip(('III', 'aVR', 'aVL', 'aVF'), relations):
        add('limb_residual', f'residual_{name}', np.sqrt(np.mean(value * value, axis=1)) / limb_den)
    return pd.DataFrame(result), families


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    positions = (np.cumsum(weights) - .5 * weights) / weights.sum()
    return float(np.interp(q, positions, values))


def compare_features(fit, reference, fake, families, settings):
    rows, failed = [], []
    for c in range(3):
        a, b, s = (x[x.label.eq(c)] for x in (fit, reference, fake))
        if any(x.empty for x in (a, b, s)):
            raise ValueError('Fit/reference/synthetic must contain all classes')
        for name, family in families.items():
            av, bv, sv = (x[name].to_numpy() for x in (a, b, s))
            aw, bw, sw = (x.weight.to_numpy() for x in (a, b, s))
            bv_iqr = weighted_quantile(bv, bw, .75) - weighted_quantile(bv, bw, .25)
            denominator = max(bv_iqr, abs(weighted_quantile(bv, bw, .5)) * .05, 1e-4)
            baseline = wasserstein_distance(av, bv, aw, bw) / denominator
            candidate = wasserstein_distance(sv, bv, sw, bw) / denominator
            threshold = max(settings['normalized_wasserstein_floor'], settings['real_reference_distance_multiplier'] * baseline)
            if family == 'limb_residual':
                # Algebraically correct synthetic leads can be better than a separately denoised real cache.
                threshold = max(settings['limb_residual_upper_floor'],
                    settings['limb_residual_reference_multiplier'] * weighted_quantile(bv, bw, .95))
                statistic = weighted_quantile(sv, sw, .95)
                passed = statistic <= threshold
            else:
                statistic = candidate
                passed = candidate <= threshold
            rows.append({'label': c, 'family': family, 'feature': name,
                'real_fit_reference_distance': baseline, 'synthetic_reference_distance': candidate,
                'gate_statistic': statistic, 'threshold': threshold, 'passed': bool(passed)})
        for family in sorted(set(families.values())):
            values = [r for r in rows if r['label'] == c and r['family'] == family]
            fraction = sum(not r['passed'] for r in values) / len(values)
            if fraction > settings['max_failed_feature_fraction_per_family_and_class']:
                failed.append({'label': c, 'family': family, 'failed_fraction': fraction})
    return pd.DataFrame(rows), failed


def hashes(values):
    return [hashlib.sha256(np.ascontiguousarray(v, dtype='<f4').tobytes()).hexdigest() for v in values]


def native_nearest(values, X, table, fit_ids, scale, labels, batch=8):
    """Exact zero-lag full native 12x5000 nearest RMSE within each anatomical class.

    No phase alignment. RMSE is divided by nearest real-window RMS. This is a
    copy diagnostic, not a membership-inference or privacy guarantee.
    """
    output = []
    for c in range(3):
        selected = np.flatnonzero(np.asarray(labels) == c)
        indices = np.flatnonzero((table.hospital_id.isin(fit_ids) & table.label.eq(c)).to_numpy())
        if not len(indices): raise ValueError('Missing fit class')
        for start in range(0, len(selected), batch):
            slots = selected[start:start + batch]
            query = np.asarray(values[slots], dtype=np.float64).reshape(len(slots), -1)
            q_norm = np.einsum('ij,ij->i', query, query)
            best = np.full(len(slots), np.inf)
            matched = np.zeros(len(slots), dtype=int)
            matched_norm = np.zeros(len(slots))
            for offset in range(0, len(indices), 32):
                target_indices = indices[offset:offset + 32]
                targets = (np.asarray(X[target_indices], dtype=np.float64) / scale).reshape(len(target_indices), -1)
                norm = np.einsum('ij,ij->i', targets, targets)
                squared = np.maximum(q_norm[:, None] + norm[None, :] - 2 * query @ targets.T, 0)
                local = squared.argmin(1)
                distances = squared[np.arange(len(slots)), local]
                changed = distances < best
                best[changed] = distances[changed]
                matched[changed] = target_indices[local[changed]]
                matched_norm[changed] = norm[local[changed]]
            for n, slot in enumerate(slots):
                output.append({'sample_index': int(slot), 'label': c,
                    'nearest_segment_index': int(matched[n]),
                    'nearest_hospital_id': str(table.iloc[matched[n]].hospital_id),
                    'native_zero_lag_rmse': float(np.sqrt(best[n] / query.shape[1])),
                    'relative_rmse': float(np.sqrt(best[n] / max(matched_norm[n], 1e-20)))})
    return pd.DataFrame(output).sort_values('sample_index').reset_index(drop=True)


def numerical_screen(values, labels, X, table, fit_ids, scale, settings):
    finite = bool(np.isfinite(values).all())
    errors, classes = [], []
    if not finite:
        return {'passed': False, 'finite': False, 'errors': ['nonfinite_generated_values']}, None
    synthetic_hashes = hashes(values)
    generated_hashes = set(synthetic_hashes)
    exact_fit_matches = set()
    fit_indices = np.flatnonzero(table.hospital_id.isin(fit_ids).to_numpy())
    for offset in range(0, len(fit_indices), 32):
        # Compare in native stored units: matches are literal copies, not rounded correlations.
        exact_fit_matches.update(generated_hashes & set(hashes(np.asarray(X[fit_indices[offset:offset + 32]], dtype=np.float32))))
    # 'values' are native-unit synthetic; nearest distances use normalized native values.
    neighbours = native_nearest(values / scale, X, table, fit_ids, scale, labels)
    if exact_fit_matches: errors.append('exact_training_window_copy')
    if neighbours.relative_rmse.le(settings['near_copy_relative_rmse']).any(): errors.append('near_training_window_copy')
    for c in range(3):
        mask = np.asarray(labels) == c
        ids = np.flatnonzero(mask)
        distinct = len({synthetic_hashes[i] for i in ids}) / max(len(ids), 1)
        std = np.std(values[mask], axis=-1)
        constant = bool((std <= 1e-8).any())
        array = np.asarray(values[mask], dtype=np.float64).reshape(len(ids), -1)
        norm = np.einsum('ij,ij->i', array, array)
        distance = np.maximum(norm[:, None] + norm[None, :] - 2 * array @ array.T, 0)
        np.fill_diagonal(distance, np.inf)
        close = float((np.sqrt(distance.min(1) / np.maximum(norm, 1e-20)) <= settings['near_copy_relative_rmse']).mean())
        classes.append({'label': c, 'distinct_fraction': distinct, 'near_duplicate_fraction': close, 'has_constant_lead': constant})
        if distinct < settings['minimum_distinct_sample_fraction']: errors.append(f'class_{c}_duplicates')
        if constant: errors.append(f'class_{c}_constant_lead')
        if close > 1 - settings['minimum_distinct_sample_fraction']: errors.append(f'class_{c}_near_duplicate_samples')
    return {'passed': not errors, 'finite': finite, 'classes': classes,
        'exact_copy_hash_count': len(exact_fit_matches), 'errors': errors,
        'nearest_metric': 'same_class_full_native_zero_lag_rmse; not phase aligned; not privacy certification'}, neighbours


def collect_features(X, rows, scale):
    frames, mapping = [], None
    for offset in range(0, len(rows), 16):
        part = rows.iloc[offset:offset + 16].reset_index(drop=True)
        value, mapping = features(np.asarray(X[part.segment_index.to_numpy()], dtype=np.float64) / scale)
        frames.append(pd.concat((part, value), axis=1))
    return pd.concat(frames, ignore_index=True), mapping
