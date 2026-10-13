"""Offline consensus QRS/R candidates; deliberately not a PVC classifier.

Derivative-energy detection is inspired by the QRS pipeline of Pan/Tompkins,
but these filters/thresholds are our own heuristic, not a faithful reproduction.
Agreement of two leads is technical consistency, not diagnostic validation.
"""
import numpy as np
from scipy import signal
from src.data import LEADS


def _separated(points, score, distance):
    accepted = []
    for p in sorted(set(map(int, points)), key=lambda q: (-float(score[q]), q)):
        if all(abs(p - q) >= distance for q in accepted):
            accepted.append(p)
    return np.array(sorted(accepted), dtype=int)


def lead_candidates(filtered, fs, spec):
    distance = max(1, round(fs * spec['refractory_seconds']))
    energy = np.gradient(filtered) ** 2
    width = max(1, round(fs * spec['energy_smooth_seconds']))
    energy = np.convolve(energy, np.ones(width) / width, mode='same')
    median = np.median(energy)
    mad = np.median(np.abs(energy - median))
    if energy.max() <= np.finfo(float).eps:
        return np.array([], dtype=int)
    threshold = median + spec['threshold_mad'] * max(mad, energy.max() * 1e-5)
    peaks, _ = signal.find_peaks(energy, height=threshold, distance=distance)
    radius = max(1, round(fs * spec['refinement_seconds']))
    refined = []
    for p in peaks:
        lo, hi = max(0, p - radius), min(len(filtered), p + radius + 1)
        refined.append(lo + int(np.argmax(np.abs(filtered[lo:hi]))))
    return _separated(refined, np.abs(filtered), distance)


def consensus_matches(a, b, tolerance):
    """One-to-one nearest pairing; a candidate cannot match multiple events."""
    a, b = np.sort(np.asarray(a, dtype=int)), np.sort(np.asarray(b, dtype=int))
    used = set()
    matches = []
    for x in a:
        lo = int(np.searchsorted(b, int(x) - tolerance, side='left'))
        hi = int(np.searchsorted(b, int(x) + tolerance, side='right'))
        available = [j for j in range(lo, hi) if j not in used]
        if available:
            index = min(available, key=lambda j: (abs(int(x) - int(b[j])), int(b[j])))
            used.add(index)
            matches.append((int(x), int(b[index])))
    return matches


def detect_candidates(raw, spec):
    raw = np.asarray(raw, dtype=float)
    fs = spec['fs']
    if raw.ndim != 2 or raw.shape[1] != 12 or not np.isfinite(raw).all():
        raise ValueError('Expected finite [samples,12] raw signal')
    if not 0 < spec['bandpass_hz'][0] < spec['bandpass_hz'][1] < fs / 2:
        raise ValueError('Invalid bandpass/sampling rate')
    sos = signal.butter(3, spec['bandpass_hz'], fs=fs, btype='bandpass', output='sos')
    filtered = signal.sosfiltfilt(sos, raw, axis=0)
    # Rank leads deterministically by robust bandpassed amplitude. The rank is
    # calculated per recording, without site labels or population-level fit.
    ordered = sorted(spec['leads'], key=lambda name: (-float(np.quantile(np.abs(filtered[:, LEADS.index(name)]), .95)), name))
    first, second = ordered[:2]
    a = lead_candidates(filtered[:, LEADS.index(first)], fs, spec)
    b = lead_candidates(filtered[:, LEADS.index(second)], fs, spec)
    pairs = consensus_matches(a, b, round(fs * spec['consensus_seconds']))
    # Retain the first lead's refined extremum; averaging two lead locations
    # need not place the center at a true extremum in either lead.
    centers = np.array([x for x, _ in pairs], dtype=int)
    before, after = round(fs * spec['before_seconds']), round(fs * spec['after_seconds'])
    valid = centers[(centers >= before) & (centers + after <= len(raw))]
    all_valid_count = len(valid)
    if len(valid) > spec['maximum_per_patient']:
        valid = valid[np.linspace(0, len(valid) - 1, spec['maximum_per_patient']).round().astype(int)]
    denominator = len(a) + len(b)
    qa = {'lead_a': first, 'lead_b': second, 'n_a': len(a), 'n_b': len(b),
          'n_consensus': len(pairs), 'n_complete_candidates': all_valid_count,
          'n_retained': len(valid), 'agreement': 2 * len(pairs) / denominator if denominator else 0.0,
          'candidate_rate_per_minute': len(pairs) * 60 * fs / len(raw),
          'pvc_verified': False, 'expert_reviewed': False}
    qa['engineering_review_flag'] = bool(qa['agreement'] < .7 or not len(valid)
                                         or not 30 <= qa['candidate_rate_per_minute'] <= 220)
    return valid, qa


def extract_windows(clean, centers, spec):
    before = round(spec['fs'] * spec['before_seconds'])
    after = round(spec['fs'] * spec['after_seconds'])
    length = before + after
    fallback = len(centers) == 0
    if fallback:
        # Keep the patient in the study even when detection fails. Never call
        # these fallback crops beats or remove patients based on detector QA.
        n = max(1, min(spec['maximum_per_patient'], len(clean) // length))
        starts = np.linspace(0, max(0, len(clean) - length), n).round().astype(int)
    else:
        starts = np.asarray(centers, dtype=int) - before
    blocks, rows = [], []
    for start in starts:
        crop = clean[start:start + length]
        block = np.zeros((length, 12), dtype=np.float32)
        block[:len(crop)] = crop
        blocks.append(block.T)
        rows.append({'start_sample': int(start), 'valid_samples': len(crop),
                     'r_candidate_sample': None if fallback else int(start + before),
                     'is_fallback': fallback, 'pvc_verified': False})
    return np.stack(blocks), rows
