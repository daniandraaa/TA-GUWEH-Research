"""Development-only signal/window audit; no training, relabeling or test access.

Manual stages: prepare, pilot, scan, review. QRS candidates are unverified.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import numpy as np
import pandas as pd
import yaml
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from src.data import LEADS
from research_v4.common import DEFAULT_RUN, context
from research_v3.candidates import detect_candidates


def matched_controls(triage, counts):
    """Descriptive review pairs, not a causal or independent-patient experiment."""
    hard = triage[triage.n_wrong_selected_models >= 4].sort_values(
        ['n_wrong_selected_models', 'hospital_id'], ascending=[False, True])
    easy = triage[triage.n_wrong_selected_models <= 1]
    used, rows = set(), []
    for patient in hard.itertuples():
        pool = easy[easy.class_name == patient.class_name].copy()
        if pool.empty:
            rows.append({'hard_id': patient.hospital_id, 'control_id': None,
                         'matching': 'no_easy_same_class', 'reused_control': False})
            continue
        subtype = pool[pool.sublocation == patient.sublocation]
        matching = 'same_class_and_sublocation' if len(subtype) else 'same_class_only'
        if len(subtype): pool = subtype.copy()
        unused = pool[~pool.hospital_id.isin(used)]
        if len(unused): pool = unused.copy()
        pool['window_distance'] = pool.hospital_id.map(counts).map(
            lambda n: abs(np.log1p(n) - np.log1p(counts[patient.hospital_id])))
        selected = pool.sort_values(['window_distance', 'hospital_id']).iloc[0].hospital_id
        rows.append({'hard_id': patient.hospital_id, 'control_id': selected,
                     'matching': matching, 'reused_control': selected in used})
        used.add(selected)
    return pd.DataFrame(rows, columns=['hard_id', 'control_id', 'matching', 'reused_control'])


def audit_context(run_id=DEFAULT_RUN):
    run, source, _, cfg, _ = context(run_id)
    triage_path = run / 'analysis_20261003_complete/development_oof_error_triage.csv'
    if not triage_path.is_file():
        raise RuntimeError('Run python -m analysis_v4.audit_completed before the input audit')
    triage = pd.read_csv(triage_path, dtype={'hospital_id': str})
    expected = {str(r['hospital_id']): r['label'] for r in read_json(run / 'data/patients.json') if r['pool'] == 'development'}
    if (not triage.hospital_id.is_unique or set(triage.hospital_id) != set(expected)
            or not (triage.pool == 'development').all()
            or any(expected[r.hospital_id] != r.label for r in triage.itertuples())):
        raise ValueError('Audit must contain only the complete development cohort')
    segments_path = source / 'data/development/segments.csv'
    segments = pd.read_csv(segments_path, dtype={'hospital_id': str})
    if set(segments.hospital_id) != set(expected) or any(expected[r.hospital_id] != r.label for r in segments.itertuples()):
        raise ValueError('Development cache labels changed')
    if not (segments.valid_samples == cfg['segment_length']).all():
        raise ValueError('Expected complete native windows')
    spec = yaml.safe_load((ROOT / 'research_v3/protocol.yaml').read_text())['candidate']
    identity = {'base_experiment_hash': read_json(run / 'protocol.json')['experiment_hash'],
        'triage_sha256': sha256_file(triage_path), 'development_cache': read_json(run / 'protocol.json')['development_cache'],
        'audit_code_sha256': sha256_file(Path(__file__)),
        'detector_code_sha256': sha256_file(ROOT / 'research_v3/candidates.py'),
        'detector_spec': spec, 'detector_retention': 'all_complete_consensus_candidates; no_64_crop_cap',
        'sampling_rate_hz': cfg['sampling_rate'], 'pool': 'development',
        'selection': 'all development; review priority >=4 wrong models; controls <=1 wrong, same class',
        'clinical_PVC_annotations_created': False, 'class_labels_changed': False}
    out = run / 'development_input_audit_20261003'
    return out, source, triage, segments, cfg, spec, identity


def prepare(run_id=DEFAULT_RUN):
    out, _, triage, segments, _, _, identity = audit_context(run_id)
    with lock(out / '.lock'):
        path = out / 'manifest.json'; digest = canonical_hash(identity)
        if path.exists():
            existing = read_json(path)
            if existing['identity'] != identity or existing['audit_hash'] != digest:
                raise RuntimeError('Input audit source changed; preserve the existing audit and use a new version')
            for name, expected_hash in existing['review_file_sha256'].items():
                if sha256_file(out / name) != expected_hash:
                    raise RuntimeError('Review inventory changed; do not insert or remove patients')
        else:
            counts = segments.groupby('hospital_id').size().to_dict()
            pairs = matched_controls(triage, counts)
            pairs.to_csv(out / 'hard_control_pairs.csv', index=False)
            priority = []
            for pair in pairs.itertuples():
                priority += [pair.hard_id] + ([] if pd.isna(pair.control_id) else [pair.control_id])
            order = list(dict.fromkeys(priority + sorted(triage.hospital_id)))
            inventory = triage.set_index('hospital_id').loc[order].reset_index()
            inventory['n_windows'] = inventory.hospital_id.map(counts)
            inventory['review_group'] = np.where(inventory.n_wrong_selected_models >= 4, 'hard',
                                       np.where(inventory.n_wrong_selected_models <= 1, 'easy', 'intermediate'))
            inventory.to_csv(out / 'development_review_inventory.csv', index=False)
            write_json(path, {'identity': identity, 'audit_hash': digest,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'expected_patients': len(inventory),
                'hard_patients': len(pairs), 'unique_controls': pairs.control_id.dropna().nunique(),
                'reused_control_pairs': int(pairs.reused_control.sum()),
                'review_file_sha256': {name: sha256_file(out / name) for name in
                    ('hard_control_pairs.csv', 'development_review_inventory.csv')},
                'control_matching_is_descriptive_not_causal': True})
    print(f'Audit prepared: {out}', flush=True)
    return out


def reconstruct_windows(blocks, rows):
    """Count each covered sample once; verify overlapping cached windows agree."""
    length = int((rows.start_sample + rows.valid_samples).max())
    clean, covered = np.zeros((length, 12), dtype=np.float32), np.zeros(length, dtype=bool)
    for block, row in zip(blocks, rows.itertuples()):
        start, stop = int(row.start_sample), int(row.start_sample + row.valid_samples)
        if start < 0 or stop <= start: raise ValueError('Invalid window coordinates')
        value = np.asarray(block[:, :row.valid_samples]).T
        prior = covered[start:stop]
        if prior.any() and not np.array_equal(clean[start:stop][prior], value[prior]):
            raise ValueError('Overlapping windows disagree; investigate cache provenance')
        clean[start:stop] = value; covered[start:stop] = True
    if not covered.all(): raise ValueError('Native window cache has a coverage gap')
    return clean


def signal_profile(values, prefix):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 12 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite 12-lead signal')
    amplitude = np.quantile(values, .95, axis=0) - np.quantile(values, .05, axis=0)
    rms = np.sqrt(np.mean((values - values.mean(0)) ** 2, axis=0))
    derivative = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=0))
    rows = [{f'{prefix}_p95_minus_p05_{lead}': float(amplitude[i]),
             f'{prefix}_rms_{lead}': float(rms[i]),
             f'{prefix}_derivative_rms_{lead}': float(derivative[i])} for i, lead in enumerate(LEADS)]
    duplicate_pairs = [f'{LEADS[i]}={LEADS[j]}' for i in range(12) for j in range(i + 1, 12)
                       if np.array_equal(values[:, i], values[:, j])]
    return {k: v for row in rows for k, v in row.items()} | {
        f'{prefix}_numerically_constant_leads': int((rms <= 1e-8).sum()),
        f'{prefix}_exact_duplicate_lead_pairs': ','.join(duplicate_pairs)}


def window_candidate_counts(centers, starts, length):
    centers, starts = np.asarray(centers), np.asarray(starts)
    return np.searchsorted(centers, starts + length, side='left') - np.searchsorted(centers, starts, side='left')


def scan(run_id=DEFAULT_RUN, patients=None):
    out = prepare(run_id)
    _, source, _, segments, cfg, spec, identity = audit_context(run_id)
    with lock(out / '.lock'):
        inventory = pd.read_csv(out / 'development_review_inventory.csv', dtype={'hospital_id': str})
        if patients is not None and (not isinstance(patients, int) or patients < 1):
            raise ValueError('patients must be a positive integer or None for all')
        selected = inventory if patients is None else inventory.iloc[:patients]
        X = np.load(source / 'data/development/X.npy', mmap_mode='r')
        if X.shape != (len(segments), 12, cfg['segment_length']): raise ValueError('Cache shape changed')
        for position, patient in enumerate(selected.itertuples(), 1):
            path = ROOT / 'PVCVTRawECGData' / f'{patient.hospital_id}.csv'
            marker = out / 'patients' / f'{patient.hospital_id}.json'
            raw_digest = sha256_file(path)
            if marker.is_file():
                prior = read_json(marker)
                if prior['raw_sha256'] != raw_digest or prior['audit_hash'] != canonical_hash(identity):
                    raise RuntimeError(f'Patient audit input changed: {patient.hospital_id}')
                print(f'AUDIT {position}/{len(selected)}: cached {patient.hospital_id}', flush=True)
                continue
            raw = pd.read_csv(path, usecols=LEADS)[LEADS].to_numpy(dtype=float)
            indices = np.flatnonzero((segments.hospital_id == patient.hospital_id).to_numpy())
            rows = segments.iloc[indices]
            clean = reconstruct_windows(X[indices], rows)
            if len(clean) > len(raw): raise ValueError('Cached windows extend beyond the raw recording')
            centers, qa = detect_candidates(raw, {**spec, 'maximum_per_patient': len(raw)})
            counts = window_candidate_counts(centers, rows.start_sample.to_numpy(), cfg['segment_length'])
            result = {'hospital_id': patient.hospital_id, 'class_name': patient.class_name,
                'sublocation': patient.sublocation, 'review_group': patient.review_group,
                'n_wrong_selected_models': int(patient.n_wrong_selected_models),
                'raw_sha256': raw_digest, 'audit_hash': canonical_hash(identity),
                'raw_samples': len(raw), 'raw_duration_seconds': len(raw) / cfg['sampling_rate'],
                'cached_covered_unique_samples': len(clean), 'tail_not_covered_seconds': (len(raw) - len(clean)) / cfg['sampling_rate'],
                'n_windows': len(indices), 'window_duration_seconds': cfg['segment_length'] / cfg['sampling_rate'],
                'windows_without_complete_consensus_candidates': int((counts == 0).sum()),
                'fraction_windows_without_candidates': float((counts == 0).mean()),
                'median_candidates_per_window': float(np.median(counts)),
                'detector_qa': qa, 'candidate_samples': centers.tolist(),
                **signal_profile(raw, 'raw'), **signal_profile(clean, 'cache')}
            write_json(marker, result)
            print(f'AUDIT {position}/{len(selected)}: {patient.hospital_id}, windows={len(indices)}, '
                  f'candidate agreement={qa["agreement"]:.3f}', flush=True)
        del X
        summaries = []
        for p in sorted((out / 'patients').glob('*.json')):
            row = read_json(p)
            if row['audit_hash'] != canonical_hash(identity): raise RuntimeError('Mixed patient audit versions')
            summaries.append({k: v for k, v in row.items() if k not in {'candidate_samples', 'detector_qa'}} |
                             {f'detector_{k}': v for k, v in row['detector_qa'].items()})
        frame = pd.DataFrame(summaries)
        frame.to_csv(out / 'patient_signal_quality.csv', index=False)
        frame.groupby(['class_name', 'review_group']).agg(n_patients=('hospital_id', 'size'),
            median_windows=('n_windows', 'median'), median_raw_seconds=('raw_duration_seconds', 'median'),
            mean_fraction_windows_no_candidates=('fraction_windows_without_candidates', 'mean'),
            median_detector_agreement=('detector_agreement', 'median')).reset_index().to_csv(out / 'quality_by_review_group.csv', index=False)
        status = {'completed_patients': len(frame), 'expected_patients': len(inventory),
            'complete': len(frame) == len(inventory), 'training_run': False,
            'test_signals_read': False, 'clinical_PVC_annotations_created': False,
            'automatic_patient_exclusion': False, 'automatic_label_changes': False}
        write_json(out / 'status.json', status)
        print(status, flush=True)
    return out


def review(run_id=DEFAULT_RUN, count=12):
    out = prepare(run_id)
    _, source, _, segments, cfg, _, identity = audit_context(run_id)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not isinstance(count, int) or count < 1: raise ValueError('Positive review count required')
    with lock(out / '.lock'):
        inventory = pd.read_csv(out / 'development_review_inventory.csv', dtype={'hospital_id': str})
        X = np.load(source / 'data/development/X.npy', mmap_mode='r')
        plots = out / 'waveform_previews'; plots.mkdir(exist_ok=True)
        for patient in inventory.iloc[:count].itertuples():
            marker = out / 'patients' / f'{patient.hospital_id}.json'
            if not marker.is_file(): raise RuntimeError('Run scan for the requested review patients first')
            data = read_json(marker)
            if data['audit_hash'] != canonical_hash(identity): raise RuntimeError('Review audit version changed')
            raw_path = ROOT / 'PVCVTRawECGData' / f'{patient.hospital_id}.csv'
            if sha256_file(raw_path) != data['raw_sha256']: raise RuntimeError('Raw review signal changed')
            raw = pd.read_csv(raw_path, usecols=LEADS)[LEADS].to_numpy(dtype=float)
            indices = np.flatnonzero((segments.hospital_id == patient.hospital_id).to_numpy())
            index = indices[len(indices) // 2]
            start = int(segments.iloc[index].start_sample); window = np.asarray(X[index]).T
            points = np.asarray(data['candidate_samples']); stop = min(len(raw), 5 * cfg['sampling_rate'])
            fig, axes = plt.subplots(12, 2, figsize=(13, 16), sharex='col')
            for lead, (left, right) in enumerate(axes):
                left.plot(np.arange(stop) / cfg['sampling_rate'], raw[:stop, lead], lw=.6)
                selected = points[points < stop]
                left.scatter(selected / cfg['sampling_rate'], raw[selected, lead], s=9, color='tab:red')
                right.plot(np.arange(len(window)) / cfg['sampling_rate'], window[:, lead], lw=.6)
                selected = points[(points >= start) & (points < start + len(window))] - start
                right.scatter(selected / cfg['sampling_rate'], window[selected, lead], s=9, color='tab:red')
                left.set_ylabel(LEADS[lead]); right.set_ylabel(LEADS[lead])
            axes[0, 0].set_title('Raw: first 5 seconds; native data units')
            axes[0, 1].set_title(f'Cached middle window; original start {start/cfg["sampling_rate"]:.2f} s')
            for ax in axes[-1]: ax.set_xlabel('Time (seconds)')
            fig.suptitle(f'{patient.hospital_id} | {patient.class_name} / {patient.sublocation} | {patient.review_group}\n'
                         'Red markers: unverified consensus candidates; not clinical PVC annotations', fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(plots / f'{patient.hospital_id}.png', dpi=120); plt.close(fig)
            print(f'REVIEW {patient.hospital_id}', flush=True)
        del X
    return plots


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'pilot', 'scan', 'review'])
    parser.add_argument('--run-id', default=DEFAULT_RUN); parser.add_argument('--count', type=int, default=12)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare(args.run_id)
    elif args.stage == 'pilot': scan(args.run_id, patients=4)
    elif args.stage == 'scan': scan(args.run_id)
    else: review(args.run_id, args.count)
