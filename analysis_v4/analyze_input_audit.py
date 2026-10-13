"""Verify completed notebook 28 and derive a development-only ablation rationale."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from src.artifacts import ROOT, canonical_hash, read_json, sha256_file, write_json
from research_v4.common import DEFAULT_RUN, context
from research_v4_mps_compat.runtime import audit_run
from .input_audit import audit_context, window_candidate_counts


def analyze(run_id=DEFAULT_RUN):
    run, source, _, cfg, _ = context(run_id)
    audit_run(run)
    out, _, triage, segments, _, _, identity = audit_context(run_id)
    manifest, status = read_json(out / 'manifest.json'), read_json(out / 'status.json')
    if (manifest['audit_hash'] != canonical_hash(identity) or not status['complete']
            or status['completed_patients'] != 230 or status['expected_patients'] != 230):
        raise ValueError('Complete the unchanged input audit before analysis')
    if any(status[key] for key in ('training_run', 'test_signals_read',
                                  'clinical_PVC_annotations_created',
                                  'automatic_patient_exclusion', 'automatic_label_changes')):
        raise ValueError('Unexpected scope or clinical annotation claim')
    for name, digest in manifest['review_file_sha256'].items():
        if sha256_file(out / name) != digest: raise ValueError('Review inventory changed')
    frame = pd.read_csv(out / 'patient_signal_quality.csv', dtype={'hospital_id': str})
    if not frame.hospital_id.is_unique or set(frame.hospital_id) != set(triage.hospital_id) or len(frame) != 230:
        raise ValueError('Incomplete/duplicated development quality table')
    lookup = frame.set_index('hospital_id')
    source_raw = read_json(source / 'data/development/manifest.json')['signal_sha256']
    checked = []
    for patient in triage.itertuples():
        row = lookup.loc[patient.hospital_id]
        saved = read_json(out / 'patients' / f'{patient.hospital_id}.json')
        raw_path = Path('PVCVTRawECGData') / f'{patient.hospital_id}.csv'
        # Resolve against workspace root, independent of caller's current directory.
        digest = sha256_file(ROOT / raw_path)
        if saved['audit_hash'] != canonical_hash(identity) or digest != saved['raw_sha256'] or digest != source_raw[raw_path.name]:
            raise ValueError('Raw audit signal differs from the original preprocessed source')
        flattened = {key: value for key, value in saved.items()
                     if key not in {'candidate_samples', 'detector_qa'}} | {
                         f'detector_{key}': value for key, value in saved['detector_qa'].items()}
        if set(flattened) != set(frame.columns):
            raise ValueError('Quality table columns differ from patient records')
        for key, value in flattened.items():
            actual = row.name if key == 'hospital_id' else row[key]
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                same = np.isclose(actual, value, rtol=1e-12, atol=1e-12)
            elif value == '':
                same = pd.isna(actual) or actual == ''
            else:
                same = actual == value
            if not same:
                raise ValueError(f'Quality table differs from patient record: {patient.hospital_id}/{key}')
        own = segments[segments.hospital_id == patient.hospital_id]
        counts = window_candidate_counts(saved['candidate_samples'], own.start_sample.to_numpy(), cfg['segment_length'])
        expected = {'n_windows': len(own), 'windows_without_complete_consensus_candidates': int((counts == 0).sum()),
                    'fraction_windows_without_candidates': float((counts == 0).mean()),
                    'median_candidates_per_window': float(np.median(counts))}
        for key, value in expected.items():
            if not np.isclose(value, saved[key], rtol=1e-12, atol=1e-12) or not np.isclose(value, row[key], rtol=1e-12, atol=1e-12):
                raise ValueError(f'Window summary mismatch: {patient.hospital_id}/{key}')
        if saved['class_name'] != patient.class_name or saved['sublocation'] != patient.sublocation or saved['n_wrong_selected_models'] != patient.n_wrong_selected_models:
            raise ValueError('Review metadata mismatch')
        if saved['detector_qa']['pvc_verified'] or saved['detector_qa']['expert_reviewed']:
            raise ValueError('Unexpected clinical annotation claim')
        if len(saved['candidate_samples']) != saved['detector_qa']['n_retained'] or saved['detector_qa']['n_retained'] != saved['detector_qa']['n_complete_candidates']:
            raise ValueError('Candidate cap unexpectedly applied')
        checked.append({'hospital_id': patient.hospital_id, 'raw_sha256': digest,
                        'patient_audit_sha256': sha256_file(out / 'patients' / f'{patient.hospital_id}.json')})
    derived = out / 'analysis_20261003_complete'; derived.mkdir(exist_ok=True)
    pairs = pd.read_csv(out / 'hard_control_pairs.csv', dtype={'hard_id': str, 'control_id': str})
    pair_rows = []
    for pair in pairs.itertuples():
        if pd.isna(pair.control_id): continue
        hard, control = lookup.loc[pair.hard_id], lookup.loc[pair.control_id]
        values = {'hard_id': pair.hard_id, 'control_id': pair.control_id, 'matching': pair.matching,
                  'reused_control': bool(pair.reused_control), 'class_name': hard.class_name}
        for key in ('n_windows', 'raw_duration_seconds', 'detector_agreement', 'fraction_windows_without_candidates'):
            values[f'hard_{key}'], values[f'control_{key}'] = hard[key], control[key]
            values[f'hard_minus_control_{key}'] = hard[key] - control[key]
        pair_rows.append(values)
    pd.DataFrame(pair_rows).to_csv(derived / 'descriptive_matched_pairs.csv', index=False)
    n = frame.n_windows.to_numpy()
    coverage = frame[['hospital_id', 'class_name', 'sublocation', 'review_group', 'n_windows']].copy()
    for k in (4, 8):
        coverage[f'windows_per_visit_k{k}'] = np.minimum(k, n)
        coverage[f'fraction_windows_per_visit_k{k}'] = np.minimum(k, n) / n
        for choice in read_json(run / 'selection.json')['choices']:
            if choice['model'] in {'encoder', 'hierarchical'}:
                # Expectation under independent uniform window sampling, not observed usage.
                coverage[f'theoretical_coverage_{choice["model"]}_k{k}'] = 1 - (1 - np.minimum(k, n) / n) ** choice['epochs']
    coverage.to_csv(derived / 'training_window_coverage_hypothesis.csv', index=False)
    flags = frame[frame.detector_engineering_review_flag]
    flags[['hospital_id', 'class_name', 'sublocation', 'review_group', 'detector_agreement',
           'detector_candidate_rate_per_minute', 'fraction_windows_without_candidates']].to_csv(derived / 'detector_review_flags.csv', index=False)
    frame[frame.fraction_windows_without_candidates > 0][['hospital_id', 'class_name', 'review_group', 'n_windows',
        'windows_without_complete_consensus_candidates', 'detector_agreement']].to_csv(derived / 'windows_without_candidates_patients.csv', index=False)
    summary = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
        'input_audit_hash': manifest['audit_hash'], 'development_patients': len(frame),
        'native_windows': int(n.sum()), 'native_window_samples': cfg['segment_length'],
        'sampling_rate_hz': cfg['sampling_rate'], 'patients_with_numerically_constant_raw_or_cache_leads': int(((frame.raw_numerically_constant_leads > 0) | (frame.cache_numerically_constant_leads > 0)).sum()),
        'patients_with_exact_duplicate_raw_or_cache_leads': int((frame.raw_exact_duplicate_lead_pairs.notna() | frame.cache_exact_duplicate_lead_pairs.notna()).sum()),
        'windows_without_complete_consensus_candidates': int(frame.windows_without_complete_consensus_candidates.sum()),
        'fraction_all_windows_without_candidates': float(frame.windows_without_complete_consensus_candidates.sum() / n.sum()),
        'patients_with_windows_without_candidates': int((frame.fraction_windows_without_candidates > 0).sum()),
        'detector_flagged_patients': len(flags), 'detector_flagged_hard_patients': int((flags.review_group == 'hard').sum()),
        'detector_agreement_median_all': float(frame.detector_agreement.median()),
        'detector_agreement_median_hard': float(lookup.loc[pairs.hard_id, 'detector_agreement'].median()),
        'detector_agreement_median_matched_controls_with_reuse': float(lookup.loc[pairs.control_id, 'detector_agreement'].median()),
        'raw_duration_median_seconds': float(frame.raw_duration_seconds.median()),
        'uncovered_tail_median_seconds': float(frame.tail_not_covered_seconds.median()),
        'n_windows_min': int(n.min()), 'n_windows_median': float(np.median(n)), 'n_windows_max': int(n.max()),
        'patients_with_more_than_four_windows': int((n > 4).sum()), 'patients_with_more_than_eight_windows': int((n > 8).sum()),
        'median_fraction_windows_per_training_visit_k4': float(np.median(np.minimum(4, n) / n)),
        'median_fraction_windows_per_training_visit_k8': float(np.median(np.minimum(8, n) / n)),
        'overlapping_windows_are_not_independent': True, 'detector_is_not_PVC_classifier': True,
        'raw_hashes_match_preprocessed_source': True, 'selection_uses_development_only': True,
        'test_signals_read': False, 'new_classifier_training': False, 'automatic_patient_or_label_changes': False,
        'analysis_source_sha256': sha256_file(Path(__file__))}
    write_json(derived / 'input_audit_analysis.json', summary)
    pd.DataFrame(checked).to_csv(derived / 'verified_patient_inputs.csv', index=False)
    print(summary)
    print('Outputs:', derived)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    analyze(parser.parse_args().run_id)
