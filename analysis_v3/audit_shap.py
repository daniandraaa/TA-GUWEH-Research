"""Audit saved V3 SHAP artifacts without training or recomputing attributions.

Small CPU forward passes describe the explained segments. Patient predictions
are read from the completed test; segment logits are not patient explanations.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from research_v3.common import DEFAULT_RUN, context, require_final_complete
from research_v3.data import load_pool, representation
from research_v3.models import build_model
from src.artifacts import ROOT, canonical_hash, read_json, sha256_file, write_json
from src.data import ECGDataset, LEADS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(explanation, run_id=DEFAULT_RUN, output=None):
    explanation = Path(explanation).resolve()
    run, source, cfg, config, _, _ = context(run_id)
    require_final_complete(run, cfg)
    require(explanation.parent == run / 'explain_primary_v3', 'Explanation belongs to a different run')
    manifest = read_json(explanation / 'manifest.json')
    metadata = {'signature', 'patient_probability_explained', 'clinical_validation',
                'feature_dependence_caveat', 'input'}
    signature = canonical_hash({k: v for k, v in manifest.items() if k not in metadata})
    require(signature == manifest['signature'] and explanation.name == signature[:12], 'SHAP signature mismatch')
    require(manifest['run_id'] == run_id, 'SHAP run-id mismatch')
    require(manifest['experiment_hash'] == read_json(run / 'protocol.json')['experiment_hash'], 'SHAP protocol mismatch')
    decision = read_json(run / 'selection.json')
    choice, seed = decision['primary'], decision['primary_seed']
    name, arm = choice['model'], choice['arm']
    require((manifest['model'], manifest['arm'], manifest['primary_seed']) == (name, arm, seed), 'SHAP is not the selected primary')
    checkpoint = run / 'final' / name / f'seed_{seed}/model.pth'
    scaler_path = run / 'scalers' / representation(arm) / 'final/scaler.joblib'
    owner = source if representation(arm) == 'window' else run / 'representations/r_candidate'
    paths = {
        'checkpoint_sha256': checkpoint, 'scaler_sha256': scaler_path,
        'code_sha256': ROOT / 'analysis_v3/explain_final.py',
        'helper_sha256': ROOT / 'analysis_v2/explain_v2.py',
    }
    for pool, prefix in [('development', 'development'), ('test', 'test')]:
        paths[f'{prefix}_waveform_sha256'] = owner / 'data' / pool / 'X.npy'
        paths[f'{prefix}_segments_sha256'] = owner / 'data' / pool / 'segments.csv'
    actual_hashes = {key: sha256_file(path) for key, path in paths.items()}
    for key, digest in actual_hashes.items():
        require(digest == manifest[key], f'Changed SHAP input: {key}')
    X, development = load_pool(run, source, arm)
    Xt, historical = load_pool(run, source, arm, 'test')
    tables = {}
    for key, table, pool in [('background', development, 'development'), ('explained', historical, 'test')]:
        saved = pd.read_csv(explanation / f'{key}_patients.csv', dtype={'hospital_id': str})
        indices = saved.segment_index.to_numpy(dtype=int)
        require(indices.tolist() == manifest[key], f'{key} order differs from manifest')
        require(saved.hospital_id.is_unique and saved.segment_index.is_unique, f'Duplicate {key} patient/segment')
        require(len(indices) > 0 and ((indices >= 0) & (indices < len(table))).all(), f'Invalid {key} indices')
        expected = table.iloc[indices].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved.drop(columns='segment_index'), expected, check_dtype=False)
        representatives = table.reset_index(names='segment_index').groupby('hospital_id', sort=True)
        middle = {hid: int(g.iloc[len(g) // 2].segment_index) for hid, g in representatives}
        require(all(middle[r.hospital_id] == r.segment_index for r in saved.itertuples()), f'{key} is not a middle segment per patient')
        cohort = {str(p['hospital_id']) for p in read_json(run / 'data/patients.json') if p['pool'] == pool}
        require(set(saved.hospital_id) <= cohort, f'{key} is outside its patient pool')
        tables[key] = saved
    bg, ex = tables['background'], tables['explained']
    require(not set(bg.hospital_id) & set(ex.hospital_id), 'Test patient entered SHAP background')
    require(manifest['sampling_rate_hz'] == config['sampling_rate'], 'Wrong SHAP sampling rate')
    require(manifest['segment_samples'] == X.shape[-1] == Xt.shape[-1], 'Wrong SHAP segment length')
    require(manifest['output'] == 'true-class segment logit' and manifest['patient_probability_explained'] is False,
            'Unexpected SHAP target')
    require(manifest['clinical_validation'] is False, 'Unsupported clinical-validation metadata')
    values = np.load(explanation / 'shap_true_class_values.npy', allow_pickle=False)
    require(values.shape == (len(ex), len(LEADS), Xt.shape[-1]) and np.isfinite(values).all(), 'Invalid SHAP array')
    require(len(list(explanation.glob('patient_*.npy'))) == len(ex), 'Unexpected per-patient SHAP inventory')
    for i, row in ex.iterrows():
        single = np.load(explanation / f'patient_{row.hospital_id}.npy', allow_pickle=False)
        require(single.shape == (1, *values.shape[1:]) and np.array_equal(single[0], values[i]), 'Individual SHAP differs from aggregate')
    importance = pd.DataFrame({'lead': LEADS, 'mean_absolute_shap': np.abs(values).mean(axis=(0, 2))})
    importance = importance.sort_values('mean_absolute_shap', ascending=False).reset_index(drop=True)
    stored = pd.read_csv(explanation / 'lead_importance.csv')
    require(stored.lead.tolist() == importance.lead.tolist(), 'Wrong lead ranking')
    require(np.allclose(stored.mean_absolute_shap, importance.mean_absolute_shap, rtol=1e-6, atol=1e-10), 'Wrong lead importance')

    # Describe segment and saved patient predictions separately. No SHAP rerun.
    dataset = ECGDataset(Xt, historical.label, scaler=joblib.load(scaler_path))
    model = build_model(name, config['model']).cpu().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            logits = torch.cat([model(dataset[int(i)][0][None]) for i in ex.segment_index]).numpy()
    finally:
        torch.backends.mha.set_fastpath_enabled(old_fastpath)
    require(np.isfinite(logits).all(), 'Nonfinite segment output')
    segment_predictions = logits.argmax(axis=1)
    patients = pd.read_csv(run / 'test' / name / f'seed_{seed}/patient_predictions.csv', dtype={'hospital_id': str})
    require(patients.hospital_id.is_unique, 'Duplicate saved test patient')
    details = ex.merge(patients, on='hospital_id', validate='one_to_one', how='left', sort=False)
    require(details.predicted_class.notna().all() and np.array_equal(details.label, details.true_label), 'Explained patient labels changed')
    details['segment_predicted_class'] = [config['class_names'][int(p)] for p in segment_predictions]
    details['segment_correct'] = segment_predictions == ex.label.to_numpy()
    details['segment_true_class_logit'] = logits[np.arange(len(ex)), ex.label.to_numpy(dtype=int)]
    details['top_shap_lead'] = [LEADS[int(i)] for i in np.abs(values).mean(axis=2).argmax(axis=1)]
    class_rows = []
    for label, rows in ex.groupby('label'):
        means = np.abs(values[rows.index.to_numpy()]).mean(axis=(0, 2))
        class_rows.extend({'label': int(label), 'class_name': config['class_names'][int(label)],
                           'patients': len(rows), 'lead': lead, 'mean_absolute_shap': float(value)}
                          for lead, value in zip(LEADS, means))
    notebook_status = {}
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        if path.name[:2] not in {'17', '18', '19'}:
            continue
        cells = [c for c in read_json(path)['cells'] if c['cell_type'] == 'code']
        notebook_status[path.name] = {'code_cells': len(cells),
            'executed_cells': sum(c.get('execution_count') is not None for c in cells),
            'saved_errors': sum(o['output_type'] == 'error' for c in cells for o in c.get('outputs', [])),
            'sha256': sha256_file(path)}
    report = {'audited_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
        'experiment_hash': manifest['experiment_hash'], 'explanation': str(explanation), 'signature': signature,
        'audit_code_sha256': sha256_file(__file__), 'input_hashes_verified': actual_hashes,
        'manifest_sha256': sha256_file(explanation / 'manifest.json'),
        'aggregate_array_sha256': sha256_file(explanation / 'shap_true_class_values.npy'),
        'model': name, 'primary_seed': seed, 'background_patients': len(bg), 'explained_patients': len(ex),
        'background_class_counts': {int(k): int(v) for k, v in bg.label.value_counts().sort_index().items()},
        'explained_class_counts': {int(k): int(v) for k, v in ex.label.value_counts().sort_index().items()},
        'background_development_only': True, 'background_test_patient_overlap': 0,
        'array_shape': list(values.shape), 'all_values_finite': True, 'individual_arrays_match': True,
        'lead_importance_recomputed': True, 'lead_ranking': importance.to_dict('records'),
        'sample_patient_predictions_correct': int(details.correct.sum()),
        'sample_segment_predictions_correct': int(details.segment_correct.sum()),
        'sample_patient_segment_disagreement': int((details.predicted_class != details.segment_predicted_class).sum()),
        'notebook_status': notebook_status, 'target': manifest['output'],
        'shap_values_recomputed': False, 'clinical_validation': False, 'patient_probability_explained': False,
        'training_run': False, 'sample_covers_all_test_patients': len(ex) == historical.hospital_id.nunique()}
    output = Path(output) if output else run / 'analysis_20261002_complete'
    output.mkdir(parents=True, exist_ok=True)
    details.to_csv(output / 'shap_v3_patient_context.csv', index=False)
    pd.DataFrame(class_rows).to_csv(output / 'shap_v3_class_lead_importance.csv', index=False)
    write_json(output / 'shap_v3_audit.json', report)
    print(f'Verified SHAP {name} seed {seed}: {len(bg)} background, {len(ex)} explained patients; {values.shape}', flush=True)
    print(f'Patient correct {report["sample_patient_predictions_correct"]}/{len(ex)}, segment correct {report["sample_segment_predictions_correct"]}/{len(ex)}; top leads: {", ".join(importance.lead.iloc[:3])}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--explanation', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    audit(args.explanation, args.run_id, args.output)
