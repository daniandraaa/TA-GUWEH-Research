"""Audit completed V3 CV and existing V2 SHAP, without fitting any model.

Run manually: .venv/bin/python -m analysis_v3.analyze_completed
Writes derived CSV/JSON/figures only to a separate analysis directory.
"""
import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)

from src.artifacts import ROOT, canonical_hash, read_json, sha256_file, write_json
from src.data import LEADS
from research_v3.common import DEFAULT_RUN, context, require_final_complete
from research_v3.classical import scenarios as tree_scenarios


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recompute(truth, probabilities):
    """Calculate from saved patient probabilities independently of training code."""
    pred = probabilities.argmax(axis=1)
    cm = confusion_matrix(truth, pred, labels=[0, 1, 2])
    specificities = []
    for i in range(3):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp))
    return dict(accuracy=accuracy_score(truth, pred),
                precision=precision_score(truth, pred, labels=[0, 1, 2], average='macro', zero_division=0),
                sensitivity=recall_score(truth, pred, labels=[0, 1, 2], average='macro', zero_division=0),
                specificity=float(np.mean(specificities)),
                f1=f1_score(truth, pred, labels=[0, 1, 2], average='macro', zero_division=0),
                auc=roc_auc_score(truth, probabilities, multi_class='ovr', average='macro', labels=[0, 1, 2]))


def notebook_status():
    rows = []
    flags = {'RUN_FULL_GRID', 'RUN_FINAL_AND_TEST', 'INSTALL_WFDB', 'DOWNLOAD_DATA', 'RUN_BENCHMARK'}
    for p in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        if p.name[:2] not in {str(i) for i in range(11, 18)}:
            continue
        n = read_json(p)
        cells = [c for c in n['cells'] if c['cell_type'] == 'code']
        values = {}
        for c in cells:
            for node in ast.parse(''.join(c['source'])).body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id in flags:
                            values[t.id] = node.value.value
        rows.append({'notebook': p.name, 'code_cells': len(cells),
                     'executed_cells': sum(c.get('execution_count') is not None for c in cells),
                     'saved_errors': sum(o['output_type'] == 'error' for c in cells for o in c.get('outputs', [])),
                     **values})
    return pd.DataFrame(rows)


def audit_shap(source, records):
    rows = []
    development = {r['hospital_id'] for r in records if r['pool'] == 'development'}
    historical = {r['hospital_id'] for r in records if r['pool'] == 'test'}
    for path in sorted((source / 'explain_patient_v2').glob('*/manifest.json')):
        m, folder = read_json(path), path.parent
        bg = pd.read_csv(folder / 'background_patients.csv', dtype={'hospital_id': str})
        ex = pd.read_csv(folder / 'explained_patients.csv', dtype={'hospital_id': str})
        require(bg.hospital_id.is_unique and ex.hospital_id.is_unique, 'SHAP repeats patients')
        require(set(bg.hospital_id) <= development and set(ex.hospital_id) <= historical, 'SHAP pool mismatch')
        require(not set(bg.hospital_id) & set(ex.hospital_id), 'SHAP background includes explained test patients')
        require(bg.segment_index.tolist() == m['background'] and ex.segment_index.tolist() == m['explained'], 'SHAP indices changed')
        for pool, selected in [('development', bg), ('test', ex)]:
            segments = pd.read_csv(source / 'data' / pool / 'segments.csv', dtype={'hospital_id': str})
            for row in selected.itertuples():
                actual = segments.iloc[row.segment_index]
                require(actual.hospital_id == row.hospital_id and actual.label == row.label, 'SHAP metadata mismatch')
        checkpoint = source / 'final' / m['model'] / f"seed_{m['primary_seed']}/model.pth"
        require(sha256_file(checkpoint) == m['checkpoint_sha256'], 'SHAP checkpoint changed')
        require(sha256_file(ROOT / 'analysis_v2/explain_v2.py') == m['code_sha256'], 'SHAP code changed')
        fields = ['model', 'primary_seed', 'checkpoint_sha256', 'code_sha256', 'background', 'explained',
                  'nsamples', 'seed', 'device', 'method', 'output', 'shap_version', 'torch_version']
        require(canonical_hash({k: m[k] for k in fields}) == m['signature'], 'SHAP signature mismatch')
        values = np.load(folder / 'shap_true_class_values.npy')
        require(values.shape == (len(ex), 12, 5000) and np.isfinite(values).all(), 'Invalid SHAP values')
        for i, hid in enumerate(ex.hospital_id):
            require(np.array_equal(np.load(folder / f'patient_{hid}.npy')[0], values[i]), 'SHAP patient/aggregate mismatch')
        importance = pd.read_csv(folder / 'lead_importance.csv').set_index('lead').loc[LEADS]
        require(np.allclose(importance.mean_absolute_shap, np.abs(values).mean(axis=(0, 2)), rtol=1e-5, atol=1e-9), 'SHAP importance mismatch')
        top = importance.sort_values('mean_absolute_shap', ascending=False)
        rows.append({'folder': str(folder), 'model': m['model'], 'seed': m['primary_seed'],
                     'background_patients': len(bg), 'explained_patients': len(ex),
                     'explained_per_class': {str(k): int(v) for k, v in ex.label.value_counts().sort_index().items()},
                     'shape': list(values.shape), 'finite': True, 'verified_checkpoint': True,
                     'top_three_leads': top.index[:3].tolist(), 'nsamples': m['nsamples'],
                     'output': m['output'], 'patient_probability_explained': m['patient_probability_explained'],
                     'clinical_validation': m['clinical_validation']})
    return rows


def analyze(run_id=DEFAULT_RUN, output=None):
    run, source, cfg, config, neural_grid, v2cfg = context(run_id)
    out = Path(output) if output else run / 'analysis_20261001_pilot'
    records = read_json(run / 'data/patients.json')
    folds = read_json(run / 'folds/manifest.json')
    classes = config['class_names']
    labels = {r['hospital_id']: r['label'] for r in records if r['pool'] == 'development'}
    signature = read_json(run / 'protocol.json')['experiment_hash']
    rows, histories, oof, missing, progress, pooled, class_rows = [], [], {}, [], [], [], []
    max_error = 0.
    expected_paths = set()
    for model_index, model in enumerate(cfg['models'] + ['extratrees']):
        tree = model == 'extratrees'
        grid = tree_scenarios(cfg) if tree else neural_grid
        arms = cfg['classical_arms'] if tree else cfg['arms']
        for arm in arms:
            completed = 0
            for sc in grid:
                for fold in folds:
                    f = fold['fold']
                    directory = run / 'cv' / arm / model / sc['id'] / f'fold_{f}'
                    metric_path = directory / 'metrics.json'
                    expected_paths.add(metric_path)
                    if not metric_path.exists():
                        missing.append(dict(model=model, arm=arm, scenario=sc['id'], fold=f))
                        continue
                    completed += 1
                    m, case = read_json(metric_path), read_json(directory / 'config.json')
                    require(case['experiment_hash'] == signature and case['model'] == model and case['arm'] == arm
                            and case['scenario'] == sc and case['fold'] == f, f'Case identity mismatch: {directory}')
                    require(case['seed'] == config['seed'] + 100 * f + 17 * model_index, 'CV seed mismatch')
                    require(set(case['train_patient_ids']) == set(fold['train_ids']), 'Training patient identity mismatch')
                    p = pd.read_csv(directory / 'validation_predictions.csv', dtype={'hospital_id': str})
                    require(p.hospital_id.is_unique and set(p.hospital_id) == set(fold['validation_ids']), 'Validation coverage mismatch')
                    require(not set(p.hospital_id) & set(case['train_patient_ids']), 'Patient leakage')
                    require(p.true_label.tolist() == [labels[h] for h in p.hospital_id], 'Validation label mismatch')
                    probs = p[[f'probability_{c}' for c in classes]].to_numpy()
                    require(np.isfinite(probs).all() and (probs >= -1e-7).all()
                            and (probs <= 1 + 1e-7).all() and np.allclose(probs.sum(1), 1, atol=2e-6), 'Invalid probabilities')
                    pred = probs.argmax(1)
                    require(p.predicted_class.tolist() == [classes[i] for i in pred]
                            and np.array_equal(p.correct.to_numpy(), pred == p.true_label.to_numpy()), 'Prediction metadata mismatch')
                    for key, actual in recompute(p.true_label.to_numpy(), probs).items():
                        error = abs(actual - m[key])
                        max_error = max(max_error, error)
                        require(error < 1e-7, f'Metric mismatch: {directory}/{key}')
                    require(m['n_validation_patients'] == len(p), 'Patient count mismatch')
                    rows.append({'arm': arm, 'model': model, 'scenario': sc['id'], 'fold': f, **m})
                    oof.setdefault((model, arm, sc['id']), []).append(p.assign(fold=f))
                    if not tree:
                        history = pd.read_csv(directory / 'history.csv')
                        require(len(history) == m['epochs_ran'] and history.epoch.tolist() == list(range(1, len(history) + 1)), 'History incomplete')
                        require(abs(history.validation_f1.max() - m['f1']) < 1e-7, 'History best score mismatch')
                        require(abs(history.set_index('epoch').loc[m['best_epoch'], 'validation_f1'] - m['f1']) < 1e-7, 'Best epoch mismatch')
                        histories.append({'model': model, 'arm': arm, 'scenario': sc['id'], 'fold': f,
                                          'best_epoch': m['best_epoch'], 'epochs_ran': m['epochs_ran'],
                                          'first_train_loss': history.train_loss.iloc[0], 'last_train_loss': history.train_loss.iloc[-1],
                                          'best_validation_f1': m['f1'], 'last_validation_f1': history.validation_f1.iloc[-1],
                                          'recorded_epoch_seconds': history.seconds.sum()})
            progress.append({'model': model, 'arm': arm, 'complete': completed,
                             'expected': len(grid) * len(folds), 'missing': len(grid) * len(folds) - completed})
    require(not set((run / 'cv').rglob('metrics.json')) - expected_paths, 'Unexpected CV cases')
    for (model, arm, sc), tables in oof.items():
        if len(tables) != len(folds):
            continue
        p = pd.concat(tables, ignore_index=True)
        require(p.hospital_id.is_unique and set(p.hospital_id) == set(labels), 'OOF coverage mismatch')
        probs = p[[f'probability_{c}' for c in classes]].to_numpy()
        cm = confusion_matrix(p.true_label, probs.argmax(1), labels=[0, 1, 2])
        pooled.append({'model': model, 'arm': arm, 'scenario': sc, 'n_patients': len(p), **recompute(p.true_label.to_numpy(), probs)})
        for i, cls in enumerate(classes):
            class_rows.append({'model': model, 'arm': arm, 'scenario': sc, 'class': cls,
                               'support': int(cm[i].sum()), 'correct': int(cm[i, i]),
                               'recall': float(cm[i, i] / cm[i].sum())})
    cv = pd.DataFrame(rows)
    summary = cv.groupby(['model', 'arm', 'scenario'], sort=False).agg(
        n_folds=('fold', 'count'), f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
        accuracy_mean=('accuracy', 'mean')).reset_index()
    recorded_summary = pd.read_csv(run / 'reports/cv_summary.csv')
    complete = summary[summary.n_folds == len(folds)]
    require(len(complete) == len(recorded_summary), 'CV summary coverage mismatch')
    merged = complete.merge(recorded_summary, on=['model', 'arm', 'scenario'], suffixes=('_audit', '_saved'), validate='one_to_one')
    require(len(merged) == len(complete), 'CV summary case identities mismatch')
    for key in ['f1_mean', 'f1_std']:
        require(np.allclose(merged[f'{key}_audit'], merged[f'{key}_saved'], atol=1e-7), 'CV summary score mismatch')
    paired = []
    effects = [('window_segment', 'window_patient', 'patient_sampling_window'),
               ('r_candidate_segment', 'r_candidate_patient', 'patient_sampling_candidate'),
               ('window_segment', 'r_candidate_segment', 'candidate_representation_segment'),
               ('window_patient', 'r_candidate_patient', 'candidate_representation_patient')]
    for model, data in cv.groupby('model', sort=False):
        pivot = data.pivot(index=['scenario', 'fold'], columns='arm', values='f1')
        for old, new, effect in effects:
            if old not in pivot or new not in pivot:
                continue
            for (sc, f), pair in pivot[[old, new]].dropna().iterrows():
                paired.append(dict(model=model, effect=effect, scenario=sc, fold=f,
                                   reference_arm=old, comparison_arm=new, delta_f1=pair[new] - pair[old]))
    paired = pd.DataFrame(paired)
    paired_summary = paired.groupby(['model', 'effect'], sort=False).agg(
        n_pairs=('delta_f1', 'count'), mean_delta=('delta_f1', 'mean'),
        min_delta=('delta_f1', 'min'), max_delta=('delta_f1', 'max'),
        wins=('delta_f1', lambda x: int((x > 1e-12).sum())),
        losses=('delta_f1', lambda x: int((x < -1e-12).sum()))).reset_index()
    # Check patient cohort, coordinates, cached arrays and raw source integrity.
    cache = run / 'representations/r_candidate/data/development'
    manifest = read_json(cache / 'manifest.json')
    segments = pd.read_csv(cache / 'segments.csv', dtype={'hospital_id': str})
    quality = pd.read_csv(cache / 'candidate_quality.csv', dtype={'hospital_id': str})
    X = np.load(cache / 'X.npy', mmap_mode='r')
    width = round(cfg['candidate']['fs'] * (cfg['candidate']['before_seconds'] + cfg['candidate']['after_seconds']))
    require(X.shape == (len(segments), 12, width) and set(segments.hospital_id) == set(labels), 'Candidate cohort/shape mismatch')
    require(segments.label.tolist() == [labels[h] for h in segments.hospital_id], 'Candidate labels changed')
    require(not segments.pvc_verified.any() and not quality.pvc_verified.any()
            and not quality.expert_reviewed.any(), 'Unexpected clinical validation flag')
    require(quality.hospital_id.is_unique and set(quality.hospital_id) == set(labels), 'QC coverage mismatch')
    for hid, group in segments.groupby('hospital_id', sort=False):
        marker = read_json(cache / 'patients' / f'{hid}.json')
        shard_path = cache / 'patients' / f'{hid}.npy'
        require(sha256_file(shard_path) == marker['array_sha256'], f'Candidate shard changed: {hid}')
        raw_path = ROOT / v2cfg['preprocessing']['input'] / f'{hid}.csv'
        require(sha256_file(raw_path) == marker['raw_sha256'] == manifest['signal_sha256'][hid], f'Raw input changed: {hid}')
        values = np.load(shard_path)
        require(np.isfinite(values).all() and np.array_equal(X[group.index], values), f'Candidate aggregate changed: {hid}')
        require(quality.set_index('hospital_id').loc[hid, 'n_retained'] == len(group), 'QC crop count mismatch')
        require(len(group) <= cfg['candidate']['maximum_per_patient'], 'Candidate cap violated')
        ordinary = group[~group.is_fallback]
        require((ordinary.r_candidate_sample - ordinary.start_sample == round(cfg['candidate']['before_seconds'] * cfg['candidate']['fs'])).all()
                and (ordinary.valid_samples == width).all(), 'Candidate coordinates mismatch')
    for representation in ['window', 'r_candidate']:
        table = segments if representation == 'r_candidate' else pd.read_csv(source / 'data/development/segments.csv', dtype={'hospital_id': str})
        for fold in folds:
            provenance = read_json(run / 'scalers' / representation / f"fold_{fold['fold']}/scaler_provenance.json")
            require(set(provenance['fit_patient_ids']) == set(fold['train_ids']), 'Scaler patient leakage')
            require(provenance['n_segments'] == int(table.hospital_id.isin(fold['train_ids']).sum()), 'Scaler segment coverage mismatch')
    shap = audit_shap(source, records)
    try:
        require_final_complete(run, cfg)
        test_gate = 'open_all_final_complete'
    except RuntimeError as exc:
        test_gate = str(exc)
    audit = {'audited_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
             'experiment_hash': signature, 'frozen_identity_valid': True,
             'completed_cv': len(cv), 'expected_cv': len(expected_paths), 'missing_cv': len(missing),
             'independently_recomputed_cv': len(cv), 'maximum_metric_absolute_error': max_error,
             'validation_patients_per_fold': sorted(cv.n_validation_patients.unique().astype(int).tolist()),
             'development_patients': len(labels), 'patient_split_and_scaler_provenance_valid': True,
             'candidate_patients': len(quality), 'candidate_crops': len(segments),
             'raw_and_candidate_shard_hashes_verified': True, 'aggregate_matches_shards': True,
             'engineering_flags': int(quality.engineering_review_flag.sum()), 'fallback_patients': int(quality.used_fallback.sum()),
             'pvc_verified': False, 'candidate_count_min': int(quality.n_retained.min()),
             'candidate_count_median': float(quality.n_retained.median()), 'candidate_count_max': int(quality.n_retained.max()),
             'agreement_median': float(quality.agreement.median()), 'shap': shap,
             'selection_exists': (run / 'selection.json').exists(),
             'final_complete': len(list((run / 'final').rglob('metrics.json'))),
             'test_complete': len(list((run / 'test').rglob('metrics.json'))), 'test_gate': test_gate,
             'incart_results_exist': (ROOT / 'results/detector_benchmark/incart/summary.json').exists(),
             'limitations': ['Partial neural grid cannot select final model.',
                             'CV used for selection/early stopping is not nested performance estimation.',
                             'Paired folds share training patients; pairs are not independent replicates.',
                             'Representation comparison includes crop duration, centers and per-patient crop counts.',
                             'Historical test has already been observed in earlier work.',
                             'SHAP explains true-class segment logits; no clinical interpretation validated.']}
    out.mkdir(parents=True, exist_ok=True)
    frames = {'progress': pd.DataFrame(progress), 'missing_cv_cases': pd.DataFrame(missing, columns=['model', 'arm', 'scenario', 'fold']),
              'independent_cv_folds': cv, 'independent_cv_summary': summary,
              'pooled_oof_metrics': pd.DataFrame(pooled), 'oof_class_recall': pd.DataFrame(class_rows),
              'matched_folds': paired, 'matched_effect_summary': paired_summary,
              'training_history_summary': pd.DataFrame(histories), 'notebook_status': notebook_status(),
              'candidate_quality_flags': quality[quality.engineering_review_flag]}
    for name, frame in frames.items():
        frame.to_csv(out / f'{name}.csv', index=False)
    write_json(out / 'audit.json', audit)
    make_figures(out, cv, summary, paired, pd.DataFrame(class_rows))
    print(f'AUDIT PASS: {len(cv)}/{len(expected_paths)} CV, max metric error {max_error:.3g}')
    print(pd.DataFrame(progress).to_string(index=False))
    print(paired_summary.to_string(index=False))
    print(f'Analysis: {out}')
    return out


def make_figures(out, cv, summary, paired, recall):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = ['encoder', 'patchtst', 'hierarchical']
    arms = ['window_segment', 'window_patient', 'r_candidate_segment', 'r_candidate_patient']
    colors = ['#2563eb', '#16a34a', '#d97706', '#dc2626']
    # Plot only matched S01, rather than comparing different winning settings.
    pilot = summary[(summary.scenario == 'S01') & (summary.n_folds == 5)]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, arm in enumerate(arms):
        table = pilot[pilot.arm == arm].set_index('model').reindex(models)
        ax.bar(np.arange(3) + (i - 1.5) * .19, table.f1_mean, width=.18,
               yerr=table.f1_std, capsize=3, label=arm, color=colors[i])
    ax.set(xticks=np.arange(3), xticklabels=['Encoder', 'CNN patch transformer', 'Hierarchical'],
           ylabel='Patient macro-F1: mean ± sample SD of 5 folds', ylim=(0, .85),
           title='Matched S01 pilot: LR 1e-4, batch 16; full neural grid pending')
    ax.legend(ncol=2, fontsize=9); ax.grid(axis='y', alpha=.18)
    fig.tight_layout(); fig.savefig(out / 'pilot_cv.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, effect in zip(axes, ['candidate_representation_patient', 'patient_sampling_window']):
        data = paired[(paired.effect == effect) & ((paired.scenario == 'S01') | (paired.model == 'extratrees'))]
        names = models + (['extratrees'] if effect == 'candidate_representation_patient' else [])
        for i, model in enumerate(names):
            values = data[data.model == model].delta_f1.to_numpy()
            ax.scatter(np.full(len(values), i), values, alpha=.5, s=25)
            if len(values): ax.scatter(i, values.mean(), marker='D', s=65, color='black')
        ax.axhline(0, color='gray', lw=1)
        ax.set(xticks=range(len(names)), xticklabels=names,
               title='Candidate − window, patient sampler' if effect.startswith('candidate') else 'Patient − segment sampler, 2.5 s windows')
        ax.tick_params(axis='x', rotation=15); ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Matched CV macro-F1 difference; diamond = mean')
    fig.suptitle('Descriptive pairs: neural S01; ExtraTrees T01–T09; shared training patients')
    fig.tight_layout(); fig.savefig(out / 'matched_differences.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, models):
        table = recall[(recall.model == model) & (recall.scenario == 'S01')]
        matrix = table.pivot(index='class', columns='arm', values='recall').reindex(
            index=['LVOT', 'RVOT_cusp', 'RVOT_septal_freewall_other'], columns=arms)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f'{matrix.iloc[i,j]:.2f}', ha='center', va='center', color='white' if matrix.iloc[i,j] > .6 else 'black')
        ax.set(xticks=range(4), xticklabels=['W segment', 'W patient', 'R segment', 'R patient'],
               yticks=range(3), yticklabels=['LVOT (57)', 'RVOT cusp (103)', 'RVOT other (70)'], title=model)
        ax.tick_params(axis='x', rotation=35)
    fig.suptitle('S01: 230 development patients, one prediction per patient')
    fig.subplots_adjust(left=.14, right=.86, bottom=.25, top=.82, wspace=.12)
    colorbar_axis = fig.add_axes([.89, .30, .014, .45])
    fig.colorbar(im, cax=colorbar_axis, label='Pooled OOF patient recall')
    fig.savefig(out / 'oof_class_recall.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    analyze(args.run_id, args.output)
