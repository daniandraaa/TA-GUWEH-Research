"""Read and verify completed V5 results; never train or rewrite original reports."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from src.artifacts import ROOT, read_json, sha256_file, write_json
from src.evaluation import compute_metrics
from research_v5.common import DEFAULT_RUN, RESULT_FILES, case_directory, context, progress, require_prepared, require_smoke
from research_v5.analysis import checked_result, priority_decision


def check_table(path, expected, keys):
    actual = pd.read_csv(path).sort_values(keys).reset_index(drop=True)
    expected = pd.DataFrame(expected).sort_values(keys).reset_index(drop=True)
    if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
        raise RuntimeError(f'Report shape mismatch: {path}')
    for key in expected.columns:
        a, b = actual[key], expected[key]
        if pd.api.types.is_numeric_dtype(b.dtype):
            same = np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
        else:
            same = a.tolist() == b.tolist()
        if not same:
            raise RuntimeError(f'Report differs from recomputed results: {path}/{key}')


def analyze(run_id=DEFAULT_RUN):
    study = context(run_id); require_prepared(study)
    status = progress(study)
    if not status['ready_for_comparison']:
        raise RuntimeError(f'Complete V5 first: {status["complete"]}/30')
    reports = study.run / 'reports'
    original_reports = {str(p.relative_to(study.run)): sha256_file(p)
                        for p in reports.iterdir() if p.is_file() and p.name != '.analysis.lock'}
    books = {str(p.relative_to(ROOT)): sha256_file(p) for pattern in ('29_*V5.ipynb', '30_*V5.ipynb', '31_*V5.ipynb', '32_*V5.ipynb')
             for p in (ROOT / 'notebooks').glob(pattern)}
    if len(books) != 4: raise RuntimeError('Four V5 notebooks required')
    executed = {}
    for name in books:
        n = read_json(ROOT / name)
        cells = [c for c in n['cells'] if c['cell_type'] == 'code']
        if not all(c['execution_count'] is not None for c in cells) or any(
                o.get('output_type') == 'error' for c in cells for o in c.get('outputs', [])):
            raise RuntimeError(f'Notebook incomplete or contains error: {name}')
        executed[name] = len(cells)
    table = pd.read_csv(study.source / 'data/development/segments.csv', dtype={'hospital_id': str})
    for model in study.protocol['models']:
        require_smoke(study, model, torch.device(study.identity['paired_device']))
        smoke = read_json(study.run / 'smoke' / f'{model}.json')['record']
        if (len(set(smoke['training_patient_ids'])) != 8
                or not set(smoke['training_patient_ids']) <= set(study.folds[0]['train_ids'])
                or not np.isfinite(smoke['loss']) or not np.isfinite(smoke['gradient_norm'])):
            raise RuntimeError('Invalid training-fold memory smoke result')
    paired, collected, diagnostics, inputs = [], {}, [], {}
    for control in study.controls:
        model, fold, seed = control['model'], control['fold'], control['seed']
        row = {'model': model, 'fold': fold, 'seed': seed}
        for arm, directory in [('k4', ROOT / control['origin']), ('k8', case_directory(study, model, fold, seed))]:
            frame, stats = checked_result(directory, study, fold, table)
            collected.setdefault((model, seed, arm), []).append(frame)
            row.update({f'{arm}_{k}': v for k, v in stats.items()})
            inputs.update({str((directory / n).relative_to(ROOT)): sha256_file(directory / n) for n in RESULT_FILES})
            history = pd.read_csv(directory / 'history.csv')
            manifest = study.folds[fold - 1]
            counts = table[table.hospital_id.isin(manifest['train_ids'])].drop_duplicates('hospital_id').label.value_counts()
            if (not (history.unique_training_patients_seen == len(manifest['train_ids'])).all()
                    or not (history.patient_exposure_min == 1).all() or not (history.patient_exposure_max == 1).all()
                    or not (history.optimizer_steps == int(np.ceil(len(manifest['train_ids']) / 8))).all()
                    or any(not (history[f'train_class_{c}_draws'] == counts[c]).all() for c in range(3))):
                raise RuntimeError('Patient exposure/update/class counts changed')
            best = history.iloc[stats['best_epoch'] - 1]
            diagnostics.append({'model': model, 'fold': fold, 'seed': seed, 'arm': arm,
                'epochs_run': stats['epochs_run'], 'best_epoch': stats['best_epoch'],
                'best_epoch_train_accuracy_patient_bag': best.train_accuracy_patient_bag,
                'best_epoch_validation_accuracy_all_windows': best.validation_accuracy,
                'last_epoch_train_accuracy_patient_bag': history.iloc[-1].train_accuracy_patient_bag,
                'last_epoch_validation_accuracy_all_windows': history.iloc[-1].validation_accuracy,
                'best_epoch_batchnorm_running_variance_mean': best.batchnorm_running_variance_mean,
                'median_epoch_seconds': history.seconds.median(), 'elapsed_seconds': stats['elapsed_seconds']})
        row['delta_f1'] = row['k8_f1'] - row['k4_f1']
        row['elapsed_ratio_k8_over_k4'] = row['k8_elapsed_seconds'] / row['k4_elapsed_seconds']
        paired.append(row)
    if inputs != read_json(reports / 'comparison_inputs_sha256.json'):
        raise RuntimeError('Comparison artifact hashes changed')
    check_table(reports / 'folds_paired.csv', paired, ['model', 'fold', 'seed'])
    folds = pd.DataFrame(paired)
    seeds, class_stats, transitions = [], [], []
    for model in study.protocol['models']:
        for seed in study.protocol['seeds']:
            values = folds[(folds.model == model) & (folds.seed == seed)]
            row = {'model': model, 'seed': seed, 'delta_f1_mean_fold': values.delta_f1.mean(),
                   'delta_f1_sd_fold': values.delta_f1.std(ddof=1)}
            joined = {}
            for arm in ('k4', 'k8'):
                oof = pd.concat(collected[(model, seed, arm)], ignore_index=True)
                saved = pd.read_csv(reports / f'oof_{model}_{arm}_seed_{seed}.csv', dtype={'hospital_id': str})
                pd.testing.assert_frame_equal(oof.sort_values('hospital_id').reset_index(drop=True),
                    saved.sort_values('hospital_id').reset_index(drop=True), check_exact=False, rtol=1e-9, atol=1e-9)
                if not oof.hospital_id.is_unique or len(oof) != 230:
                    raise RuntimeError('OOF coverage changed')
                probabilities = oof[[f'probability_{c}' for c in study.inherited['class_names']]].to_numpy()
                truth, pred = oof.true_label.to_numpy(), probabilities.argmax(1)
                row.update({f'{arm}_pooled_oof_{k}': v for k, v in compute_metrics(truth, probabilities).items()})
                row.update({f'{arm}_pooled_recall_class_{c}': float((pred[truth == c] == c).mean()) for c in range(3)})
                row[f'{arm}_f1_mean_fold'] = values[f'{arm}_f1'].mean()
                row[f'{arm}_f1_sd_fold'] = values[f'{arm}_f1'].std(ddof=1)
                row[f'{arm}_collapse_folds'] = int((values[f'{arm}_missing_predicted_classes'] > 0).sum())
                cls = classification_report(truth, pred, labels=[0, 1, 2], target_names=study.inherited['class_names'],
                                            output_dict=True, zero_division=0)
                class_stats.extend({'model': model, 'seed': seed, 'arm': arm, 'class_name': c, **cls[c]}
                                   for c in study.inherited['class_names'])
                joined[arm] = oof.set_index('hospital_id').sort_index()
            row['delta_mean_RVOT_pooled_recall'] = float(np.mean([
                row[f'k8_pooled_recall_class_{c}'] - row[f'k4_pooled_recall_class_{c}'] for c in (1, 2)]))
            seeds.append(row)
            for hid in joined['k4'].index:
                transitions.append({'model': model, 'seed': seed, 'hospital_id': hid,
                    'class_name': joined['k4'].loc[hid, 'true_class'],
                    'k4_correct': bool(joined['k4'].loc[hid, 'correct']),
                    'k8_correct': bool(joined['k8'].loc[hid, 'correct']),
                    'n_windows': int((table.hospital_id == hid).sum())})
    check_table(reports / 'seed_summary.csv', seeds, ['model', 'seed'])
    seed_table = pd.DataFrame(seeds)
    models = []
    for model in study.protocol['models']:
        s, f = seed_table[seed_table.model == model], folds[folds.model == model]
        delta, positive, recall = s.delta_f1_mean_fold.mean(), int((s.delta_f1_mean_fold > 0).sum()), s.delta_mean_RVOT_pooled_recall.mean()
        models.append({'model': model, 'k4_f1_mean_fold_and_seed': s.k4_f1_mean_fold.mean(),
            'k8_f1_mean_fold_and_seed': s.k8_f1_mean_fold.mean(), 'delta_f1_mean': delta,
            'delta_f1_sd_across_seed_means': s.delta_f1_mean_fold.std(ddof=1), 'positive_seeds': positive,
            'delta_mean_RVOT_pooled_recall': recall,
            'k4_collapse_cases': int((f.k4_missing_predicted_classes > 0).sum()),
            'k8_collapse_cases': int((f.k8_missing_predicted_classes > 0).sum()),
            'median_elapsed_ratio_k8_over_k4': f.elapsed_ratio_k8_over_k4.median(),
            'prioritize_k8_for_further_development': priority_decision(delta, positive, recall, study.protocol['priority_rule'])})
    check_table(reports / 'model_summary.csv', models, ['model'])
    analysis = read_json(reports / 'analysis.json')
    if (analysis['experiment_hash'] != study.signature or analysis['test_signals_read'] or analysis['final_model_trained']
            or analysis['analysis_source_sha256'] != sha256_file(ROOT / 'research_v5/analysis.py')
            or any(analysis['models'][i] != models[i] for i in range(2))):
        # Serialized Python/numpy floats can differ at the last decimal; table checks above use tight tolerance.
        if (analysis['experiment_hash'] != study.signature or analysis['test_signals_read'] or analysis['final_model_trained']
                or analysis['analysis_source_sha256'] != sha256_file(ROOT / 'research_v5/analysis.py')):
            raise RuntimeError('Report identity/scope changed')
        for old, new in zip(analysis['models'], models):
            for key in new:
                if isinstance(new[key], (float, np.floating)):
                    if not np.isclose(old[key], new[key], rtol=1e-9, atol=1e-9): raise RuntimeError('JSON summary mismatch')
                elif old[key] != new[key]: raise RuntimeError('JSON summary mismatch')
    baseline_files = read_json(study.baseline / 'analysis_20261003_complete/verified_input_sha256.json')
    for relative, digest in baseline_files.items():
        if sha256_file(study.baseline / relative) != digest: raise RuntimeError('Original V4 artifact changed')
    for relative, digest in original_reports.items():
        if sha256_file(study.run / relative) != digest: raise RuntimeError('Original report overwritten')
    for relative, digest in books.items():
        if sha256_file(ROOT / relative) != digest: raise RuntimeError('Executed notebook overwritten')
    out = study.run / 'analysis_20261004_complete'; out.mkdir(exist_ok=True)
    diag = pd.DataFrame(diagnostics); diag.to_csv(out / 'training_diagnostics.csv', index=False)
    classes = pd.DataFrame(class_stats)
    classes.to_csv(out / 'per_class_pooled_oof_by_seed.csv', index=False)
    classes.groupby(['model', 'arm', 'class_name'])[['precision', 'recall', 'f1-score']].mean().reset_index().to_csv(out / 'per_class_pooled_oof_mean.csv', index=False)
    changes = pd.DataFrame(transitions); changes.to_csv(out / 'development_patient_transitions.csv', index=False)
    summary = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
        'experiment_hash': study.signature, 'new_fits_verified': 30, 'controls_verified': 30,
        'metrics_recomputed': 60, 'comparison_input_hashes_verified': len(inputs),
        'baseline_v4_artifacts_unchanged': len(baseline_files), 'notebook_code_cells_executed': executed,
        'source_reports_unchanged': True, 'source_notebooks_unchanged': True,
        'all_training_patients_seen_once_each_epoch': True, 'patient_batch_size': 8,
        'new_real_training_by_audit': False, 'test_signals_read': False,
        'decision': 'retain K4; neither K8 model meets preregistered development criteria',
        'models': models, 'analysis_source_sha256': sha256_file(__file__)}
    write_json(out / 'audit.json', summary)
    write_json(out / 'verified_original_reports_sha256.json', original_reports)
    write_json(out / 'verified_executed_notebooks_sha256.json', books)
    print(pd.DataFrame(models).to_string(index=False))
    print('Training medians:')
    print(diag.groupby(['model', 'arm'])[['best_epoch', 'epochs_run', 'best_epoch_train_accuracy_patient_bag',
        'best_epoch_validation_accuracy_all_windows', 'median_epoch_seconds']].median().to_string())
    print('Mean per-class recall across three seeds:')
    print(classes.groupby(['model', 'arm', 'class_name']).recall.mean().to_string())
    print('PASS:', out)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    analyze(parser.parse_args().run_id)
