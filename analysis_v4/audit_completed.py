"""Verify saved complete V4 results; no training or ECG signal inference.

Run: .venv/bin/python -m analysis_v4.audit_completed
Outputs are separate from the original run reports and fitted artefacts.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.artifacts import ROOT, canonical_hash, read_json, sha256_file, write_json
from research_v4.common import DEFAULT_RUN, context, cv_directory, require_final_complete, scenarios
from research_v4_mps_compat.runtime import audit_run, build_model, RECORD
from analysis_v3.analyze_completed import recompute, require
from analysis_v3.analyze_final import bootstrap_scores


def checked_predictions(path, expected, classes):
    frame = pd.read_csv(path, dtype={'hospital_id': str})
    require(frame.hospital_id.is_unique and set(frame.hospital_id) == set(expected),
            f'Patient coverage changed: {path}')
    frame = frame.set_index('hospital_id').loc[sorted(expected)].reset_index()
    truth = np.asarray([expected[h] for h in frame.hospital_id])
    require(np.array_equal(frame.true_label, truth), f'Labels changed: {path}')
    require(frame.true_class.tolist() == [classes[k] for k in truth], f'Class names changed: {path}')
    probabilities = frame[[f'probability_{c}' for c in classes]].to_numpy()
    require(np.isfinite(probabilities).all() and (probabilities >= -1e-7).all()
            and (probabilities <= 1 + 1e-7).all() and np.allclose(probabilities.sum(1), 1, atol=2e-6),
            f'Invalid probabilities: {path}')
    prediction = probabilities.argmax(1)
    require(frame.predicted_class.tolist() == [classes[k] for k in prediction]
            and np.array_equal(frame.correct, truth == prediction), f'Prediction metadata changed: {path}')
    return frame, truth, probabilities


def checked_case(path, signature):
    case = read_json(path / 'config.json')
    require(case['case_hash'] == canonical_hash({k: v for k, v in case.items() if k != 'case_hash'}),
            f'Case signature changed: {path}')
    require(case['experiment_hash'] == signature, f'Base experiment changed: {path}')
    return case


def check_pooling_weights(path, expected, segments, pooling):
    weights = pd.read_csv(path, dtype={'hospital_id': str})
    expected_indices = np.flatnonzero(segments.hospital_id.isin(expected).to_numpy())
    require(weights.segment_index.is_unique and set(weights.segment_index) == set(expected_indices),
            f'Evaluation did not use all patient segments: {path}')
    linked = segments.iloc[weights.segment_index.to_numpy()]
    require(np.array_equal(linked.hospital_id.to_numpy(), weights.hospital_id.to_numpy()),
            f'Segment ownership changed: {path}')
    require(np.isfinite(weights.weight).all() and (weights.weight >= 0).all(), f'Invalid weights: {path}')
    grouped = weights.groupby('hospital_id')
    require(set(grouped.groups) == set(expected) and np.allclose(grouped.weight.sum(), 1, atol=2e-6),
            f'Patient pooling weights changed: {path}')
    counts = weights.hospital_id.map(grouped.size()).to_numpy()
    require(np.array_equal(counts, weights.n_patient_segments), f'Segment count mismatch: {path}')
    if pooling == 'mean':
        require(np.allclose(weights.weight, 1 / counts, atol=2e-6), f'Non-uniform mean pooling: {path}')


def check_compatibility(run, directory, record):
    identity = {'run_compatibility_hash': record['compatibility_hash'],
                'case_path': str(directory.relative_to(run))}
    saved = read_json(directory / RECORD)
    require(saved['identity'] == identity and saved['compatibility_hash'] == canonical_hash(identity),
            f'LeadTime backend provenance mismatch: {directory}')


def analyze(run_id=DEFAULT_RUN, output=None):
    torch.set_num_threads(1)
    run, source, old, cfg, config = context(run_id)
    require_final_complete(run, cfg)
    signature = read_json(run / 'protocol.json')['experiment_hash']
    compatibility = audit_run(run)
    for name, digest in compatibility['preserved_artifact_sha256_at_activation'].items():
        require(sha256_file(run / name) == digest, f'Pre-patch result changed: {name}')
    out = Path(output).resolve() if output else run / 'analysis_20261003_complete'
    out.mkdir(parents=True, exist_ok=True)
    require(out != run and out != run / 'reports', 'Use a separate derived output directory')
    records, folds = read_json(run / 'data/patients.json'), read_json(run / 'folds/manifest.json')
    classes = config['class_names']
    cohorts = {pool: {str(r['hospital_id']): r['label'] for r in records if r['pool'] == pool}
               for pool in ('development', 'test')}
    development, historical = cohorts['development'], cohorts['test']
    require(len(development) == 230 and len(historical) == 59 and not set(development) & set(historical),
            'Default cohort changed')
    require(sorted(h for f in folds for h in f['validation_ids']) == sorted(development), 'OOF fold coverage changed')
    segments = {pool: pd.read_csv(source / 'data' / pool / 'segments.csv', dtype={'hospital_id': str})
                for pool in cohorts}
    for fold in folds:
        require(not set(fold['train_ids']) & set(fold['validation_ids'])
                and set(fold['train_ids']) | set(fold['validation_ids']) == set(development), 'Fold leakage/cohort changed')
    for scope, ids in [(f'fold_{f["fold"]}', f['train_ids']) for f in folds] + [('final', sorted(development))]:
        marker = read_json(run / 'scalers' / scope / 'provenance.json')
        indices = np.flatnonzero(segments['development'].hospital_id.isin(ids).to_numpy())
        require(marker['sha256'] == sha256_file(run / 'scalers' / scope / 'scaler.joblib')
                and marker['spec']['training_patient_ids'] == sorted(ids)
                and marker['spec']['indices_hash'] == canonical_hash(indices.tolist())
                and marker['spec']['n_segments'] == len(indices), f'Scaler provenance changed: {scope}')
    test_cache = read_json(run / 'test_cache.json')
    require(test_cache == {n: sha256_file(source / 'data/test' / n) for n in test_cache}, 'Historical cache changed')
    selections = read_json(run / 'selection.json')
    require(selections['test_used_for_selection'] is False and selections['primary_seed'] == cfg['primary_seed'],
            'Selection policy changed')
    max_error, rows, inputs = 0., [], {}
    for name in ('protocol.json', 'selection.json', 'test_cache.json', RECORD,
                 'data/patients.json', 'folds/manifest.json'):
        inputs[name] = sha256_file(run / name)
    for p in (run / 'scalers').rglob('*'):
        if p.is_file() and p.suffix in {'.json', '.joblib'}:
            inputs[str(p.relative_to(run))] = sha256_file(p)

    def record_inputs(directory):
        for p in directory.iterdir():
            if p.is_file() and p.suffix in {'.json', '.csv', '.pth'}:
                inputs[str(p.relative_to(run))] = sha256_file(p)

    def verify_metrics(directory, expected, phase, model, seed, fold=None, pooling='mean'):
        nonlocal max_error
        fname = 'patient_predictions.csv' if phase == 'test' else 'validation_predictions.csv'
        frame, truth, probs = checked_predictions(directory / fname, expected, classes)
        computed, result = recompute(truth, probs), read_json(directory / 'metrics.json')
        for k, value in computed.items():
            error = abs(value - result['metrics'][k]); max_error = max(max_error, error)
            require(error < 1e-7, f'Metric changed: {directory}/{k}')
        require(np.array_equal(result['confusion_matrix'], confusion_matrix(truth, probs.argmax(1), labels=[0, 1, 2])),
                f'Confusion matrix changed: {directory}')
        report = classification_report(truth, probs.argmax(1), labels=[0, 1, 2], target_names=classes,
                                       zero_division=0, output_dict=True)
        for cls in classes:
            for k, value in report[cls].items():
                require(abs(value - result['classification_report'][cls][k]) < 1e-7, f'Class metric changed: {directory}')
        count_field = 'n_patients' if phase == 'test' else 'n_validation_patients'
        require(result[count_field] == len(expected), f'Patient count changed: {directory}')
        weight_name = 'segment_weights.csv' if phase == 'test' else 'validation_segment_weights.csv'
        check_pooling_weights(directory / weight_name, expected, segments['test' if phase == 'test' else 'development'], pooling)
        if phase != 'test':
            history = pd.read_csv(directory / 'history.csv')
            require(len(history) == result['epochs_run'] and history.epoch.tolist() == list(range(1, len(history) + 1)),
                    f'Epoch history changed: {directory}')
            best = int(history.loc[history.validation_f1.idxmax(), 'epoch'])
            require(best == result['best_epoch'] and abs(history.validation_f1.max() - computed['f1']) < 1e-7,
                    f'Best epoch mismatch: {directory}')
            if phase != 'diagnostic':
                case = read_json(directory / 'config.json')
                require((history.unique_training_patients_seen == len(case['train_patient_ids'])).all()
                        and (history.patient_exposure_min == 1).all() and (history.patient_exposure_max == 1).all(),
                        f'Patient training exposure changed: {directory}')
        rows.append({'phase': phase, 'model': model, 'seed': seed, 'fold': fold, 'pooling': pooling,
                     'best_epoch': result.get('best_epoch'), **computed})
        record_inputs(directory)
        return frame, computed

    cv_rows, reconstructions = [], []
    for model in cfg['models']:
        for pooling in cfg['pooling']:
            for sc in scenarios(cfg):
                cases = []
                for f in folds:
                    directory = cv_directory(run, model, pooling, sc['id'], f['fold'], cfg['cv_seed'])
                    case = checked_case(directory, signature)
                    require(case['model'] == model and case['pooling'] == pooling and case['scenario'] == sc
                            and case['fold'] == f['fold'] and case['seed'] == cfg['cv_seed']
                            and case['train_patient_ids'] == sorted(f['train_ids'])
                            and case['validation_patient_ids'] == sorted(f['validation_ids']), 'CV configuration changed')
                    require(case['settings'] == {**cfg['training'], **sc, 'scheduler_t_max': 100}
                            and case['training_mode'] == 'bag' and case['segment_budget'] == sc['batch_size']
                            and case['patient_batch_size'] == sc['batch_size'] // cfg['segments_per_patient'],
                            'CV optimizer/batch recipe mismatch')
                    if model == 'lead_time': check_compatibility(run, directory, compatibility)
                    _, values = verify_metrics(directory, {h: development[h] for h in f['validation_ids']},
                                               'cv', model, cfg['cv_seed'], f['fold'], pooling)
                    result = read_json(directory / 'metrics.json')
                    row = {**rows[-1], 'scenario': sc['id'], 'epochs_run': result['epochs_run'],
                           'missing_predicted_classes': result['validation_missing_predicted_classes']}
                    cv_rows.append(row); cases.append(row)
                reconstructions.append({'model': model, 'pooling': pooling, 'scenario': sc['id'],
                    'learning_rate': sc['learning_rate'], 'batch_size': sc['batch_size'],
                    'patient_batch_size': sc['batch_size'] // cfg['segments_per_patient'],
                    'f1_mean': np.mean([c['f1'] for c in cases]), 'f1_std': np.std([c['f1'] for c in cases], ddof=1),
                    'accuracy_mean': np.mean([c['accuracy'] for c in cases]),
                    'epochs': int(np.median([c['best_epoch'] for c in cases])),
                    'collapse_folds': sum(c['missing_predicted_classes'] > 0 for c in cases)})
    expected_choices = [sorted([c for c in reconstructions if c['model'] == m],
                              key=lambda c: (-c['f1_mean'], c['f1_std'], c['pooling'], c['scenario']))[0]
                        for m in cfg['models']]
    for expected, saved in zip(expected_choices, selections['choices']):
        require(expected.keys() == saved.keys() and all(np.isclose(v, saved[k], atol=1e-10)
                if isinstance(v, (int, float, np.number)) else v == saved[k] for k, v in expected.items()),
                'Frozen choice does not match independently recomputed CV')
    primary = sorted(expected_choices, key=lambda c: (-c['f1_mean'], c['f1_std'], c['model']))[0]
    require(primary['model'] == selections['primary']['model'], 'Primary ranking changed')
    for model in cfg['models'][:3]:
        for f in folds:
            for seed in cfg['diagnostic_seeds']:
                directory = run / 'diagnostic/Adam' / model / f'fold_{f["fold"]}' / f'seed_{seed}'
                checked_case(directory, signature)
                verify_metrics(directory, {h: development[h] for h in f['validation_ids']},
                               'diagnostic', model, seed, f['fold'])
    reused = 0
    for choice in expected_choices:
        model = choice['model']
        for f in folds:
            for seed in cfg['stability_seeds']:
                directory = run / 'stability' / model / f'fold_{f["fold"]}' / f'seed_{seed}'
                binding = read_json(directory / 'config.json')
                if seed == cfg['cv_seed']:
                    origin = cv_directory(run, model, choice['pooling'], choice['scenario'], f['fold'], seed)
                    require(binding['reused_cv_case'] == str(origin.relative_to(run))
                            and binding['source_config_sha256'] == sha256_file(origin / 'config.json')
                            and binding['source_metrics_sha256'] == sha256_file(origin / 'metrics.json')
                            and read_json(directory / 'metrics.json') == read_json(origin / 'metrics.json'), 'Reused CV changed')
                    reused += 1; record_inputs(directory)
                else:
                    case = checked_case(directory, signature)
                    sc = next(s for s in scenarios(cfg) if s['id'] == choice['scenario'])
                    require(case['model'] == model and case['pooling'] == choice['pooling']
                            and case['scenario'] == sc and case['seed'] == seed and case['fold'] == f['fold']
                            and case['train_patient_ids'] == sorted(f['train_ids'])
                            and case['validation_patient_ids'] == sorted(f['validation_ids'])
                            and case['settings'] == {**cfg['training'], **sc, 'scheduler_t_max': 100},
                            'Stability configuration mismatch')
                    if model == 'lead_time': check_compatibility(run, directory, compatibility)
                    verify_metrics(directory, {h: development[h] for h in f['validation_ids']},
                                   'stability', model, seed, f['fold'], choice['pooling'])
    checkpoints, test_frames, oof_frames = [], {}, {}
    for choice in expected_choices:
        model = choice['model']
        network = build_model(model, choice['pooling'], config['model'], cfg['new_model']).cpu().eval()
        for seed in cfg['final_seeds']:
            directory = run / 'final' / model / f'seed_{seed}'
            case, fit = checked_case(directory, signature), read_json(directory / 'metrics.json')
            require(case['train_patient_ids'] == sorted(development) and case['validation_patient_ids'] == []
                    and case['model'] == model and case['pooling'] == choice['pooling']
                    and case['scenario']['id'] == choice['scenario'] and case['fold'] is None and case['seed'] == seed
                    and fit['fit_complete'] and fit['epochs_run'] == fit['best_epoch'] == choice['epochs'] == case['settings']['epochs'],
                    'Final recipe/cohort mismatch')
            sc = next(s for s in scenarios(cfg) if s['id'] == choice['scenario'])
            require(case['settings'] == {**cfg['training'], **sc, 'scheduler_t_max': 100, 'epochs': choice['epochs']},
                    'Final optimizer/learning-rate/batch mismatch')
            if model == 'lead_time': check_compatibility(run, directory, compatibility)
            history = pd.read_csv(directory / 'history.csv')
            require(len(history) == choice['epochs'] and not any(c.startswith('validation_') for c in history.columns),
                    'Final history includes validation or wrong epochs')
            require((history.unique_training_patients_seen == len(development)).all()
                    and (history.patient_exposure_min == 1).all() and (history.patient_exposure_max == 1).all(),
                    'Final patient exposure changed')
            weight_path = directory / 'model.pth'; digest = sha256_file(weight_path)
            require(digest == fit['model_sha256'], 'Final checkpoint hash changed')
            state = torch.load(weight_path, map_location='cpu', weights_only=True)
            require(all(torch.isfinite(v).all().item() for v in state.values()), 'Nonfinite checkpoint')
            network.load_state_dict(state, strict=True); del state
            checkpoints.append({'model': model, 'seed': seed, 'sha256': digest, 'strict_load_valid': True})
            record_inputs(directory)
            test_dir = run / 'test' / model / f'seed_{seed}'
            test_case = checked_case(test_dir, signature)
            require(test_case['final_model_sha256'] == digest and test_case['selection_hash'] == canonical_hash(selections)
                    and test_case['test_cache'] == test_cache, 'Test provenance changed')
            frame, _ = verify_metrics(test_dir, historical, 'test', model, seed, pooling=choice['pooling'])
            test_frames[(model, seed)] = frame
        selected = pd.concat([pd.read_csv(cv_directory(run, model, choice['pooling'], choice['scenario'], f['fold'], cfg['cv_seed'])
                            / 'validation_predictions.csv', dtype={'hospital_id': str}) for f in folds])
        require(selected.hospital_id.is_unique and set(selected.hospital_id) == set(development), 'Selected OOF incomplete')
        oof_frames[model] = selected.set_index('hospital_id').loc[sorted(development)]
        del network
    test = pd.DataFrame([r for r in rows if r['phase'] == 'test'])
    order, primary_seed = sorted(historical), cfg['primary_seed']
    truth = np.array([historical[h] for h in order])
    predictions = {m: test_frames[(m, primary_seed)][[f'probability_{c}' for c in classes]].to_numpy().argmax(1) for m in cfg['models']}
    boot = bootstrap_scores(truth, predictions, repeats=5000, seed=20261003)
    overview, paired, seed_stability, class_rows, comparisons = [], [], [], [], []
    for choice in expected_choices:
        model = choice['model']; all_seeds = test[test.model == model]
        seed42 = all_seeds[all_seeds.seed == primary_seed].iloc[0]
        low, high = np.quantile(boot[model], [.025, .975])
        overview.append({**choice, 'primary_model': model == primary['model'],
            'test_f1_seed42': seed42.f1, 'test_accuracy_seed42': seed42.accuracy, 'test_auc_seed42': seed42.auc,
            'test_f1_five_seed_mean': all_seeds.f1.mean(), 'test_f1_five_seed_sd': all_seeds.f1.std(ddof=1),
            'test_f1_ci_low': low, 'test_f1_ci_high': high})
        if model != primary['model']:
            delta = boot[model] - boot[primary['model']]; ci = np.quantile(delta, [.025, .975])
            paired.append({'model': model, 'reference': primary['model'], 'delta_f1': seed42.f1 - test[test.model == primary['model']].query('seed == 42').iloc[0].f1,
                           'ci_low': ci[0], 'ci_high': ci[1], 'posthoc_fixed_model_historical_test': True})
        array = np.stack([test_frames[(model, s)][[f'probability_{c}' for c in classes]].to_numpy().argmax(1) for s in cfg['final_seeds']])
        changed = np.any(array != array[0], axis=0)
        seed_stability.append({'model': model, 'n_patients': len(truth), 'prediction_changed_across_seeds': int(changed.sum()),
                               'fraction_changed': changed.mean()})
        for seed in cfg['final_seeds']:
            frame = test_frames[(model, seed)]; pred = frame[[f'probability_{c}' for c in classes]].to_numpy().argmax(1)
            report = classification_report(truth, pred, labels=[0, 1, 2], target_names=classes, zero_division=0, output_dict=True)
            class_rows.extend({'model': model, 'seed': seed, 'class': c, **report[c]} for c in classes)
        if model in cfg['models'][:3]:
            prior = [read_json(old / 'test' / model / f'seed_{s}' / 'metrics.json')['metrics']['f1'] for s in cfg['final_seeds']]
            comparisons.append({'model': model, 'v3_seed42_f1': prior[0], 'v4_seed42_f1': seed42.f1,
                'seed42_delta': seed42.f1 - prior[0], 'v3_five_seed_mean': np.mean(prior),
                'v4_five_seed_mean': all_seeds.f1.mean(), 'five_seed_mean_delta': all_seeds.f1.mean() - np.mean(prior),
                'isolates_single_component': False})
    development_table = pd.DataFrame([r for r in records if r['pool'] == 'development']).set_index('hospital_id').loc[sorted(development)]
    wrong = np.zeros(len(development_table), dtype=int)
    for model, frame in oof_frames.items():
        pred = frame[[f'probability_{c}' for c in classes]].to_numpy().argmax(1)
        development_table[f'{model}_prediction'] = [classes[k] for k in pred]
        wrong += pred != frame.true_label.to_numpy()
    development_table['n_wrong_selected_models'] = wrong
    development_table = development_table.reset_index().sort_values(['n_wrong_selected_models', 'hospital_id'], ascending=[False, True])
    subgroup = development_table.groupby(['class_name', 'sublocation'], dropna=False).agg(
        n_patients=('hospital_id', 'size'), wrong_all_five=('n_wrong_selected_models', lambda x: int((x == 5).sum())),
        wrong_at_least_four=('n_wrong_selected_models', lambda x: int((x >= 4).sum()))).reset_index()
    outputs = {'model_overview': pd.DataFrame(overview), 'independent_cv_folds': pd.DataFrame(cv_rows),
               'independent_all_validation_test_metrics': pd.DataFrame(rows), 'checkpoint_inventory': pd.DataFrame(checkpoints),
               'paired_seed42_historical_bootstrap': pd.DataFrame(paired), 'test_prediction_seed_stability': pd.DataFrame(seed_stability),
               'test_per_class_all_seeds': pd.DataFrame(class_rows), 'v3_v4_historical_comparison': pd.DataFrame(comparisons),
               'development_oof_error_triage': development_table, 'development_sublocation_error_summary': subgroup}
    for name, frame in outputs.items(): frame.to_csv(out / f'{name}.csv', index=False)
    audit = {'audited_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id, 'base_experiment_hash': signature,
        'compatibility_hash': compatibility['compatibility_hash'], 'cv_metrics_recomputed': 450,
        'diagnostic_metrics_recomputed': 45, 'stability_new_fits_recomputed': 50, 'stability_reused_cv_verified': reused,
        'final_checkpoints_strict_loaded': len(checkpoints), 'test_metrics_recomputed': len(test), 'maximum_metric_error': max_error,
        'development_patients': len(development), 'historical_test_patients': len(historical), 'test_class_counts': np.bincount(truth).tolist(),
        'primary_model': primary['model'], 'primary_seed': primary_seed, 'selection_reconstructed_from_cv': True,
        'bootstrap_repeats': 5000, 'bootstrap_seed': 20261003, 'bootstrap_scope': 'stratified patient resampling, fixed seed42 checkpoints; not training-seed uncertainty',
        'development_wrong_all_five_models': int((wrong == 5).sum()), 'development_wrong_at_least_four_models': int((wrong >= 4).sum()),
        'checkpoint_predictions_rerun': False, 'ecg_signals_loaded_for_analysis': False, 'real_training_by_analysis': False,
        'test_is_historical_not_external': True, 'selection_cv_is_nested': False,
        'audit_source_sha256': sha256_file(Path(__file__)), 'verified_input_files': len(inputs)}
    audit['analysis_dependency_sha256'] = {
        name: sha256_file(ROOT / 'analysis_v3' / name)
        for name in ('analyze_completed.py', 'analyze_final.py')}
    write_json(out / 'verified_input_sha256.json', inputs); write_json(out / 'final_audit.json', audit)
    make_figure(out, outputs['model_overview'], test_frames, classes, primary['model'])
    print('AUDIT PASS:', {k: audit[k] for k in ('cv_metrics_recomputed', 'final_checkpoints_strict_loaded', 'test_metrics_recomputed', 'maximum_metric_error')})
    print(outputs['model_overview'][['model', 'f1_mean', 'test_f1_seed42', 'test_f1_five_seed_mean', 'test_f1_five_seed_sd']].to_string(index=False))
    print('Development errors across all five models:', int((wrong == 5).sum()), '; at least four:', int((wrong >= 4).sum()))
    print('Outputs:', out)
    return audit


def make_figure(out, overview, test_frames, classes, primary):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    positions = np.arange(len(overview))
    axes[0].errorbar(positions - .12, overview.f1_mean, yerr=overview.f1_std, fmt='o', capsize=3, label='Selected development CV: mean ± fold SD')
    axes[0].errorbar(positions + .12, overview.test_f1_five_seed_mean, yerr=overview.test_f1_five_seed_sd, fmt='s', capsize=3, label='Historical test: mean ± seed SD')
    axes[0].scatter(positions + .12, overview.test_f1_seed42, marker='*', color='black', s=90, label='Historical test: frozen seed 42', zorder=3)
    axes[0].set_xticks(positions, overview.model, rotation=20, ha='right'); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Patient macro-F1'); axes[0].legend(fontsize=8); axes[0].set_title('V4 completed results; CV is selection optimistic')
    frame = test_frames[(primary, 42)]; pred = frame[[f'probability_{c}' for c in classes]].to_numpy().argmax(1)
    cm = confusion_matrix(frame.true_label, pred, labels=[0, 1, 2]); axes[1].imshow(cm, cmap='Blues')
    short = ['LVOT', 'RVOT cusp', 'RVOT septal/freewall/other']
    axes[1].set_xticks(range(3), short, rotation=20, ha='right'); axes[1].set_yticks(range(3), short)
    for (i, j), count in np.ndenumerate(cm):
        axes[1].text(j, i, str(count), ha='center', va='center',
                     color='white' if count > cm.max() / 2 else 'black')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True'); axes[1].set_title(f'Frozen primary: {primary}, seed 42; 59 patients')
    fig.tight_layout(); fig.savefig(out / 'v4_completed_results.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=DEFAULT_RUN); parser.add_argument('--output')
    args = parser.parse_args(); analyze(args.run_id, args.output)
