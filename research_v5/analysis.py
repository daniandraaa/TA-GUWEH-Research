"""Paired development analysis after all 30 candidates; no test endpoint."""
import os
import numpy as np
import pandas as pd
from src.artifacts import ROOT, lock, read_json, sha256_file, write_json
from src.evaluation import compute_metrics
from .common import RESULT_FILES, case_directory, progress, require_prepared


def checked_result(directory, study, fold, table):
    result = read_json(directory / 'metrics.json')
    frame = pd.read_csv(directory / 'validation_predictions.csv', dtype={'hospital_id': str})
    expected = {r['hospital_id']: r['label'] for r in study.records
                if r['hospital_id'] in study.folds[fold - 1]['validation_ids']}
    if not frame.hospital_id.is_unique or set(frame.hospital_id) != set(expected):
        raise RuntimeError('Validation patient coverage changed')
    truth = np.asarray([expected[h] for h in frame.hospital_id])
    classes = study.inherited['class_names']
    probabilities = frame[[f'probability_{c}' for c in classes]].to_numpy()
    if (not np.array_equal(frame.true_label, truth) or not np.isfinite(probabilities).all()
            or (probabilities < -1e-7).any() or (probabilities > 1 + 1e-7).any()
            or not np.allclose(probabilities.sum(1), 1, atol=2e-6)):
        raise RuntimeError('Invalid validation labels/probabilities')
    pred = probabilities.argmax(1)
    if (frame.true_class.tolist() != [classes[c] for c in truth]
            or frame.predicted_class.tolist() != [classes[c] for c in pred]
            or not np.array_equal(frame.correct, pred == truth)):
        raise RuntimeError('Prediction metadata changed')
    computed = compute_metrics(truth, probabilities)
    for key, value in computed.items():
        if not np.isclose(value, result['metrics'][key], atol=1e-9, rtol=1e-9):
            raise RuntimeError(f'Saved {key} differs from predictions: {directory}')
    recall = {f'recall_class_{c}': float((pred[truth == c] == c).mean()) for c in range(3)}
    for c in range(3):
        if not np.isclose(recall[f'recall_class_{c}'], result[f'validation_recall_class_{c}']):
            raise RuntimeError('Saved class recall differs from predictions')
    history = pd.read_csv(directory / 'history.csv')
    if (history.epoch.tolist() != list(range(1, len(history) + 1))
            or result['epochs_run'] != len(history)
            or result['best_epoch'] != int(history.loc[history.validation_f1.idxmax(), 'epoch'])
            or not np.isclose(history.validation_f1.max(), computed['f1'])):
        raise RuntimeError('Epoch selection/history mismatch')
    weights = pd.read_csv(directory / 'validation_segment_weights.csv', dtype={'hospital_id': str})
    expected_indices = np.flatnonzero(table.hospital_id.isin(expected).to_numpy())
    if (not weights.segment_index.is_unique or set(weights.segment_index) != set(expected_indices)
            or not np.isfinite(weights.weight).all() or (weights.weight < 0).any()
            or not np.array_equal(table.iloc[weights.segment_index].hospital_id.to_numpy(), weights.hospital_id)):
        raise RuntimeError('Evaluation must include all windows for each validation patient')
    groups = weights.groupby('hospital_id')
    if set(groups.groups) != set(expected) or not np.allclose(groups.weight.sum(), 1, atol=2e-6):
        raise RuntimeError('Invalid global patient pooling weights')
    count = weights.hospital_id.map(groups.size())
    if not np.array_equal(count, weights.n_patient_segments):
        raise RuntimeError('Pooling window counts changed')
    model = read_json(directory / 'config.json')['model']
    if study.choices[model]['pooling'] == 'mean' and not np.allclose(weights.weight, 1 / count, atol=2e-6):
        raise RuntimeError('Expected uniform mean pooling')
    return frame, {**computed, **recall, 'best_epoch': result['best_epoch'],
        'epochs_run': result['epochs_run'], 'elapsed_seconds': result['elapsed_seconds'],
        'missing_predicted_classes': int(sum(not (pred == c).any() for c in range(3)))}


def priority_decision(delta_mean, positive_seeds, rvot_recall_delta, rules):
    return bool(delta_mean > 0 and positive_seeds >= rules['minimum_positive_seeds'] and rvot_recall_delta >= 0)


def analyze(study):
    require_prepared(study)
    status = progress(study)
    if not status['ready_for_comparison']:
        raise RuntimeError(f'Perbandingan terkunci: {status["complete"]}/30 selesai. '
                           f'Lengkapi notebook30/31. Missing: {status["missing"][:5]}')
    out = study.run / 'reports'; out.mkdir(exist_ok=True)
    with lock(out / '.analysis.lock'):
        table = pd.read_csv(study.source / 'data/development/segments.csv', dtype={'hospital_id': str})
        paired, collected, inputs = [], {}, {}
        for control in study.controls:
            model, fold, seed = control['model'], control['fold'], control['seed']
            origin, candidate = ROOT / control['origin'], case_directory(study, model, fold, seed)
            row = {'model': model, 'fold': fold, 'seed': seed}
            for arm, directory in [('k4', origin), ('k8', candidate)]:
                frame, stats = checked_result(directory, study, fold, table)
                collected.setdefault((model, seed, arm), []).append(frame)
                row.update({f'{arm}_{key}': value for key, value in stats.items()})
                inputs.update({str((directory / name).relative_to(ROOT)): sha256_file(directory / name)
                               for name in RESULT_FILES})
            row['delta_f1'] = row['k8_f1'] - row['k4_f1']
            row['elapsed_ratio_k8_over_k4'] = row['k8_elapsed_seconds'] / row['k4_elapsed_seconds']
            paired.append(row)
        marker = out / 'comparison_inputs_sha256.json'
        if marker.exists() and read_json(marker) != inputs:
            raise RuntimeError('Results changed after the first complete comparison; preserve the old report')
        write_json(marker, inputs)
        folds = pd.DataFrame(paired)
        folds.to_csv(out / 'folds_paired.csv', index=False)
        seeds = []
        for model in study.protocol['models']:
            for seed in study.protocol['seeds']:
                values = folds[(folds.model == model) & (folds.seed == seed)]
                row = {'model': model, 'seed': seed,
                    'delta_f1_mean_fold': float(values.delta_f1.mean()),
                    'delta_f1_sd_fold': float(values.delta_f1.std(ddof=1))}
                for arm in ('k4', 'k8'):
                    oof = pd.concat(collected[(model, seed, arm)], ignore_index=True)
                    if not oof.hospital_id.is_unique or set(oof.hospital_id) != {r['hospital_id'] for r in study.records}:
                        raise RuntimeError('Pooled OOF must contain exactly230 unique development patients')
                    oof.to_csv(out / f'oof_{model}_{arm}_seed_{seed}.csv', index=False)
                    p = oof[[f'probability_{c}' for c in study.inherited['class_names']]].to_numpy()
                    truth, pred = oof.true_label.to_numpy(), p.argmax(1)
                    row.update({f'{arm}_pooled_oof_{k}': v for k, v in compute_metrics(truth, p).items()})
                    row.update({f'{arm}_pooled_recall_class_{c}': float((pred[truth == c] == c).mean()) for c in range(3)})
                    row[f'{arm}_f1_mean_fold'] = float(values[f'{arm}_f1'].mean())
                    row[f'{arm}_f1_sd_fold'] = float(values[f'{arm}_f1'].std(ddof=1))
                    row[f'{arm}_collapse_folds'] = int((values[f'{arm}_missing_predicted_classes'] > 0).sum())
                row['delta_mean_RVOT_pooled_recall'] = float(np.mean([
                    row[f'k8_pooled_recall_class_{c}'] - row[f'k4_pooled_recall_class_{c}'] for c in (1, 2)]))
                seeds.append(row)
        seed_table = pd.DataFrame(seeds)
        seed_table.to_csv(out / 'seed_summary.csv', index=False)
        models = []
        for model in study.protocol['models']:
            s = seed_table[seed_table.model == model]
            f = folds[folds.model == model]
            delta, positive, recall = float(s.delta_f1_mean_fold.mean()), int((s.delta_f1_mean_fold > 0).sum()), float(s.delta_mean_RVOT_pooled_recall.mean())
            models.append({'model': model, 'k4_f1_mean_fold_and_seed': float(s.k4_f1_mean_fold.mean()),
                'k8_f1_mean_fold_and_seed': float(s.k8_f1_mean_fold.mean()),
                'delta_f1_mean': delta, 'delta_f1_sd_across_seed_means': float(s.delta_f1_mean_fold.std(ddof=1)),
                'positive_seeds': positive, 'delta_mean_RVOT_pooled_recall': recall,
                'k4_collapse_cases': int((f.k4_missing_predicted_classes > 0).sum()),
                'k8_collapse_cases': int((f.k8_missing_predicted_classes > 0).sum()),
                'median_elapsed_ratio_k8_over_k4': float(f.elapsed_ratio_k8_over_k4.median()),
                'prioritize_k8_for_further_development': priority_decision(delta, positive, recall, study.protocol['priority_rule'])})
        model_table = pd.DataFrame(models)
        model_table.to_csv(out / 'model_summary.csv', index=False)
        write_json(out / 'analysis.json', {'experiment_hash': study.signature, 'new_fits_verified': 30,
            'controls_verified': 30, 'development_patients': 230,
            'test_signals_read': False, 'final_model_trained': False,
            'inference': 'development prioritization only; not nested or external validation',
            'fold_seed_cases_are_not_independent': True, 'models': models,
            'analysis_source_sha256': sha256_file(__file__)})
        plot(study, seed_table, folds, out)
    print(model_table.to_string(index=False), flush=True)
    print('Development-only comparison complete. No automatic final/test stage.', flush=True)
    return model_table


def plot(study, seeds, folds, out):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, study.protocol['models']):
        s = seeds[seeds.model == model]
        x = np.arange(len(s))
        for offset, arm, color in [(-.18, 'k4', '#6482a8'), (.18, 'k8', '#dd8c48')]:
            ax.bar(x + offset, s[f'{arm}_f1_mean_fold'], .36,
                   yerr=s[f'{arm}_f1_sd_fold'], capsize=3, label=arm.upper(), color=color)
        ax.set_xticks(x, s.seed.astype(str)); ax.set_ylim(0, 1)
        ax.set_title(model); ax.set_xlabel('Training seed'); ax.legend()
    axes[0].set_ylabel('Mean fold patient macro-F1 ± fold SD')
    fig.suptitle('Development K4/K8 comparison; selected V4 controls')
    fig.tight_layout(); fig.savefig(out / 'k4_k8_development.png', dpi=180)
    fig.savefig(out / 'k4_k8_development.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, model in zip(axes, study.protocol['models']):
        for seed in study.protocol['seeds']:
            f = folds[(folds.model == model) & (folds.seed == seed)]
            ax.plot(f.fold, f.delta_f1, marker='o', label=str(seed))
        ax.axhline(0, color='black', lw=.8); ax.set_xticks(range(1, 6))
        ax.set_title(model); ax.set_xlabel('Fixed development fold'); ax.legend(title='Seed')
    axes[0].set_ylabel('Macro-F1 change (K8 − K4)')
    fig.tight_layout(); fig.savefig(out / 'paired_fold_changes.png', dpi=180)
    fig.savefig(out / 'paired_fold_changes.pdf'); plt.close(fig)
