"""Audit complete V3 CV/final/test and make scientific figures; no training.

Run: .venv/bin/python -m analysis_v3.analyze_final
All derived outputs are isolated from the frozen training package.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.artifacts import canonical_hash, read_json, sha256_file, write_json
from research_v3.common import DEFAULT_RUN, context, require_final_complete
from research_v3.models import build_model
from .analyze_completed import analyze as audit_cv, recompute, require


def bootstrap_scores(truth, predictions, repeats=5000, seed=20261002):
    """Shared stratified patient draws for paired, fixed-model uncertainty."""
    truth = np.asarray(truth)
    require(set(np.unique(truth)) == {0, 1, 2}, 'Bootstrap needs all three classes')
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(np.flatnonzero(truth == k),
                                        size=(repeats, int((truth == k).sum())), replace=True)
                              for k in range(3)], axis=1)
    offset = np.arange(repeats)[:, None] * 9
    scores = {}
    for name, pred in predictions.items():
        pred = np.asarray(pred)
        require(pred.shape == truth.shape, 'Bootstrap prediction shape mismatch')
        codes = truth[indices] * 3 + pred[indices]
        cm = np.bincount((codes + offset).ravel(), minlength=repeats * 9).reshape(repeats, 3, 3)
        tp = np.diagonal(cm, axis1=1, axis2=2)
        denom = cm.sum(axis=1) + cm.sum(axis=2)
        scores[name] = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0).mean(axis=1)
    return scores


def analyze(run_id=DEFAULT_RUN, output=None):
    run, source, cfg, config, _, _ = context(run_id)
    require_final_complete(run, cfg)
    out = Path(output).resolve() if output else run / 'analysis_20261002_complete'
    audit_cv(run_id, out)
    cv = pd.read_csv(out / 'independent_cv_folds.csv')
    summary = pd.read_csv(run / 'reports/cv_summary.csv')
    require(len(cv) == 630, 'Default full V3 grid is incomplete')
    selection = read_json(run / 'selection.json')
    require(selection['test_used_for_selection'] is False and selection['selection_source'] == 'development CV'
            and selection['primary_seed'] == cfg['primary_seed'], 'Selection policy changed')
    choices = selection['choices']
    expected = []
    for model in cfg['models'] + ['extratrees']:
        winning = summary[summary.model == model].sort_values(['f1_mean', 'f1_std', 'arm', 'scenario'],
                                                            ascending=[False, True, True, True]).iloc[0]
        choice = next(c for c in choices if c['model'] == model)
        require(choice['arm'] == winning.arm and choice['scenario'] == winning.scenario, 'Selection does not match CV')
        require(np.isclose(choice['f1_mean'], winning.f1_mean) and np.isclose(choice['f1_std'], winning.f1_std), 'Selected CV score mismatch')
        if model != 'extratrees':
            cases = cv[(cv.model == model) & (cv.arm == choice['arm']) & (cv.scenario == choice['scenario'])]
            require(choice['epochs'] == int(cases.best_epoch.median()), 'Final epoch not derived from selected CV')
        expected.append(choice)
    primary = sorted([c for c in expected if c['model'] != 'extratrees'],
                     key=lambda c: (-c['f1_mean'], c['f1_std'], c['model']))[0]
    require(selection['primary'] == primary, 'Primary model does not match preregistered rule')
    primary_name, primary_seed = primary['model'], cfg['primary_seed']
    records = read_json(run / 'data/patients.json')
    development = {r['hospital_id']: r['label'] for r in records if r['pool'] == 'development'}
    historical = {r['hospital_id']: r['label'] for r in records if r['pool'] == 'test'}
    order = sorted(historical)
    truth = np.array([historical[h] for h in order])
    classes = config['class_names']
    rows, class_rows, checkpoints, probabilities, predictions, oof_frames = [], [], [], {}, {}, {}
    max_error = 0.
    signature = read_json(run / 'protocol.json')['experiment_hash']
    patient_table = pd.DataFrame({'hospital_id': order, 'true_label': truth, 'true_class': [classes[i] for i in truth]})
    for choice in choices:
        model = choice['model']
        is_tree = model == 'extratrees'
        net = None if is_tree else build_model(model, config['model']).cpu().eval()
        all_preds = []
        for seed in cfg['final_seeds']:
            directory = run / 'final' / model / f'seed_{seed}'
            case, fit = read_json(directory / 'config.json'), read_json(directory / 'metrics.json')
            require(canonical_hash({k: v for k, v in case.items() if k != 'case_hash'}) == case['case_hash'], 'Final case hash mismatch')
            require(case['experiment_hash'] == signature and case['fold'] is None and case['model'] == model
                    and case['seed'] == seed and case['arm'] == choice['arm'] and case['scenario']['id'] == choice['scenario'], 'Final identity mismatch')
            require(len(case['train_patient_ids']) == len(development) and set(case['train_patient_ids']) == set(development), 'Final cohort mismatch')
            if is_tree:
                model_path = directory / 'model.joblib'
                tree = joblib.load(model_path)
                require(np.array_equal(tree.classes_, [0, 1, 2]) and tree.n_features_in_ == 132
                        and tree.n_estimators == cfg['classical']['n_estimators'] and fit['fit_complete'], 'Tree checkpoint mismatch')
                require(tree.max_features == choice['max_features'] and tree.min_samples_leaf == choice['min_samples_leaf'], 'Tree setting mismatch')
                del tree
            else:
                require(case['settings']['epochs'] == choice['epochs'] == fit['epochs_ran'] == fit['best_epoch']
                        and case['settings']['learning_rate'] == choice['learning_rate']
                        and case['settings']['batch_size'] == choice['batch_size'], 'Neural final setting mismatch')
                history = pd.read_csv(directory / 'history.csv')
                require(len(history) == choice['epochs'] and history.epoch.tolist() == list(range(1, len(history) + 1))
                        and not any(c.startswith('validation_') for c in history.columns), 'Final history mismatch')
                model_path = directory / 'model.pth'
                weights = torch.load(model_path, map_location='cpu', weights_only=True)
                require(all(torch.isfinite(v).all().item() for v in weights.values()), 'Nonfinite model weights')
                net.load_state_dict(weights, strict=True)
                del weights
            checkpoints.append({'model': model, 'seed': seed, 'path': str(model_path), 'sha256': sha256_file(model_path),
                                'case_hash': case['case_hash'], 'loaded_valid': True})
            directory = run / 'test' / model / f'seed_{seed}'
            result = read_json(directory / 'metrics.json')
            table = pd.read_csv(directory / 'patient_predictions.csv', dtype={'hospital_id': str})
            require(table.hospital_id.is_unique and set(table.hospital_id) == set(historical), 'Historical test cohort mismatch')
            table = table.set_index('hospital_id').loc[order].reset_index()
            require(np.array_equal(table.true_label, truth) and table.true_class.tolist() == [classes[i] for i in truth], 'Test labels changed')
            p = table[[f'probability_{c}' for c in classes]].to_numpy()
            require(np.isfinite(p).all() and (p >= -1e-7).all() and (p <= 1 + 1e-7).all()
                    and np.allclose(p.sum(1), 1, atol=2e-6), 'Invalid test probabilities')
            pred = p.argmax(1)
            require(table.predicted_class.tolist() == [classes[i] for i in pred]
                    and np.array_equal(table.correct.to_numpy(), pred == truth), 'Test metadata mismatch')
            metrics = recompute(truth, p)
            for k, value in metrics.items():
                error = abs(result['metrics'][k] - value)
                max_error = max(max_error, error)
                require(error < 1e-7, f'Test metric mismatch: {model}/{seed}/{k}')
            cm = confusion_matrix(truth, pred, labels=[0, 1, 2])
            require(result['n_patients'] == len(order) and np.array_equal(result['confusion_matrix'], cm), 'Test CM/count mismatch')
            report = classification_report(truth, pred, labels=[0, 1, 2], target_names=classes, zero_division=0, output_dict=True)
            for cls in classes:
                for k, value in report[cls].items():
                    require(abs(result['classification_report'][cls][k] - value) < 1e-7, 'Classification report mismatch')
            rows.append({'model': model, 'seed': seed, 'primary_seed': seed == primary_seed, **metrics,
                         'brier_multiclass': float(np.mean(np.sum((p - np.eye(3)[truth])**2, axis=1))),
                         'nll': float(-np.log(np.clip(p[np.arange(len(truth)), truth], 1e-12, 1)).mean())})
            for i, cls in enumerate(classes):
                class_rows.append({'model': model, 'seed': seed, 'class': cls, 'support': int((truth == i).sum()),
                                   'n_predicted': int((pred == i).sum()), 'n_correct': int(((pred == i) & (truth == i)).sum()),
                                   **report[cls]})
            all_preds.append(pred)
            patient_table[f'{model}_seed_{seed}'] = [classes[i] for i in pred]
            if seed == primary_seed:
                probabilities[model], predictions[model] = p, pred
        array = np.stack(all_preds)
        patient_table[f'{model}_unique_predictions'] = [len(set(array[:, i])) for i in range(len(order))]
        del net
        selected_oof = pd.concat([pd.read_csv(p, dtype={'hospital_id': str}) for p in
            sorted((run / 'cv' / choice['arm'] / model / choice['scenario']).glob('fold_*/validation_predictions.csv'))], ignore_index=True)
        require(selected_oof.hospital_id.is_unique and set(selected_oof.hospital_id) == set(development), 'Selected OOF mismatch')
        oof_frames[model] = selected_oof.set_index('hospital_id').sort_index()
    final_scaler = read_json(run / 'scalers/window/final/scaler_provenance.json')
    require(set(final_scaler['fit_patient_ids']) == set(development), 'Final scaler includes non-development patients')
    test = pd.DataFrame(rows)
    original = pd.read_csv(run / 'reports/historical_test_patient_metrics.csv')
    joined = test.merge(original, on=['model', 'seed'], suffixes=('_audit', '_saved'), validate='one_to_one')
    require(len(joined) == len(test) == 20, 'Historical report case coverage mismatch')
    for k in ['accuracy', 'precision', 'sensitivity', 'specificity', 'f1', 'auc']:
        require(np.allclose(joined[k + '_audit'], joined[k + '_saved'], atol=1e-7), 'Saved report mismatch')
    boot = bootstrap_scores(truth, predictions)
    overview, differences, comparison, stability = [], [], [], []
    for choice in choices:
        model = choice['model']
        primary_metrics = test[(test.model == model) & (test.seed == primary_seed)].iloc[0]
        seeds = test[test.model == model]
        low, high = np.quantile(boot[model], [.025, .975])
        overview.append({**choice, 'primary_model': model == primary_name,
                         'test_f1_seed42': primary_metrics.f1, 'test_accuracy_seed42': primary_metrics.accuracy,
                         'test_f1_mean': seeds.f1.mean(), 'test_f1_sd': seeds.f1.std(ddof=1),
                         'test_f1_min': seeds.f1.min(), 'test_f1_max': seeds.f1.max(),
                         'test_f1_ci_low': low, 'test_f1_ci_high': high})
        if model != primary_name:
            d = boot[model] - boot[primary_name]
            ci = np.quantile(d, [.025, .975])
            primary_f1 = test[(test.model == primary_name) & (test.seed == primary_seed)].iloc[0].f1
            differences.append({'comparison_model': model, 'reference_model': primary_name,
                                'seed': primary_seed, 'delta_f1': primary_metrics.f1 - primary_f1,
                                'ci_low': ci[0], 'ci_high': ci[1], 'posthoc': True})
        unstable = patient_table[f'{model}_unique_predictions'] > 1
        stability.append({'model': model, 'n_patients': len(order), 'n_changed_prediction': int(unstable.sum()),
                          'fraction_changed': float(unstable.mean())})
        if model != 'extratrees':
            v2 = [read_json(source / 'test' / model / f'seed_{s}/metrics.json')['metrics']['f1'] for s in cfg['final_seeds']]
            comparison.append({'model': model, 'v2_seed42_f1': v2[0], 'v3_seed42_f1': primary_metrics.f1,
                               'seed42_delta': primary_metrics.f1 - v2[0], 'v2_five_seed_mean': np.mean(v2),
                               'v3_five_seed_mean': seeds.f1.mean(), 'five_seed_mean_delta': seeds.f1.mean() - np.mean(v2),
                               'causal_single_component_comparison': False})
    hard = pd.DataFrame({'hospital_id': sorted(development)})
    hard['true_label'] = [development[h] for h in hard.hospital_id]
    hard['true_class'] = [classes[k] for k in hard.true_label]
    n_wrong = np.zeros(len(hard), dtype=int)
    for model, p in oof_frames.items():
        p = p.loc[hard.hospital_id]
        pred = p[[f'probability_{c}' for c in classes]].to_numpy().argmax(1)
        hard[model + '_predicted_class'] = [classes[i] for i in pred]
        if model != 'extratrees': n_wrong += pred != hard.true_label.to_numpy()
    hard['n_wrong_neural_models'] = n_wrong
    overview = pd.DataFrame(overview)
    frames = {'model_overview': overview, 'independent_test_metrics': test,
              'test_class_metrics_all_seeds': pd.DataFrame(class_rows), 'checkpoint_inventory': pd.DataFrame(checkpoints),
              'paired_primary_test_bootstrap': pd.DataFrame(differences), 'seed_prediction_stability': pd.DataFrame(stability),
              'test_patient_seed_predictions': patient_table, 'v2_v3_historical_comparison': pd.DataFrame(comparison),
              'selected_development_oof_errors': hard}
    for name, frame in frames.items(): frame.to_csv(out / f'{name}.csv', index=False)
    for model, p in oof_frames.items(): p.to_csv(out / f'{model}_selected_oof.csv')
    write_json(out / 'final_audit.json', {'audited_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
               'experiment_hash': signature, 'cv_verified': 630, 'final_checkpoints_verified': len(checkpoints),
               'test_metrics_recomputed': len(test), 'maximum_test_metric_error': max_error,
               'test_patients': len(order), 'class_counts': np.bincount(truth).tolist(),
               'selection_reconstructed_from_cv': True, 'primary_model': primary_name, 'primary_seed': primary_seed,
               'primary_epochs': primary['epochs'], 'bootstrap_repeats': 5000, 'bootstrap_seed': 20261002,
               'bootstrap_scope': 'stratified patients, fixed checkpoint; shared draws for paired differences',
               'checkpoint_predictions_rerun': False, 'historical_test_is_external': False,
               'selection_cv_is_nested': False, 'real_training_run_by_analysis': False,
               'incart_benchmark_complete': (run.parents[2] / 'results/detector_benchmark/incart/summary.json').exists()})
    make_figures(out, summary, overview, test, probabilities, truth, classes)
    print('FINAL AUDIT PASS: 20 checkpoints; 20 test metric sets; maximum error', max_error)
    print(overview[['model', 'arm', 'scenario', 'f1_mean', 'f1_std', 'test_f1_seed42', 'test_f1_mean', 'test_f1_sd']].to_string(index=False))
    print(pd.DataFrame(differences).to_string(index=False))
    print('Analysis:', out)
    return out


def make_figures(out, cv, overview, test, probabilities, truth, classes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['encoder', 'patchtst', 'hierarchical', 'extratrees']
    labels = ['Encoder', 'CNN patch transformer', 'Hierarchical', 'ExtraTrees']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    table = overview.set_index('model').loc[names]
    x = np.arange(4)
    axes[0].errorbar(x, table.f1_mean, yerr=table.f1_std, fmt='o', capsize=4, color='#2563eb')
    axes[0].set_title('Selected development CV: mean ± SD of 5 folds')
    for i, model in enumerate(names):
        values = test[test.model == model].sort_values('seed').f1.to_numpy()
        axes[1].scatter(i + np.linspace(-.09, .09, len(values)), values, color='#2563eb', s=28)
    axes[1].scatter(x, table.test_f1_mean, marker='_', s=240, color='black', label='Five-seed mean')
    axes[1].scatter(x, table.test_f1_seed42, marker='D', s=45, color='#dc2626', label='Primary seed 42')
    axes[1].vlines(x, table.test_f1_ci_low, table.test_f1_ci_high, color='#dc2626', alpha=.6,
                   label='95% patient bootstrap CI, seed 42')
    axes[1].set_title('Historical test: 59 patients, five training seeds'); axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 1)); ax.tick_params(axis='x', rotation=15); ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Patient macro-F1')
    fig.suptitle('CV selected Hierarchical; historical test does not reselect model or seed')
    fig.tight_layout(); fig.savefig(out / 'selected_cv_and_test.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    display_classes = ['LVOT', 'RVOT cusp', 'RVOT septal /\nfree-wall / other']
    vmax = max(confusion_matrix(truth, p.argmax(1), labels=[0, 1, 2]).max() for p in probabilities.values())
    for ax, model, label in zip(axes.flat, names, labels):
        cm = confusion_matrix(truth, probabilities[model].argmax(1), labels=[0, 1, 2])
        ax.imshow(cm, vmin=0, vmax=vmax, cmap='Blues')
        for i in range(3):
            for j in range(3): ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > .6 * vmax else 'black')
        f1 = table.loc[model, 'test_f1_seed42']
        ax.set(xticks=range(3), xticklabels=display_classes, yticks=range(3), yticklabels=display_classes,
               xlabel='Predicted patient group', ylabel='True patient group', title=f'{label}: macro-F1 {f1:.3f}, seed 42')
        ax.tick_params(axis='both', labelsize=9)
    fig.suptitle('Historical test; supports LVOT 14, RVOT cusp 21, RVOT other 24')
    fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(out / 'primary_confusion_all_models.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    arms = ['window_segment', 'window_patient', 'r_candidate_segment', 'r_candidate_patient']
    for ax, model, label in zip(axes, names[:3], labels[:3]):
        matrix = cv[cv.model == model].pivot(index='arm', columns='scenario', values='f1_mean').reindex(index=arms)
        im = ax.imshow(matrix, vmin=.3, vmax=.7, cmap='Blues', aspect='auto')
        for i in range(4):
            for j in range(9):
                value = matrix.iloc[i, j]
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=9, color='white' if value > .6 else 'black')
        ax.set(xticks=range(9), xticklabels=matrix.columns, yticks=range(4), yticklabels=arms, title=label)
    fig.suptitle('Complete neural grid: mean patient macro-F1, 5 development folds')
    fig.subplots_adjust(left=.23, right=.88, bottom=.06, top=.91, hspace=.5)
    colorbar = fig.add_axes([.91, .2, .014, .58]); fig.colorbar(im, cax=colorbar, label='Mean CV macro-F1')
    fig.savefig(out / 'full_cv_grid.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    analyze(args.run_id, args.output)
