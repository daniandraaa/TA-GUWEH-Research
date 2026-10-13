"""Audit completed feature-only CV without loading any serialized model.

Run from project root: .venv/bin/python -m analysis_v11.completed_features
The frozen experiment's existing analysis directory is never overwritten.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score

from src.artifacts import ROOT, read_json, sha256_file, write_json
from src.evaluation import compute_metrics
from research_v11.analysis import audit_case, audit_choices, paired_bootstrap
from research_v11.benchmark import require_pass
from research_v11.features import read_features
from research_v11.morphology import columns_for
from research_v11_recovery.common import context, cache_status


ARMS = ('qrs_rr', 'morphology', 'morphology_rr')
OUT_NAME = 'analysis_20261007_complete_features'
SNAPSHOT = ROOT / 'tmp/v11_completed_features_before_audit_20261007.json'


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def compare_frame(actual, saved, sort_by):
    a = actual.sort_values(sort_by).reset_index(drop=True)
    b = saved.sort_values(sort_by).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False,
                                  rtol=0, atol=1e-10)


def plot_results(summary, scores, matrices, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    titles = {'qrs_rr': 'QRS + RR', 'morphology': 'Morphology',
              'morphology_rr': 'Morphology + RR'}
    colors = ['#758694', '#167c80', '#ca724a']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for arm, color in zip(ARMS, colors):
        rows = scores[scores.arm.eq(arm)]
        axes[0].plot(rows.fold, rows.f1, marker='o', color=color, label=titles[arm])
    axes[0].set(xlabel='Outer fold', ylabel='Patient macro-F1', xticks=range(1, 6),
                ylim=(0, 1), title='Five-fold development validation')
    axes[0].legend(fontsize=9)
    x = np.arange(3)
    axes[1].bar(x - .18, summary.mean_training_f1, width=.36, color='#a0acb5', label='Training')
    axes[1].bar(x + .18, summary.mean_fold_f1, width=.36, color=colors,
                yerr=summary.sd_fold_f1, capsize=4, label='Validation (fold SD)')
    axes[1].set(xticks=x, xticklabels=[titles[a] for a in ARMS], ylim=(0, 1.03),
                ylabel='Mean patient macro-F1', title='Training–validation gap')
    axes[1].tick_params(axis='x', labelsize=9)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(destination / 'fold_scores_and_gap.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    labels = ['LVOT', 'RVOT cusp', 'RVOT other']
    for ax, arm in zip(axes, ARMS):
        cm = np.array(matrices[arm]['counts'])
        normalized = cm / cm.sum(1, keepdims=True)
        ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{cm[i, j]}\n({normalized[i, j]:.0%})',
                        ha='center', va='center', fontsize=10,
                        color='white' if normalized[i, j] > .55 else '#162330')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=labels,
               yticklabels=labels, xlabel='Predicted class', title=titles[arm])
        ax.tick_params(axis='x', labelsize=9, rotation=20)
    axes[0].set_ylabel('True class')
    fig.suptitle('Saved out-of-fold predictions: 230 patients per arm; rows sum to 100%')
    fig.tight_layout()
    fig.savefig(destination / 'pooled_oof_confusions.png', dpi=170)
    plt.close(fig)


def main():
    study = context()
    out = study.run / OUT_NAME
    ensure(not out.exists(), f'Preserve existing audit: {out}')
    ensure(SNAPSHOT.exists(), 'Create preservation snapshot before this audit')
    print('Verifying frozen source, LUDB assets and saved feature manifests...', flush=True)
    ludb_hash = require_pass(study)
    features, feature_hash = read_features(study)  # hashes raw files; no waveform parsing/inference
    expected_sources = {'v10_source_experiment_hash': study.identity['source_v10_hash'],
                        'ludb_gate': ludb_hash, 'morphology_manifest': feature_hash}
    labels = {r['hospital_id']: r['label'] for r in study.records}
    rows, case_rows, frames = [], [], {a: [] for a in ARMS}
    case_folders = sorted((study.run / 'cases').glob('*/*/*/completed.json'))
    expected = {f'{a}/fold_{f}/{phase}' for a in ARMS for f in range(1, 6)
                for phase in [*[f'inner_S{s:02}' for s in range(1, 10)], 'outer']}
    ensure({str(p.parent.relative_to(study.run / 'cases')) for p in case_folders} == expected,
           'Completed case set differs from the three full feature-only arms')
    for n, completed in enumerate(case_folders, 1):
        folder = completed.parent
        identity, result, frame = audit_case(study, folder)
        arm, fold, phase = identity['arm'], identity['fold'], identity['phase']
        ensure(folder.parent.name == f'fold_{fold}' and folder.parent.parent.name == arm,
               'Case path and identity differ')
        ensure(identity['source_manifests'] == expected_sources, 'Feature/LUDB input binding changed')
        ensure(identity['raw_feature_count'] == len(columns_for(arm)), 'Feature count changed')
        ensure(not identity['test_signals_read'] and not identity['outer_callbacks']
               and not result['test_signals_read'], 'Development-only fit required')
        for key in ['arm', 'fold', 'phase', 'C']:
            ensure(result[key] == identity[key], f'Result identity changed: {key}')
        ensure(result['training_patients'] == len(identity['training_ids'])
               and result['evaluation_patients'] == len(identity['evaluation_ids']),
               'Result patient count changed')
        expected_name = ('outer' if phase == 'outer' else
                         f"inner_S{study.protocol['inverse_regularization_C'].index(identity['C']) + 1:02}")
        ensure(folder.name == expected_name, 'C scenario and folder differ')
        for name in ['predictions.csv', 'training_predictions.csv']:
            f = pd.read_csv(folder / name, dtype={'hospital_id': str})
            predicted = f[[f'probability_{c}' for c in study.classes]].to_numpy().argmax(1)
            ensure(f.true_class.tolist() == [study.classes[labels[h]] for h in f.hospital_id],
                   'Saved class text and labels differ')
            ensure(f.predicted_class.tolist() == [study.classes[x] for x in predicted]
                   and f.correct.tolist() == (predicted == f.true_label.to_numpy()).tolist(),
                   'Saved prediction text differs from probability argmax')
        case_rows.append({'arm': arm, 'fold': fold, 'phase': phase, 'C': identity['C'],
                          'training_f1': result['training_f1'], 'evaluation_f1': result['evaluation_f1'],
                          'solver_iterations': max(result['solver_iterations'])})
        if phase == 'outer':
            audit_choices(study, identity)
            probabilities = frame[[f'probability_{c}' for c in study.classes]].to_numpy()
            recalls = recall_score(frame.true_label, probabilities.argmax(1), labels=[0, 1, 2],
                                   average=None, zero_division=0)
            rows.append({'arm': arm, 'fold': fold, 'C': result['C'], 'f1': result['evaluation_f1'],
                         'training_f1': result['training_f1'], 'gap': result['f1_gap'],
                         **{f'recall_{i}': float(r) for i, r in enumerate(recalls)}})
            frames[arm].append(frame.assign(arm=arm, fold=fold))
        if n % 50 == 0:
            print(f'Verified {n}/150 saved fits; no model loaded or fitted', flush=True)
    scores = pd.DataFrame(rows)
    summary = scores.groupby('arm', sort=False).agg(completed_folds=('fold', 'count'),
        mean_fold_f1=('f1', 'mean'), sd_fold_f1=('f1', 'std'),
        mean_training_f1=('training_f1', 'mean'), mean_gap=('gap', 'mean'),
        recall_LVOT=('recall_0', 'mean'), recall_cusp=('recall_1', 'mean'),
        recall_other=('recall_2', 'mean')).reset_index()
    summary['full_cv_complete'] = summary.completed_folds.eq(5)
    pooled, matrices, full = [], {}, {}
    for arm in ARMS:
        f = pd.concat(frames[arm]).sort_values('hospital_id').reset_index(drop=True)
        ensure(len(f) == 230 and f.hospital_id.is_unique and f.hospital_id.tolist() == sorted(labels),
               'OOF does not cover the exact development cohort once')
        probability = f[[f'probability_{c}' for c in study.classes]].to_numpy()
        pooled.append({'arm': arm, 'patients': len(f), **compute_metrics(f.true_label.to_numpy(), probability)})
        cm = confusion_matrix(f.true_label, probability.argmax(1), labels=[0, 1, 2])
        matrices[arm] = {'classes': study.classes, 'counts': cm.tolist(),
                         'pooled_recall': (np.diag(cm) / cm.sum(1)).tolist(),
                         'row_true_column_predicted': True}
        full[arm] = f
    original = study.run / 'analysis'
    for frame, file, sort in [(scores, 'outer_scores.csv', ['arm', 'fold']),
                              (summary, 'summary.csv', ['arm']),
                              (pd.DataFrame(pooled), 'pooled_oof.csv', ['arm']),
                              (pd.concat(full.values()), 'oof_predictions.csv', ['arm', 'hospital_id'])]:
        compare_frame(frame, pd.read_csv(original / file, dtype={'hospital_id': str}), sort)
    comparisons, deltas = {}, []
    for control, treatment in [('qrs_rr', 'morphology'), ('morphology', 'morphology_rr')]:
        name = f'{treatment} minus {control}'
        pairs = scores[scores.arm.eq(control)].merge(scores[scores.arm.eq(treatment)],
                                                    on='fold', suffixes=('_control', '_treatment'))
        delta = pairs.f1_treatment - pairs.f1_control
        a, b = full[control], full[treatment]
        pa = a[[f'probability_{c}' for c in study.classes]].to_numpy().argmax(1)
        pb = b[[f'probability_{c}' for c in study.classes]].to_numpy().argmax(1)
        y = a.true_label.to_numpy()
        comparisons[name] = {**paired_bootstrap(a, b, study.classes),
            'mean_fold_delta_f1': float(delta.mean()), 'folds_improved': int((delta > 0).sum()),
            'folds_total': len(delta), 'patient_predictions_changed': int((pa != pb).sum()),
            'incorrect_to_correct': int(((pa != y) & (pb == y)).sum()),
            'correct_to_incorrect': int(((pa == y) & (pb != y)).sum())}
        deltas.extend({'comparison': name, 'fold': int(row.fold),
                       'delta_f1': float(row.f1_treatment - row.f1_control)}
                      for row in pairs.itertuples())
    quality = pd.read_csv(study.run / 'features/patient_quality.csv', dtype={'hospital_id': str})
    ensure(quality.hospital_id.tolist() == features.hospital_id.tolist()
           and quality.label.tolist() == features.label.tolist(), 'Feature QA cohort changed')
    qa = {'patients': len(quality), 'no_complete_candidate_patients': int(quality.complete_candidates.eq(0).sum()),
          'median_missing_fraction': float(quality.feature_missing_fraction.median()),
          'patients_with_any_missing_feature': int(quality.feature_missing_fraction.gt(0).sum()),
          'maximum_missing_fraction': float(quality.feature_missing_fraction.max()),
          'median_complete_candidates_by_class': quality.groupby('label').complete_candidates.median().to_dict(),
          'class_patients': quality.groupby('label').size().to_dict(),
          'feature_counts': {a: len(columns_for(a)) for a in ARMS}, 'pvc_verified': False}
    prior = read_json(SNAPSHOT)
    changed = [p for p, digest in prior.items() if not (ROOT / p).is_file() or sha256_file(ROOT / p) != digest]
    ensure(not changed, f'Protected artifacts changed: {changed}')
    out.mkdir(parents=True)
    scores.to_csv(out / 'outer_scores.csv', index=False)
    summary.set_index('arm').loc[list(ARMS)].reset_index().to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(pooled).to_csv(out / 'pooled_oof.csv', index=False)
    pd.DataFrame(case_rows).to_csv(out / 'cases_verified.csv', index=False)
    pd.DataFrame(deltas).to_csv(out / 'paired_fold_delta.csv', index=False)
    write_json(out / 'paired_bootstrap.json', comparisons)
    write_json(out / 'confusion_matrices.json', matrices)
    write_json(out / 'feature_quality.json', qa)
    plot_results(summary.set_index('arm').loc[list(ARMS)].reset_index(), scores, matrices, out)
    audit = {'experiment_hash': study.signature, 'cases_verified': len(case_rows),
             'inner_cases': sum(row['phase'] == 'inner' for row in case_rows), 'outer_cases': len(rows),
             'inner_choices_reproduced': len(rows), 'full_oof_sets': len(full),
             'patients_per_oof_set': 230, 'all_saved_metrics_reproduced': True,
             'original_analysis_tables_reproduced': True, 'checkpoint_bytes_verified_only': True,
             'preprocessing_and_class_weight_fit_ids_training_only': True,
             'ludb_manifest_sha256': ludb_hash, 'morphology_manifest_sha256': feature_hash,
             'feature_patient_shards_and_raw_source_hashes_verified': True,
             'cache_status': cache_status(study), 'completed_arms': list(ARMS),
             'planned_cases_all_seven_arms': 350, 'remaining_cache_dependent_cases': 200,
             'new_training_performed_by_audit': False, 'model_loaded_by_audit': False,
             'new_waveform_inference_performed_by_audit': False,
             'raw_source_bytes_hashed_without_signal_parsing': True,
             'test_signals_read': False, 'pvc_verified': False,
             'fully_nested_confirmatory_estimate': False,
             'protected_files_checked': len(prior), 'changed_protected_files': changed,
             'audit_source_sha256': sha256_file(Path(__file__))}
    write_json(out / 'audit.json', audit)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(comparisons, indent=2), flush=True)
    print(json.dumps(matrices, indent=2), flush=True)
    print(json.dumps(qa, indent=2), flush=True)
    print(f'Audit saved: {out}; {len(prior)} protected files unchanged', flush=True)


if __name__ == '__main__':
    main()
