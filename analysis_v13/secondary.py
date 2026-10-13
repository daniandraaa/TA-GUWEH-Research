"""Analisis sekunder V13 yang dipra-registrasi sebelum test dibuka (lihat PRAREGISTRASI_ANALISIS_SEKUNDER_V13).

1. Endpoint biner LVOT vs RVOT: P(RVOT) = P(RVOT cusp) + P(RVOT other), ambang tetap 0,5.
2. Ensemble 5 seed final: rata-rata probabilitas, kelas = argmax (3 kelas) atau ambang 0,5 (biner).
3. Pembanding tebakan mayoritas (semua RVOT) untuk endpoint biner.
Tanpa training, tanpa memilih seed/ambang dari test. Endpoint utama tetap macro-F1 tiga kelas.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve
from src.artifacts import ROOT, read_json, sha256_file, write_json
from src.reports import plotting
from research_v13.analysis import bootstrap_difference, holm, mcnemar_exact
from research_v13.common import DEFAULT_RUN, context, cv_directory, test_directory

PACKAGE = Path(__file__).resolve().parent
PREREGISTRATION = ROOT / 'docs/revisi/PRAREGISTRASI_ANALISIS_SEKUNDER_V13.md'
THRESHOLD = 0.5
ARMS = ('baseline', 'finetune')


def code_hashes():
    return {p.name: sha256_file(p) for p in sorted(PACKAGE.glob('*.py'))}


def require_preregistered():
    text = PREREGISTRATION.read_text()
    for name, digest in code_hashes().items():
        if f'{name}: {digest}' not in text:
            raise RuntimeError(f'Kode {name} berubah setelah pra-registrasi; analisis sekunder tidak sah dijalankan')


def probabilities(frame, class_names):
    return frame[[f'probability_{c}' for c in class_names]].to_numpy(dtype=float)


def binary_metrics(truth3, prob3):
    y = (np.asarray(truth3) > 0).astype(int)
    p = prob3[:, 1] + prob3[:, 2]
    pred = (p > THRESHOLD).astype(int)
    return {'macro_f1': float(f1_score(y, pred, average='macro', labels=[0, 1], zero_division=0)),
            'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'sensitivity_LVOT': float(recall_score(y, pred, pos_label=0, zero_division=0)),
            'sensitivity_RVOT': float(recall_score(y, pred, pos_label=1, zero_division=0)),
            'auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            'n_LVOT': int((y == 0).sum()), 'n_RVOT': int((y == 1).sum())}


def majority_baseline(truth3):
    y = (np.asarray(truth3) > 0).astype(int); pred = np.ones_like(y)
    return {'macro_f1': float(f1_score(y, pred, average='macro', labels=[0, 1], zero_division=0)),
            'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': 0.5,
            'sensitivity_LVOT': 0.0, 'sensitivity_RVOT': 1.0, 'auc': 0.5}


def three_class(truth, prob):
    pred = prob.argmax(1)
    return {'macro_f1': float(f1_score(truth, pred, average='macro', labels=[0, 1, 2], zero_division=0)),
            'accuracy': float(accuracy_score(truth, pred)),
            'auc': float(roc_auc_score(truth, prob, multi_class='ovr', labels=[0, 1, 2]))}


def to_binary_probabilities(prob3):
    return np.column_stack([prob3[:, 0], prob3[:, 1] + prob3[:, 2]])


def cv_binary(study, selection):
    rows = []
    for model in study.protocol['models']:
        for arm in ARMS:
            scenario = selection[model][arm]['scenario']
            for fold in study.protocol['folds']:
                for seed in study.protocol['cv_seeds']:
                    frame = pd.read_csv(cv_directory(study, model, scenario, fold, seed) / 'validation_predictions.csv',
                                        dtype={'hospital_id': str})
                    rows.append({'model': model, 'arm': arm, 'scenario': scenario, 'fold': fold, 'seed': seed,
                                 **binary_metrics(frame.true_label.to_numpy(), probabilities(frame, study.class_names))})
    cases = pd.DataFrame(rows)
    summary = cases.groupby(['model', 'arm', 'scenario']).agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_sd=('macro_f1', 'std'), auc_mean=('auc', 'mean'), auc_sd=('auc', 'std'),
        accuracy_mean=('accuracy', 'mean'), balanced_accuracy_mean=('balanced_accuracy', 'mean'),
        sensitivity_LVOT_mean=('sensitivity_LVOT', 'mean'), sensitivity_RVOT_mean=('sensitivity_RVOT', 'mean'),
        cases=('macro_f1', 'size')).reset_index()
    return cases, summary


def load_test(study, model, arm):
    frames = [pd.read_csv(test_directory(study, model, arm, s) / 'patient_predictions.csv', dtype={'hospital_id': str})
              .sort_values('hospital_id').reset_index(drop=True) for s in study.protocol['final_seeds']]
    if any(not f.hospital_id.equals(frames[0].hospital_id) for f in frames):
        raise RuntimeError('Pasien test antar-seed tidak identik')
    return frames[0].hospital_id, frames[0].true_label.to_numpy(), [probabilities(f, study.class_names) for f in frames]


def analyze(run_id=DEFAULT_RUN):
    require_preregistered()
    study = context(run_id)
    if not all((test_directory(study, m, a, s) / 'metrics.json').is_file()
               for m in study.protocol['models'] for a in ARMS for s in study.protocol['final_seeds']):
        raise RuntimeError('Jalankan test V13 (notebook 74) sampai lengkap dahulu')
    selection = read_json(study.run / 'selection.json')['choices']
    out = study.run / 'reports_secondary'; out.mkdir(parents=True, exist_ok=True)
    cv_cases, cv_summary = cv_binary(study, selection)
    cv_cases.to_csv(out / 'cv_binary_cases.csv', index=False); cv_summary.to_csv(out / 'cv_binary_summary.csv', index=False)
    rows, ensembles = [], {}
    for model in study.protocol['models']:
        for arm in ARMS:
            ids, truth, probs = load_test(study, model, arm)
            for seed, prob in zip(study.protocol['final_seeds'], probs):
                rows.append({'model': model, 'arm': arm, 'seed': seed, 'kind': 'single_seed',
                             **{f'3class_{k}': v for k, v in three_class(truth, prob).items()},
                             **{f'binary_{k}': v for k, v in binary_metrics(truth, prob).items()}})
            mean = np.mean(probs, axis=0); ensembles[(model, arm)] = (truth, mean)
            rows.append({'model': model, 'arm': arm, 'seed': 'ensemble_5', 'kind': 'ensemble',
                         **{f'3class_{k}': v for k, v in three_class(truth, mean).items()},
                         **{f'binary_{k}': v for k, v in binary_metrics(truth, mean).items()}})
    test = pd.DataFrame(rows); test.to_csv(out / 'test_secondary_all.csv', index=False)
    majority = majority_baseline(truth)
    write_json(out / 'binary_majority_baseline.json', {'description': 'tebak semua pasien RVOT', **majority})
    single = test[test.kind == 'single_seed']
    overview = single.groupby(['model', 'arm']).agg(
        **{f'{c}_mean': (c, 'mean') for c in ['3class_macro_f1', '3class_auc', 'binary_macro_f1', 'binary_auc', 'binary_accuracy', 'binary_sensitivity_LVOT']},
        **{f'{c}_sd': (c, 'std') for c in ['3class_macro_f1', 'binary_macro_f1']}).reset_index()
    ens = test[test.kind == 'ensemble'].set_index(['model', 'arm'])
    for c in ['3class_macro_f1', '3class_auc', 'binary_macro_f1', 'binary_auc', 'binary_accuracy', 'binary_sensitivity_LVOT']:
        overview[f'{c}_ensemble'] = [ens.loc[(r.model, r.arm), c] for r in overview.itertuples()]
    overview.to_csv(out / 'bab4_tabel_sekunder.csv', index=False)
    p = study.protocol['statistics']; stats = []
    for model in study.protocol['models']:
        truth, base = ensembles[(model, 'baseline')]; _, tuned = ensembles[(model, 'finetune')]
        for endpoint, a, b, y in (('3 kelas', base, tuned, truth),
                                  ('biner LVOT vs RVOT', to_binary_probabilities(base), to_binary_probabilities(tuned), (truth > 0).astype(int))):
            stats.append({'model': model, 'endpoint': endpoint, 'predictions': 'ensemble 5 seed',
                          **mcnemar_exact(a.argmax(1) == y, b.argmax(1) == y),
                          **bootstrap_difference(y, a, b, p['bootstrap_resamples'], p['bootstrap_seed'])})
    stats = pd.DataFrame(stats)
    for endpoint in stats.endpoint.unique():
        mask = stats.endpoint == endpoint
        stats.loc[mask, 'p_holm'] = holm(stats.loc[mask, 'p_value'])
    stats.to_csv(out / 'stat_sekunder_ensemble.csv', index=False)
    plt = plotting(study.run)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, arm in zip(axes, ARMS):
        for model in study.protocol['models']:
            truth, mean = ensembles[(model, arm)]
            fpr, tpr, _ = roc_curve((truth > 0).astype(int), mean[:, 1] + mean[:, 2])
            ax.plot(fpr, tpr, label=f'{model} (AUC {roc_auc_score((truth > 0).astype(int), mean[:, 1] + mean[:, 2]):.3f})')
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=.8)
        ax.set_title(f'ROC LVOT vs RVOT — {arm} (ensemble 5 seed)'); ax.set_xlabel('1 − spesifisitas'); ax.set_ylabel('Sensitivitas RVOT'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'roc_biner_ensemble.png', dpi=180); plt.close(fig)
    lines = ['# Analisis sekunder V13 (dipra-registrasi)', '',
             'Endpoint utama tetap macro-F1 tiga kelas (lihat reports/RINGKASAN_V13.md). Bagian ini sekunder.', '',
             f'Pembanding tebakan mayoritas (semua RVOT): akurasi {majority["accuracy"]:.3f}, macro-F1 {majority["macro_f1"]:.3f}, AUC 0,5.', '',
             '## Test: rata-rata 5 seed dan ensemble', '']
    for _, r in overview.iterrows():
        lines.append(f'- {r["model"]}/{r["arm"]}: 3 kelas F1 {r["3class_macro_f1_mean"]:.3f} (ensemble {r["3class_macro_f1_ensemble"]:.3f}); '
                     f'biner F1 {r["binary_macro_f1_mean"]:.3f} (ensemble {r["binary_macro_f1_ensemble"]:.3f}), '
                     f'AUC biner {r["binary_auc_mean"]:.3f}, sensitivitas LVOT {r["binary_sensitivity_LVOT_mean"]:.3f}')
    lines += ['', '## CV development (endpoint biner, 5 fold × 3 seed)', '']
    for r in cv_summary.itertuples():
        lines.append(f'- {r.model}/{r.arm} ({r.scenario}): F1 {r.macro_f1_mean:.3f} ± {r.macro_f1_sd:.3f}; AUC {r.auc_mean:.3f}; '
                     f'sensitivitas LVOT {r.sensitivity_LVOT_mean:.3f}')
    lines += ['', '## Baseline vs fine-tuning (ensemble)', '']
    for r in stats.itertuples():
        lines.append(f'- {r.model}, {r.endpoint}: ΔF1 {r.observed_difference:+.3f} [CI95 {r.ci95_low:+.3f}, {r.ci95_high:+.3f}], '
                     f'McNemar p={r.p_value:.4f}, Holm p={r.p_holm:.4f}')
    lines += ['', 'Catatan: gagasan endpoint biner muncul setelah melihat prediksi test V4 pada 59 pasien yang sama; '
              'bukti yang tidak terpengaruh test adalah hasil CV development di atas.']
    (out / 'RINGKASAN_SEKUNDER_V13.md').write_text('\n'.join(lines) + '\n')
    write_json(out / 'manifest.json', {'experiment_hash': study.signature, 'code': code_hashes(),
                                       'preregistration_sha256': sha256_file(PREREGISTRATION), 'threshold': THRESHOLD})
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    analyze(parser.parse_args().run_id)
