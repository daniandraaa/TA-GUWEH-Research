"""Verify completed real-only epoch ablation without altering V8/V9 artifacts.

Reads saved probabilities and CPU checkpoints; performs no training/inference.
"""
from pathlib import Path
from collections import Counter
import os
import json
import numpy as np
import pandas as pd
import torch
from src.artifacts import ROOT, read_json, sha256_file, write_json
from src.evaluation import compute_metrics
from research_v8.analysis import checked_predictions as v8_predictions
from research_v8.common import directory as v8_directory
from research_v9.common import (context, progress, directory, checked_complete,
    require_prepared, verify_files)
from research_v9.analysis import checked_new_predictions
from research_v9.common import DEFAULT_RUN
from research_v8.data import load_development


def close_frames(actual, expected):
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-10)


def unchanged(path):
    before = read_json(path)
    bad = [p for p, h in before.items() if not (ROOT / p).is_file() or sha256_file(ROOT / p) != h]
    if bad: raise RuntimeError(f'Protected source/results changed: {bad}')
    return len(before)


def audit():
    study = context(DEFAULT_RUN); require_prepared(study)
    if not progress(study)['ready_for_comparison']: raise RuntimeError('Complete notebook 46–50 first')
    original = study.run / 'analysis_complete'
    original_audit = read_json(original / 'audit.json'); verify_files(original, original_audit)
    if (original_audit['experiment_hash'] != study.signature or original_audit['cases_verified'] != 30
            or original_audit['pilot'] or original_audit['test_signals_read']
            or original_audit['analysis_code_sha256'] != sha256_file(ROOT / 'research_v9/analysis.py')):
        raise RuntimeError('Original analysis identity changed')
    sources = {str((original / 'audit.json').relative_to(ROOT)): sha256_file(original / 'audit.json')}
    sources.update({str((original / n).relative_to(ROOT)): h for n, h in original_audit['files'].items()})
    _, table = load_development(study)  # mmap not indexed: only patient/window metadata is used
    labels = {r['hospital_id']: r['label'] for r in study.records}
    cases = []; predictions = []; traces = []
    checkpoints = new_epochs = new_updates = reused = 0
    for model in study.protocol['models']:
        marker = study.run / f'smoke/{model}.json'; smoke = read_json(marker); verify_files(study.run, smoke)
        if (smoke['experiment_hash'] != study.signature or smoke['device'] != 'mps' or not smoke['passed']
                or smoke['shape'] != [8, 4, 12, 5000] or smoke['disposable_optimizer_steps'] != 1
                or not smoke['weights_discarded'] or smoke['test_signals_read'] or smoke['outer_validation_signals_read']):
            raise RuntimeError('Manual smoke coverage/identity changed')
        sources[str(marker.relative_to(ROOT))] = sha256_file(marker)
        print(f'AUDIT {model}: 15 budgets, predictions, fixed last checkpoints', flush=True)
        for row in [r for r in study.plan if r['model'] == model]:
            m, fold, seed = row['model'], row['fold'], row['seed']
            baseline, base_result = v8_predictions(study.parent, m, fold, seed, 'real_only', table)
            current, result = checked_new_predictions(study, row, table)
            for rule, frame, score in [('v8_per_seed', baseline, base_result), ('v9_median_inner', current, result)]:
                cases.append({'model': m, 'fold': fold, 'seed': seed, 'rule': rule, 'v9_reference': row['reuse'], **score})
                predictions.append(frame.assign(model=m, fold=fold, seed=seed, rule=rule))
            folder, metrics = checked_complete(study, m, fold, seed)
            case = read_json(folder / 'case.json'); history = pd.read_csv(folder / 'history.csv')
            if not np.isfinite(history.select_dtypes('number')).all().all(): raise RuntimeError('Nonfinite history')
            verify_files(folder, read_json(folder / 'fit_finished.json'))
            last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
            weights = torch.load(folder / 'model.pth', map_location='cpu', weights_only=True)
            if (last['case_hash'] != case['case_hash'] or last['device'] != 'mps' or last['epoch'] != row['epochs']
                    or set(weights) != set(last['model']) or metrics['selected_checkpoint_sha256'] != sha256_file(folder / 'model.pth')
                    or any(not torch.isfinite(v).all() or not torch.equal(v, last['model'][k]) for k, v in weights.items())):
                raise RuntimeError('Checkpoint identity/finite last weights changed')
            close_frames(pd.DataFrame(last['history']), history)
            del last, weights
            counts = table[table.hospital_id.isin(case['train_ids'])].groupby('hospital_id').size()
            if (not history.bag_windows.eq(int(counts.clip(upper=4).sum())).all()
                    or not history.eligible_bags.eq(int(counts.ge(4).sum())).all()):
                raise RuntimeError('Short bags or native K4 exposure changed')
            for c, n in Counter(labels[h] for h in case['train_ids']).items():
                if not history[f'class_{c}_anchors'].eq(n).all() or not history[f'class_{c}_replacements'].eq(0).all():
                    raise RuntimeError('Patient class exposure changed')
            old_folder = v8_directory(study.parent, m, fold, seed, 'real_only')
            old_history = pd.read_csv(old_folder / 'history.csv'); common = min(len(history), len(old_history))
            loss_difference = float(np.max(np.abs(history.train_loss.iloc[:common].to_numpy() - old_history.train_loss.iloc[:common].to_numpy())))
            acc_difference = float(np.max(np.abs(history.train_patient_bag_accuracy.iloc[:common].to_numpy() - old_history.train_patient_bag_accuracy.iloc[:common].to_numpy())))
            traces.append({'model': m, 'fold': fold, 'seed': seed, 'v9_epoch': row['epochs'],
                'v8_epoch': row['v8_selected_epoch'], 'v9_reference': row['reuse'], 'common_epochs': common,
                'training_loss_max_difference': loss_difference, 'training_accuracy_max_difference': acc_difference})
            for origin in {directory(study, m, fold, seed), old_folder, folder}:
                marker = origin / 'completed.json'; saved = read_json(marker)
                digest = sha256_file(marker); name = str(marker.relative_to(ROOT))
                if name in original_audit['source_completed_manifests'] and original_audit['source_completed_manifests'][name] != digest:
                    raise RuntimeError('Source differs from original completed analysis')
                sources[name] = digest
                sources.update({str((origin / n).relative_to(ROOT)): h for n, h in saved['files'].items()})
            checkpoints += 1
            if row['reuse']: reused += 1
            else: new_epochs += len(history); new_updates += int(history.optimizer_steps.sum())
    cases = pd.DataFrame(cases); predictions = pd.concat(predictions, ignore_index=True); traces = pd.DataFrame(traces)
    close_frames(cases, pd.read_csv(original / 'cases.csv'))
    close_frames(predictions, pd.read_csv(original / 'patient_predictions.csv', dtype={'hospital_id': str}))
    summary = cases.groupby(['model', 'rule']).agg(f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
        recall_lvot=('recall_class_0', 'mean'), recall_cusp=('recall_class_1', 'mean'), recall_other=('recall_class_2', 'mean'),
        collapse_cases=('missing_predicted_classes', lambda x: int(x.gt(0).sum()))).reset_index()
    close_frames(summary, pd.read_csv(original / 'summary.csv'))
    seeds = cases.groupby(['model', 'seed', 'rule']).f1.mean().rename('mean_fold_f1').reset_index()
    close_frames(seeds, pd.read_csv(original / 'seed_summary.csv'))
    keys = ['model', 'fold', 'seed']
    paired = cases[cases.rule == 'v9_median_inner'].merge(cases[cases.rule == 'v8_per_seed'], on=keys, suffixes=('_v9', '_v8'), validate='one_to_one')
    deltas = paired[keys].copy()
    for name in ('f1', 'recall_class_0', 'recall_class_1', 'recall_class_2', 'epochs'):
        deltas[f'delta_{name}'] = paired[f'{name}_v9'] - paired[f'{name}_v8']
    deltas['v9_reference'] = paired.v9_reference_v9
    close_frames(deltas, pd.read_csv(original / 'paired_deltas.csv'))
    pooled = []; confusion = []
    for (m, seed, rule), frame in predictions.groupby(['model', 'seed', 'rule']):
        if not frame.hospital_id.is_unique or set(frame.hospital_id) != set(labels): raise RuntimeError('OOF coverage changed')
        p = frame[[f'probability_{c}' for c in study.inherited['class_names']]].to_numpy()
        pooled.append({'model': m, 'seed': seed, 'rule': rule, **compute_metrics(frame.true_label.to_numpy(), p)})
        for true_label in range(3):
            for predicted in range(3):
                confusion.append({'model': m, 'seed': seed, 'rule': rule, 'true_label': true_label, 'predicted_label': predicted,
                    'count': int(((frame.true_label.to_numpy() == true_label) & (p.argmax(1) == predicted)).sum())})
    pooled = pd.DataFrame(pooled); close_frames(pooled, pd.read_csv(original / 'pooled_oof.csv'))
    confusion = pd.DataFrame(confusion)
    effects = []
    for m in study.protocol['models']:
        for seed in study.protocol['seeds']:
            g = cases[(cases.model == m) & (cases.seed == seed)].groupby('rule').mean(numeric_only=True)
            effects.append({'model': m, 'seed': seed, 'mean_fold_f1_delta': float(g.loc['v9_median_inner', 'f1'] - g.loc['v8_per_seed', 'f1']),
                'recall_cusp_delta': float(g.loc['v9_median_inner', 'recall_class_1'] - g.loc['v8_per_seed', 'recall_class_1']),
                'recall_other_delta': float(g.loc['v9_median_inner', 'recall_class_2'] - g.loc['v8_per_seed', 'recall_class_2'])})
    effects = pd.DataFrame(effects)
    notebooks = []
    for path in sorted((ROOT / 'notebooks').glob('*V9.ipynb')):
        notebook = read_json(path); code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        errors = [o.get('evalue') for c in code for o in c.get('outputs', []) if o['output_type'] == 'error']
        executed = sum(c.get('execution_count') is not None for c in code)
        if errors or executed != len(code): raise RuntimeError('Incomplete/error notebook')
        notebooks.append({'file': path.name, 'executed_code_cells': executed, 'total_code_cells': len(code), 'errors': errors})
    protected = {p.name: unchanged(p) for p in (ROOT / 'tmp/v9_frozen_before.json', ROOT / 'tmp/v9_completed_before_audit_20261006.json')}
    output = study.run / 'analysis_20261006_complete'; output.mkdir(exist_ok=True)
    for name, frame in [('cases_verified.csv', cases), ('summary_verified.csv', summary), ('seed_effects.csv', effects),
        ('pooled_oof_verified.csv', pooled), ('training_trace_comparison.csv', traces), ('confusion_by_seed.csv', confusion)]:
        frame.to_csv(output / name, index=False)
    plot(output, seeds, summary)
    result = {'experiment_hash': study.signature, 'comparison_cases_verified': len(cases) // 2,
        'new_fits_verified': checkpoints - reused, 'v8_references_verified': reused,
        'finite_last_checkpoints_verified': checkpoints, 'new_training_epochs': new_epochs, 'new_optimizer_steps': new_updates,
        'original_tables_reproduced': True, 'common_training_traces_verified': len(traces),
        'bitwise_equal_common_loss_accuracy_traces': int(((traces.training_loss_max_difference == 0) & (traces.training_accuracy_max_difference == 0)).sum()),
        'unique_development_patients': len(labels), 'comparison_prediction_rows': len(predictions),
        'protected_unchanged': protected, 'notebooks': notebooks, 'source_sha256': sources,
        'audit_code_sha256': sha256_file(__file__), 'new_training_performed_by_audit': False,
        'new_waveform_inference_performed': False, 'test_signals_read': False,
        'median_rule_consistently_improves_both_models_and_both_rvot_recalls': False}
    write_json(output / 'audit.json', result)
    report(output, result, summary, pooled, effects)
    print(f'VERIFIED: {checkpoints} cases = {checkpoints - reused} new fits + {reused} references; {output / "REPORT.md"}', flush=True)
    return result


def plot(output, seeds, summary):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rules = ['v8_per_seed', 'v9_median_inner']; labels = ['V8\nEpoch per seed', 'V9\nMedian inner']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for m, ax in zip(['hierarchical', 'encoder'], axes):
        for seed in [42, 43, 44]:
            g = seeds[(seeds.model == m) & (seeds.seed == seed)].set_index('rule')
            ax.plot([0, 1], g.loc[rules].mean_fold_f1, marker='o', label=f'Seed {seed}')
        ax.set(title=m, xticks=[0, 1], xticklabels=labels, ylim=(0, 1), ylabel='Mean fivefold patient macro-F1')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'paired_seed_macro_f1.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for m, ax in zip(['hierarchical', 'encoder'], axes):
        g = summary[summary.model == m].set_index('rule')
        for i, (key, label) in enumerate([('recall_cusp', 'RVOT cusp'), ('recall_other', 'RVOT septal/freewall/other')]):
            ax.bar(np.arange(2) + (i - .5) * .32, g.loc[rules, key], width=.32, label=label)
        ax.set(title=m, xticks=[0, 1], xticklabels=labels, ylim=(0, 1), ylabel='Mean fold/seed patient recall')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'rvot_recall_comparison.png', dpi=180); plt.close(fig)


def report(output, result, summary, pooled, effects):
    lines = ['# Audit V9 — 6 Oktober 2026', '',
        '**V9 selesai dan valid; median epoch belum menjadi perbaikan yang konsisten.**', '',
        f'{result["new_fits_verified"]} fit baru + {result["v8_references_verified"]} referensi V8, '
        f'{result["new_training_epochs"]} epoch baru dan {result["new_optimizer_steps"]:,} optimizer update terverifikasi.', '',
        '| Model | Aturan | Mean fold/seed macro-F1 ± SD | Recall cusp | Recall other |', '|---|---|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.model} | {r.rule} | {r.f1_mean:.4f} ± {r.f1_std:.4f} | {r.recall_cusp:.1%} | {r.recall_other:.1%} |')
    lines += ['', 'Encoder: F1 +0,0141, recall cusp +11,19 poin persentase, recall other −9,52 poin persentase. '
        'Mean fold-F1 naik pada dua seed, turun pada seed 44. SD antar-kasus turun sedikit, bukan bukti bahwa seluruh generalisasi lebih stabil.', '',
        'Hierarchical: F1 −0,0139, recall cusp +2,81 poin persentase, recall other −7,14 poin persentase. '
        'Mean fold-F1 turun pada seed 42/43, naik pada seed 44; SD antar-kasus meningkat.', '',
        '![Efek per seed](paired_seed_macro_f1.png)', '', '![Recall RVOT](rvot_recall_comparison.png)', '',
        '| Model | Aturan | Mean pooled OOF ± SD tiga seed |', '|---|---|---:|']
    for (m, rule), g in pooled.groupby(['model', 'rule']):
        lines.append(f'| {m} | {rule} | {g.f1.mean():.4f} ± {g.f1.std():.4f} |')
    lines += ['', 'Pooled OOF dan mean fold-F1 adalah ukuran berbeda karena F1 nonlinear. Keduanya mempertahankan arah rata-rata '
        'positif Encoder/negatif Hierarchical, tetapi beberapa arah per seed berbeda. Tidak mengganti ukuran primer untuk memilih angka tertinggi.', '',
        'Confusion matrix pada tiga seed berulang menunjukkan RVOT other→cusp: Encoder 99→121 dari 210 evaluasi other; '
        'Hierarchical 81→91. Ini 70 pasien other yang masing-masing dinilai tiga kali, bukan 210 pasien independen.', '',
        f'{result["bitwise_equal_common_loss_accuracy_traces"]}/30 lintasan loss/accuracy pada epoch bersama identik bit demi bit dengan V8. '
        'Perubahan hasil konsisten dengan aturan checkpoint yang berbeda; tidak ditemukan perubahan recipe, crop, scaler atau pasien.', '',
        'Notebook 46–50 tersimpan lengkap tanpa error. Log percobaan pertama notebook 49 pernah berhenti pada guard disk '
        '(3,17 GiB tersedia, 3,53 GiB diperlukan); run berikutnya selesai. Ini keterbatasan disk yang telah teratasi, bukan error numerik model.', '',
        '## Keputusan berikutnya', '',
        'Tidak mengulang notebook 46–50 atau menaikkan epoch secara umum. Bekukan V9 sebagai ablation aturan epoch, termasuk hasil negatif. '
        'Median tidak dipilih sebagai aturan final universal untuk kedua model. V8 per-seed tetap kontrol untuk kajian lanjutan; '
        'hasil V9 Encoder dilaporkan beserta kehilangan recall other.', '',
        'Prioritas berikut: audit kecocokan checkpoint pretrained ECG, adapter lead/sampling/durasi/normalisasi, '
        'kemudian protokol pembanding from-scratch, linear probe dan fine-tuning. Bobot/adapter pretrained belum diunduh atau diuji pada MPS.', '',
        'Semua 230 pasien dan tiga kelas tetap. LR/pooling sebelumnya dipilih pada development V4; '
        'V9 merupakan ablation eksploratif, bukan estimasi nested yang bebas pilihan sebelumnya. 10 referensi identik bukan replikasi baru.', '',
        'Tidak ada final/test baru, kesimpulan batas maksimal kinerja, atau jaminan 0,9/Q1. '
        'Audit hanya membaca checkpoint CPU dan prediksi tersimpan; source, hasil dan notebook lama tidak berubah.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__': audit()
