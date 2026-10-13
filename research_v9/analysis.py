"""Audit fixed budgets and paired real-patient predictions without inference."""
import json
import os
import numpy as np
import pandas as pd
import torch
from src.artifacts import ROOT, read_json, sha256_file, write_json
from src.evaluation import compute_metrics
from research_v8.analysis import checked_predictions as v8_predictions
from research_v8.common import directory as v8_directory
from research_v8.data import load_development
from .common import checked_complete, directory, progress, require_prepared, write_manifest


def diagnose(study):
    require_prepared(study)
    output = study.run / 'diagnosis'; output.mkdir(exist_ok=True)
    rows = []; histories = []
    for row in study.plan:
        folder = v8_directory(study.parent, row['model'], row['fold'], row['seed'])
        history = pd.read_csv(folder / 'history.csv')
        selected = int(history.loc[history.inner_f1.idxmax(), 'epoch'])
        if selected != row['v8_selected_epoch']: raise RuntimeError('Inner epoch is not the first maximum')
        rows.append({**row, 'inner_best_f1': float(history.inner_f1.max()),
            'inner_epochs_run': len(history), 'epoch_delta': row['epochs'] - selected})
        histories.append(history[['epoch', 'inner_f1']].assign(model=row['model'], fold=row['fold'], seed=row['seed']))
    plan = pd.DataFrame(rows)
    summary = plan.groupby(['model', 'fold']).agg(inner_epoch_min=('v8_selected_epoch', 'min'),
        inner_epoch_max=('v8_selected_epoch', 'max'), median_epoch=('epochs', 'first'),
        reuse_cases=('reuse', 'sum')).reset_index()
    for name, data in [('selectors.csv', plan), ('epoch_summary.csv', summary), ('inner_curves.csv', pd.concat(histories))]:
        data.to_csv(output / name, index=False)
    plt = plotting(); fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharey=True)
    for ax, ((model, fold), group) in zip(axes.flat, plan.groupby(['model', 'fold'], sort=False)):
        for seed in study.protocol['seeds']:
            h = next(h for h in histories if h.model.iloc[0] == model and h.fold.iloc[0] == fold and h.seed.iloc[0] == seed)
            ax.plot(h.epoch, h.inner_f1, alpha=.75, linewidth=1, label=f'Seed {seed}')
        budget = int(group.epochs.iloc[0]); ax.axvline(budget, color='black', linestyle='--', label=f'Median {budget}')
        ax.set(title=f'{model}\nFold {fold} — median {budget}', xlabel='Inner epoch', ylim=(0, 1))
    axes[0, 0].set_ylabel('Inner patient macro-F1'); axes[1, 0].set_ylabel('Inner patient macro-F1')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc='lower center', ncol=3)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(output / 'inner_epoch_curves.png', dpi=150); plt.close(fig)
    lines = ['# Diagnosis epoch inner V8 dan rencana V9', '',
        'Median dihitung hanya dari tiga epoch terpilih inner per model/fold. Tidak memakai skor outer untuk menghitung anggaran.', '',
        '| Model | Fold | Epoch min | Epoch max | Median V9 | Reuse |', '|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples(): lines.append(f'| {r.model} | {r.fold} | {r.inner_epoch_min} | {r.inner_epoch_max} | {r.median_epoch} | {r.reuse_cases} |')
    lines += ['', '![Lintasan inner](inner_epoch_curves.png)', '',
        '10 kasus identik dipakai melalui referensi ke V8; 20 kasus baru perlu dilatih manual. Tidak membuat inner fit baru.', '',
        'Median tiga seed adalah hipotesis rekayasa untuk mengurangi sensitivitas pilihan epoch terhadap inisialisasi. '
        'Ketiganya memakai split inner pasien yang sama; ini belum menguji stabilitas terhadap perubahan split pasien.', '',
        'Outer refit memakai seluruh training fold, dari inisialisasi baru; bukan memakai bobot selector. '
        'Arsitektur, LR, pooling, scaler, batch, crop, seed dan cosine T_max=100 tetap sesuai kontrol V8.', '',
        'V9 tetap ablation development eksploratif setelah LR/pooling dipilih pada V4. Belum ada skor V9, final/test atau jaminan peningkatan.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_manifest(output / 'audit.json', {'experiment_hash': study.signature, 'inner_selectors_verified': len(rows),
        'outer_scores_used_for_budget': False, 'new_training_performed': False, 'test_signals_read': False,
        'files': {q.name: sha256_file(q) for q in output.iterdir() if q.is_file() and q.name != 'audit.json'}})
    print(f'DIAGNOSIS complete: {output / "REPORT.md"}', flush=True)


def checked_new_predictions(study, row, table):
    model, fold, seed = row['model'], row['fold'], row['seed']
    folder, result = checked_complete(study, model, fold, seed)
    if row['reuse']: return v8_predictions(study.parent, model, fold, seed, 'real_only', table)
    frame = pd.read_csv(folder / 'predictions.csv', dtype={'hospital_id': str})
    ids = study.folds[fold - 1]['outer_validation_ids']
    expected = {r['hospital_id']: r['label'] for r in study.records if r['hospital_id'] in ids}
    if not frame.hospital_id.is_unique or set(frame.hospital_id) != set(expected): raise RuntimeError('Outer patient coverage changed')
    truth = np.array([expected[h] for h in frame.hospital_id]); classes = study.inherited['class_names']
    probability = frame[[f'probability_{c}' for c in classes]].to_numpy()
    if (not np.array_equal(frame.true_label, truth) or not np.isfinite(probability).all()
            or (probability < 0).any() or not np.allclose(probability.sum(1), 1, atol=2e-6)):
        raise RuntimeError('Invalid outer probabilities/labels')
    pred = probability.argmax(1)
    if frame.predicted_class.tolist() != [classes[c] for c in pred] or not np.array_equal(frame.correct, truth == pred):
        raise RuntimeError('Prediction metadata changed')
    computed = compute_metrics(truth, probability)
    for k, value in computed.items():
        if not np.isclose(value, result['metrics'][k], rtol=1e-8, atol=1e-9): raise RuntimeError(f'Metric differs: {k}')
    history = pd.read_csv(folder / 'history.csv')
    if history.epoch.tolist() != list(range(1, row['epochs'] + 1)) or any(c.startswith(('inner_', 'validation_')) for c in history.columns):
        raise RuntimeError('Wrong fixed epoch history or outer selection callback')
    case = read_json(folder / 'case.json'); train_ids = set(case['train_ids'])
    steps = int(np.ceil(len(train_ids) / study.protocol['patient_batch_size']))
    if not history.optimizer_steps.eq(steps).all() or not history.anchor_patients.eq(len(train_ids)).all():
        raise RuntimeError('Patient update budget changed')
    for r in history.itertuples():
        anchors = json.loads(r.anchor_exposure_json)
        if set(anchors) != train_ids or set(anchors.values()) != {1}: raise RuntimeError('Anchor exposure changed')
        if any(getattr(r, k) != 0 for k in ('replacements', 'synthetic_windows', 'real_donor_windows')):
            raise RuntimeError('Real-only training used augmentation')
        if json.loads(r.real_donor_exposure_json) or json.loads(r.synthetic_exposure_json): raise RuntimeError('Unexpected donors')
    if result['optimizer_steps'] != steps * row['epochs']: raise RuntimeError('Total optimizer updates differ')
    old_history = pd.read_csv(v8_directory(study.parent, model, fold, seed, 'real_only') / 'history.csv')
    common = min(len(old_history), len(history))
    for col in ('learning_rate', 'train_loss', 'train_patient_bag_accuracy'):
        if not np.allclose(history[col].iloc[:common], old_history[col].iloc[:common], rtol=1e-5, atol=1e-6):
            raise RuntimeError(f'Training differs from V8 before the same epoch: {col}')
    for col in ('optimizer_steps', 'anchor_exposure_json', 'bag_windows', 'eligible_bags'):
        if history[col].iloc[:common].tolist() != old_history[col].iloc[:common].tolist(): raise RuntimeError(f'Crop/update trace differs: {col}')
    contributions = pd.read_csv(folder / 'segment_weights.csv', dtype={'hospital_id': str})
    indices = np.flatnonzero(table.hospital_id.isin(ids).to_numpy())
    if (not contributions.segment_index.is_unique or set(contributions.segment_index) != set(indices)
            or not np.array_equal(table.iloc[contributions.segment_index].hospital_id.to_numpy(), contributions.hospital_id)
            or not np.isfinite(contributions.weight).all() or (contributions.weight < 0).any()
            or not np.allclose(contributions.groupby('hospital_id').weight.sum(), 1, atol=2e-6)):
        raise RuntimeError('All-window patient pooling changed')
    counts = contributions.groupby('hospital_id').size()
    if not np.array_equal(contributions.n_patient_segments, contributions.hospital_id.map(counts)): raise RuntimeError('Window count differs')
    if study.choices[model]['pooling'] == 'mean' and not np.allclose(contributions.weight, 1 / contributions.hospital_id.map(counts), atol=2e-6):
        raise RuntimeError('Mean pooling differs')
    selected = torch.load(folder / 'model.pth', map_location='cpu', weights_only=True)
    last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    if (last['epoch'] != row['epochs'] or last['case_hash'] != case['case_hash']
            or set(selected) != set(last['model']) or result['selected_checkpoint_sha256'] != sha256_file(folder / 'model.pth')
            or any(not torch.isfinite(v).all() or not torch.equal(v, last['model'][k]) for k, v in selected.items())):
        raise RuntimeError('Selected checkpoint is not the finite last fixed-epoch model')
    recall = {f'recall_class_{c}': float((pred[truth == c] == c).mean()) for c in range(3)}
    if any(not np.isclose(recall[f'recall_class_{c}'], result[f'validation_recall_class_{c}']) for c in range(3)):
        raise RuntimeError('Class recall differs')
    return frame, {**computed, **recall, 'epochs': row['epochs'], 'selected_epoch': row['epochs'],
        'optimizer_steps': result['optimizer_steps'], 'elapsed_seconds': result['elapsed_seconds'],
        'missing_predicted_classes': int(sum(not (pred == c).any() for c in range(3)))}


def plotting():
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def analyze(study, pilot=False):
    require_prepared(study)
    if not pilot and not progress(study)['ready_for_comparison']: raise RuntimeError('Lengkapi notebook 48 dan 49 dahulu')
    _, table = load_development(study); rows = []; predictions = []; source_hashes = {}
    for row in study.plan:
        m, f, s = row['model'], row['fold'], row['seed']
        if pilot and (f, s) != (1, 42): continue
        control, baseline = v8_predictions(study.parent, m, f, s, 'real_only', table)
        frame, median_result = checked_new_predictions(study, row, table)
        for rule, pred, result in [('v8_per_seed', control, baseline), ('v9_median_inner', frame, median_result)]:
            rows.append({'model': m, 'fold': f, 'seed': s, 'rule': rule, 'v9_reference': row['reuse'], **result})
            predictions.append(pred.assign(model=m, fold=f, seed=s, rule=rule))
        for folder in (v8_directory(study.parent, m, f, s, 'real_only'), directory(study, m, f, s)):
            source_hashes[str((folder / 'completed.json').relative_to(ROOT))] = sha256_file(folder / 'completed.json')
    cases = pd.DataFrame(rows); pred = pd.concat(predictions, ignore_index=True)
    keys = ['model', 'fold', 'seed']
    paired = cases[cases.rule == 'v9_median_inner'].merge(cases[cases.rule == 'v8_per_seed'], on=keys, suffixes=('_v9', '_v8'), validate='one_to_one')
    delta = paired[keys].copy()
    for key in ('f1', 'recall_class_0', 'recall_class_1', 'recall_class_2', 'epochs'):
        delta[f'delta_{key}'] = paired[f'{key}_v9'] - paired[f'{key}_v8']
    delta['v9_reference'] = paired.v9_reference_v9
    if not np.allclose(delta.loc[delta.v9_reference, 'delta_f1'], 0): raise RuntimeError('Identical references must have zero effect')
    summary = cases.groupby(['model', 'rule']).agg(f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
        recall_lvot=('recall_class_0', 'mean'), recall_cusp=('recall_class_1', 'mean'),
        recall_other=('recall_class_2', 'mean'), collapse_cases=('missing_predicted_classes', lambda x: int(x.gt(0).sum()))).reset_index()
    seed_summary = cases.groupby(['model', 'seed', 'rule']).f1.mean().rename('mean_fold_f1').reset_index()
    pooled = []
    if not pilot:
        for (model, seed, rule), group in pred.groupby(['model', 'seed', 'rule']):
            if len(group) != len(study.records) or not group.hospital_id.is_unique: raise RuntimeError('OOF coverage must be once per patient/rule/seed')
            p = group[[f'probability_{c}' for c in study.inherited['class_names']]].to_numpy()
            pooled.append({'model': model, 'seed': seed, 'rule': rule, **compute_metrics(group.true_label.to_numpy(), p)})
    output = study.run / ('analysis_pilot' if pilot else 'analysis_complete'); output.mkdir(exist_ok=True)
    for name, data in [('cases.csv', cases), ('paired_deltas.csv', delta), ('summary.csv', summary),
        ('seed_summary.csv', seed_summary), ('patient_predictions.csv', pred), ('pooled_oof.csv', pd.DataFrame(pooled))]:
        data.to_csv(output / name, index=False)
    plt = plotting(); fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, study.protocol['models']):
        group = paired[paired.model == model]
        for r in group.itertuples(): ax.plot([0, 1], [r.f1_v8, r.f1_v9], 'o-', alpha=.55, linewidth=1)
        ax.set(title=model, xticks=[0, 1], xticklabels=['V8\nPer-seed inner epoch', 'V9\nMedian inner epoch'], ylim=(0, 1))
    axes[0].set_ylabel('Outer real-patient macro-F1')
    fig.tight_layout(); fig.savefig(output / 'paired_epoch_f1.png', dpi=150); plt.close(fig)
    lines = ['# V9 — ablation aturan epoch real-only' + (' (pilot)' if pilot else ''), '',
        'Metrik dihitung ulang dari prediksi pasien real; checkpoint, epoch, pooling, crop/exposure dan lintasan training diperiksa.', '',
        '| Model | Aturan | Mean fold macro-F1 | SD deskriptif | Recall LVOT | Recall cusp | Recall other |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        sd = f'{r.f1_std:.4f}' if np.isfinite(r.f1_std) else '—'
        lines.append(f'| {r.model} | {r.rule} | {r.f1_mean:.4f} | {sd} | {r.recall_lvot:.4f} | {r.recall_cusp:.4f} | {r.recall_other:.4f} |')
    lines += ['', '![Perbandingan epoch](paired_epoch_f1.png)', '',
        'Ukuran primer tetap mean fold/seed macro-F1. Pooled OOF dilaporkan terpisah karena F1 nonlinear. '
        'Periksa arah efek per seed dan kedua kelas RVOT sebelum menyatakan manfaat.', '',
        '10 kasus median identik dengan V8 dan direferensikan; bukan 10 replikasi baru. '
        'Tiga seed/fivefold bukan 15 sampel pasien independen. Pilot hanya memeriksa proses; selesaikan seluruh kasus meskipun hasil negatif.', '',
        'Median adalah hipotesis untuk mengurangi variasi pilihan epoch antar-inisialisasi. '
        'Split inner tetap; belum membuktikan ketahanan terhadap split pasien, transfer learning, atau estimasi nested bebas seluruh pilihan sebelumnya.', '',
        'Tidak membuka historical test, tidak memilih seed terbaik, dan tidak menjamin 0,9/Q1. '
        'Langkah berikut: audit kecocokan pretrained ECG, lalu tetapkan protokol pembanding from-scratch versus fine-tuning sebelum training baru.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_manifest(output / 'audit.json', {'experiment_hash': study.signature, 'cases_verified': len(rows) // 2,
        'predictions_recomputed': True, 'same_epoch_training_traces_checked': True, 'pilot': pilot,
        'test_signals_read': False, 'analysis_code_sha256': sha256_file(__file__), 'source_completed_manifests': source_hashes,
        'files': {q.name: sha256_file(q) for q in output.iterdir() if q.is_file() and q.name != 'audit.json'}})
    print(f'ANALYSIS complete: {output / "REPORT.md"}', flush=True)
