"""Tahap V14: prepare, smoke, CV biner berpasangan dengan V13, dan perbandingan pra-registrasi."""
import gc
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
import research_v4.training as v4_training
from src import evaluation as original
from src.artifacts import lock, read_json, seed_everything, sha256_file, write_json
from src.experiment import bind_case
from src.splits import fold_indices
from research_v3.sampling import patient_class_weights
from research_v4.common import disk_check
from research_v4.data import scaler_for
from research_v4.models import build_model
from research_v4.training import fit
from research_v13.analysis import holm, wilcoxon
from research_v13.common import cv_directory as v13_cv_directory
from research_v13.data import GanBags, PatientBags, collate_bags, load_pool as v13_load_pool, load_synthetic
from .common import cv_cases, cv_directory, progress

V13_PROBABILITIES = ['probability_LVOT', 'probability_RVOT_cusp', 'probability_RVOT_septal_freewall_other']


def compute_metrics(truth, probabilities):
    """Rumus identik src.evaluation.compute_metrics; AUC biner memakai P(RVOT) satu kolom."""
    n = probabilities.shape[1]
    if n != 2:
        return original.compute_metrics(truth, probabilities)
    pred = probabilities.argmax(axis=1)
    cm = confusion_matrix(truth, pred, labels=np.arange(n))
    specificity = []
    for i in range(n):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(float(tn / (tn + fp)) if tn + fp else 0.0)
    auc = float(roc_auc_score(truth, probabilities[:, 1])) if len(np.unique(truth)) == n else None
    labels = np.arange(n)
    return {'accuracy': float(accuracy_score(truth, pred)),
            'precision': float(precision_score(truth, pred, average='macro', labels=labels, zero_division=0)),
            'sensitivity': float(recall_score(truth, pred, average='macro', labels=labels, zero_division=0)),
            'specificity': float(np.mean(specificity)),
            'f1': float(f1_score(truth, pred, average='macro', labels=labels, zero_division=0)), 'auc': auc}


def report(truth, probabilities, class_names):
    if probabilities.shape[1] != 2:
        return original.report(truth, probabilities, class_names)
    pred = probabilities.argmax(1); labels = np.arange(len(class_names))
    return {'metrics': compute_metrics(truth, probabilities),
            'confusion_matrix': confusion_matrix(truth, pred, labels=labels).tolist(),
            'classification_report': classification_report(truth, pred, labels=labels, target_names=class_names,
                                                           zero_division=0, output_dict=True)}


# Loop training V4 dipakai apa adanya; hanya fungsi metriknya diganti versi biner di proses V14.
v4_training.compute_metrics = compute_metrics
v4_training.report = report


def binary_table(study, table):
    mapped = table.copy()
    mapped['label'] = mapped.label.map(study.protocol['label_map']).astype(int)
    return mapped


def load_development(study):
    X, table = v13_load_pool(study.v13, 'development')
    return X, binary_table(study, table)


class BinaryGanBags(GanBags):
    """Augmentasi V13 (1 dari 4 window diganti sintetis) dengan kelompok sintetis RVOT digabung."""

    def __init__(self, X, table, ids, scaler, synthetic, synthetic_labels, label_map, size=4, sample_seed=None):
        PatientBags.__init__(self, X, table, ids, scaler, size, sample_seed)
        self.size, self.synthetic = size, synthetic
        mapped = np.asarray([label_map[int(v)] for v in synthetic_labels])
        self.groups = {c: np.flatnonzero(mapped == c) for c in (0, 1)}
        if any(not len(g) for g in self.groups.values()):
            raise ValueError('Setiap kelas biner memerlukan window sintetis')
        self.replacements = 0


def make_loader(study, X, table, ids, scaler, scenario, seed, synthetic_folder):
    k = study.v13.protocol['segments_per_patient']
    if scenario['augmentation'] == 'gan':
        values, labels = load_synthetic(synthetic_folder)
        data = BinaryGanBags(X, table, ids, scaler, values, labels, study.protocol['label_map'], k, sample_seed=seed + 10000)
    else:
        data = PatientBags(X, table, ids, scaler, k, sample_seed=seed + 10000)
    return DataLoader(data, batch_size=scenario['batch_size'] // k, shuffle=True, num_workers=0,
                      collate_fn=collate_bags, generator=torch.Generator().manual_seed(seed))


def prepare(study):
    X, table = load_development(study)
    for fold in study.v13.folds:
        train, _ = fold_indices(table, fold)
        scaler_for(study.run, f'fold_{fold["fold"]}', X, table, train, fold['train_ids'])
    counts = table.drop_duplicates('hospital_id').label.value_counts().sort_index()
    out = study.run / 'reports'; out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'preparation.json', {'experiment_hash': study.signature,
        'development_patients': int(table.hospital_id.nunique()), 'development_windows': len(table),
        'class_patients': {study.class_names[int(k)]: int(v) for k, v in counts.items()},
        'scenarios': progress(study)['scenarios'], 'cv_fits_expected': progress(study)['cv_expected'],
        'test_signals_read': False})
    print(f'V14 prepared: {table.hospital_id.nunique()} pasien development '
          f'(LVOT {int(counts[0])}, RVOT {int(counts[1])}); {progress(study)["cv_expected"]} fit CV. '
          f'Skenario: {progress(study)["scenarios"]}. Test tidak dibaca.', flush=True)


def settings_for(study, scenario):
    return {**study.v13.protocol['training'], 'learning_rate': scenario['learning_rate'], 'batch_size': scenario['batch_size']}


def run_case(study, model, arm, fold, seed, device):
    if device.type not in {'cpu', 'mps'}:
        raise ValueError('Resume epoch hanya divalidasi pada CPU/MPS')
    X, table = load_development(study)
    scenario = study.scenarios[model][arm]
    manifest = study.v13.folds[fold - 1]
    train, _ = fold_indices(table, manifest)
    synthetic = study.v13.gan.folder(fold) if scenario['augmentation'] == 'gan' else None
    effective = seed + 100 * fold  # sama dengan V13 agar berpasangan
    s = settings_for(study, scenario)
    directory = cv_directory(study, model, arm, fold, seed)
    identity = {'experiment_hash': study.signature, 'device': str(device), 'model': model, 'arm': arm,
                'scenario': scenario, 'fold': fold, 'seed': seed, 'effective_seed': effective, 'settings': s,
                'train_patient_ids': sorted(manifest['train_ids']),
                'validation_patient_ids': sorted(manifest['validation_ids']),
                'synthetic_generated_sha256': sha256_file(synthetic / 'generated.json') if synthetic else None}
    with lock(directory / '.lock'):
        bind_case(directory, identity)
        if (directory / 'metrics.json').exists():
            return read_json(directory / 'metrics.json')
        disk_check(study.run, study.protocol)
        scaler = scaler_for(study.run, f'fold_{fold}', X, table, train, manifest['train_ids'])
        seed_everything(effective)
        loader = make_loader(study, X, table, manifest['train_ids'], scaler, scenario, effective, synthetic)
        validation = PatientBags(X, table, manifest['validation_ids'], scaler)
        network = build_model(model, 'mean', study.model_config, {}).to(device)
        seed_everything(effective + 20000)
        print(f'V14 CV fold{fold} {model}/{arm} ({scenario["id"]}) seed={seed} LR={s["learning_rate"]} '
              f'batch={s["batch_size"]} augmentasi={scenario["augmentation"]} kelas=LVOT/RVOT', flush=True)
        try:
            return fit(network, loader, validation, device, s, directory, study.class_names,
                       patient_class_weights(table.iloc[train], 2), study.protocol['evaluation_segment_chunk'])
        finally:
            del network, loader, validation; gc.collect()
            if device.type == 'mps': torch.mps.empty_cache()


def smoke(study, device):
    """Satu update per model x arm pada fold 1; bobot dibuang."""
    X, table = load_development(study)
    manifest = study.v13.folds[0]
    train, _ = fold_indices(table, manifest)
    scaler = scaler_for(study.run, 'fold_1', X, table, train, manifest['train_ids'])
    records = []
    for model in study.protocol['models']:
        for arm in study.protocol['arms']:
            scenario = study.scenarios[model][arm]
            seed_everything(1442)
            loader = make_loader(study, X, table, manifest['train_ids'], scaler, scenario, 1442, study.v13.gan.folder(1))
            values, labels, mask, _ = next(iter(loader))
            network = build_model(model, 'mean', study.model_config, {}).to(device).train()
            optimizer = torch.optim.AdamW(network.parameters(), lr=scenario['learning_rate'])
            started = time.monotonic()
            logp = network(values.to(device), mask.to(device))
            loss = F.nll_loss(logp, labels.to(device), weight=patient_class_weights(table.iloc[train], 2).to(device))
            loss.backward(); optimizer.step()
            if logp.shape[1] != 2 or set(labels.tolist()) - {0, 1}:
                raise RuntimeError('Keluaran/label smoke bukan biner')
            if not torch.isfinite(loss) or not all(torch.isfinite(p).all() for p in network.parameters()):
                raise RuntimeError(f'Smoke gagal {model}/{arm}')
            replaced = getattr(loader.dataset, 'replacements', 0)
            if scenario['augmentation'] == 'gan' and replaced == 0:
                raise RuntimeError('Augmentasi GAN tidak mengganti window apa pun')
            records.append({'model': model, 'arm': arm, 'scenario': scenario['id'], 'batch_shape': list(values.shape),
                            'output_classes': int(logp.shape[1]), 'gan_replacements_in_batch': replaced,
                            'loss': float(loss.detach().cpu()), 'seconds': time.monotonic() - started})
            print(f'SMOKE PASS {model}/{arm} ({scenario["id"]}) batch={list(values.shape)} '
                  f'keluaran={int(logp.shape[1])} kelas ganti_GAN={replaced}', flush=True)
            del network, optimizer, loader; gc.collect()
            if device.type == 'mps': torch.mps.empty_cache()
    write_json(study.run / 'smoke.json', {'experiment_hash': study.signature, 'device': str(device),
                                          'passed': True, 'cases': records, 'weights_discarded': True})


def require_smoke(study, device):
    path = study.run / 'smoke.json'
    if not path.exists() or read_json(path)['experiment_hash'] != study.signature or read_json(path)['device'] != str(device):
        raise RuntimeError('Jalankan smoke V14 pada device yang sama dahulu')


def cv(study, device, models=None, arms=None, folds=None, seeds=None):
    require_smoke(study, device)
    for model, arm, fold, seed in cv_cases(study, models, arms, folds, seeds):
        run_case(study, model, arm, fold, seed, device)


def binary_scores(truth, p_rvot):
    pred = (np.asarray(p_rvot) > 0.5).astype(int)
    return {'f1': float(f1_score(truth, pred, average='macro', labels=[0, 1], zero_division=0)),
            'auc': float(roc_auc_score(truth, p_rvot)), 'accuracy': float(accuracy_score(truth, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(truth, pred)),
            'sensitivity_LVOT': float(recall_score(truth, pred, pos_label=0, zero_division=0)),
            'sensitivity_RVOT': float(recall_score(truth, pred, pos_label=1, zero_division=0))}


def compare(study):
    state = progress(study)
    if state['cv_complete'] != state['cv_expected']:
        raise RuntimeError(f'Perbandingan terkunci: CV V14 baru {state["cv_complete"]}/{state["cv_expected"]}')
    rows = []
    for model, arm, fold, seed in cv_cases(study):
        scenario = study.scenarios[model][arm]['id']
        mine = pd.read_csv(cv_directory(study, model, arm, fold, seed) / 'validation_predictions.csv', dtype={'hospital_id': str})
        old = pd.read_csv(v13_cv_directory(study.v13, model, scenario, fold, seed) / 'validation_predictions.csv',
                          dtype={'hospital_id': str})
        mine, old = mine.sort_values('hospital_id').reset_index(drop=True), old.sort_values('hospital_id').reset_index(drop=True)
        if not mine.hospital_id.equals(old.hospital_id):
            raise RuntimeError('Pasien validasi V14 dan V13 tidak identik')
        truth = old.true_label.map(study.protocol['label_map']).to_numpy()
        if not np.array_equal(truth, mine.true_label.to_numpy()):
            raise RuntimeError('Label biner V14 tidak sesuai pemetaan V13')
        v13 = binary_scores(truth, old[V13_PROBABILITIES[1]] + old[V13_PROBABILITIES[2]])
        v14 = binary_scores(truth, mine.probability_RVOT.to_numpy())
        rows.append({'model': model, 'arm': arm, 'scenario': scenario, 'fold': fold, 'seed': seed,
                     'best_epoch_v14': read_json(cv_directory(study, model, arm, fold, seed) / 'metrics.json')['best_epoch'],
                     **{f'v13_collapsed_{k}': v for k, v in v13.items()}, **{f'v14_binary_{k}': v for k, v in v14.items()}})
    cases = pd.DataFrame(rows)
    out = study.run / 'reports'; out.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out / 'cv_paired_cases.csv', index=False)
    metrics = ['f1', 'auc', 'balanced_accuracy', 'sensitivity_LVOT', 'sensitivity_RVOT', 'accuracy']
    summary = []
    for (model, arm), g in cases.groupby(['model', 'arm'], sort=False):
        row = {'model': model, 'arm': arm, 'scenario': g.scenario.iloc[0], 'pairs': len(g)}
        for m in metrics:
            row[f'v13_collapsed_{m}_mean'] = g[f'v13_collapsed_{m}'].mean()
            row[f'v14_binary_{m}_mean'] = g[f'v14_binary_{m}'].mean()
        row[f'v13_collapsed_f1_sd'] = g.v13_collapsed_f1.std(); row['v14_binary_f1_sd'] = g.v14_binary_f1.std()
        for m in ('f1', 'auc'):
            w = wilcoxon(g[f'v13_collapsed_{m}'], g[f'v14_binary_{m}'])
            row[f'delta_{m}'] = w['mean_difference']; row[f'p_{m}'] = w['p_value']
        summary.append(row)
    summary = pd.DataFrame(summary)
    for m in ('f1', 'auc'):
        summary[f'p_{m}_holm'] = holm(summary[f'p_{m}'])
    summary.to_csv(out / 'cv_comparison.csv', index=False)
    alpha = study.protocol['decision']['alpha']; decision = {}
    for r in summary[summary.arm == 'finetune'].itertuples():
        use_v14 = bool(r.delta_f1 > 0 and r.p_f1_holm < alpha)
        decision[r.model] = {'approach': 'v14_binary' if use_v14 else 'v13_collapsed', 'delta_f1': float(r.delta_f1),
                             'p_f1_holm': float(r.p_f1_holm), 'v13_collapsed_f1': float(r.v13_collapsed_f1_mean),
                             'v14_binary_f1': float(r.v14_binary_f1_mean)}
    write_json(out / 'decision.json', {'experiment_hash': study.signature, 'rule': study.protocol['decision']['rule'],
                                       'source': 'development CV only', 'test_read': False, 'choices': decision})
    plot(study, cases, out)
    lines = ['# Ringkasan V14: model biner khusus vs model 3 kelas digabung (CV development)', '',
             'Pasangan identik: fold, seed efektif, resep, scaler dan skenario sama dengan V13. Test tidak dibaca. '
             'Ambang tetap 0,5. Wilcoxon dua sisi, Holm atas 6 model x arm.', '']
    for r in summary.itertuples():
        lines.append(f'- {r.model}/{r.arm} ({r.scenario}): F1 V13-gabung {r.v13_collapsed_f1_mean:.3f} → V14-biner '
                     f'{r.v14_binary_f1_mean:.3f} (Δ {r.delta_f1:+.3f}, Holm p={r.p_f1_holm:.4f}); AUC '
                     f'{r.v13_collapsed_auc_mean:.3f} → {r.v14_binary_auc_mean:.3f} (Holm p={r.p_auc_holm:.4f}); '
                     f'sensitivitas LVOT {r.v13_collapsed_sensitivity_LVOT_mean:.3f} → {r.v14_binary_sensitivity_LVOT_mean:.3f}')
    lines += ['', '## Keputusan pra-registrasi (arm fine-tuning)', '']
    lines += [f'- {m}: **{d["approach"]}** (Δ F1 {d["delta_f1"]:+.3f}, Holm p={d["p_f1_holm"]:.4f})' for m, d in decision.items()]
    (out / 'RINGKASAN_V14.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    return decision


def plot(study, cases, out):
    from src.reports import plotting
    plt = plotting(study.run)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, arm in zip(axes, study.protocol['arms']):
        g = cases[cases.arm == arm]
        for i, model in enumerate(study.protocol['models']):
            h = g[g.model == model]
            for a, b in zip(h.v13_collapsed_f1, h.v14_binary_f1):
                ax.plot([i - .15, i + .15], [a, b], color='gray', alpha=.35, lw=.8)
            ax.scatter(np.full(len(h), i - .15), h.v13_collapsed_f1, s=14, label='V13 3 kelas digabung' if i == 0 else None)
            ax.scatter(np.full(len(h), i + .15), h.v14_binary_f1, s=14, label='V14 biner khusus' if i == 0 else None)
        ax.set_xticks(range(3)); ax.set_xticklabels(study.protocol['models']); ax.set_title(f'{arm} (15 pasangan fold x seed)')
        ax.set_ylabel('Macro-F1 LVOT vs RVOT (CV)'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'cv_paired_f1.png', dpi=180); plt.close(fig)
