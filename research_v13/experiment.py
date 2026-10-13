"""Tahap manual: prepare, smoke, CV (baseline + grid GAN), seleksi, final, test."""
import gc
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.artifacts import canonical_hash, lock, read_json, seed_everything, sha256_file, write_json
from src.evaluation import prediction_table, report
from src.experiment import bind_case
from src.splits import fold_indices
from research_v3.sampling import patient_class_weights
from research_v4.common import disk_check
from research_v4.data import scaler_for
from research_v4.models import build_model
from research_v4.training import fit, predict
from . import gan_final
from .common import all_scenarios, baseline, cv_cases, cv_directory, final_directory, grid, progress, test_directory
from .data import GanBags, PatientBags, collate_bags, load_pool, load_synthetic


def prepare(study):
    X, table = load_pool(study)
    rows = []
    for fold in study.folds:
        train, val = fold_indices(table, fold)
        scaler_for(study.run, f'fold_{fold["fold"]}', X, table, train, fold['train_ids'])
        rows.append({'fold': fold['fold'], 'train_patients': len(fold['train_ids']),
                     'validation_patients': len(fold['validation_ids']),
                     'train_segments': len(train), 'validation_segments': len(val)})
    scaler_for(study.run, 'final', X, table, np.arange(len(X)), table.hospital_id.unique().tolist())
    out = study.run / 'reports'; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'fold_inventory.csv', index=False)
    counts = table.drop_duplicates('hospital_id').label.value_counts().sort_index()
    write_json(out / 'preparation.json', {'experiment_hash': study.signature,
        'development_patients': int(table.hospital_id.nunique()), 'development_windows': len(table),
        'class_patients': {study.class_names[int(k)]: int(v) for k, v in counts.items()},
        'cv_fits_expected': progress(study)['cv_expected'], 'final_fits_expected': progress(study)['final_expected'],
        'test_signals_read': False})
    gan_final.prepare(study)
    print(f'Prepared: {table.hospital_id.nunique()} pasien development, {len(table)} window; '
          f'{progress(study)["cv_expected"]} fit CV direncanakan. Test tetap terkunci.', flush=True)


def settings_for(study, scenario, epochs=None):
    s = {**study.protocol['training'], 'learning_rate': scenario['learning_rate'], 'batch_size': scenario['batch_size']}
    if epochs is not None:
        s['epochs'] = int(epochs)
    return s


def make_loader(study, X, table, ids, scaler, scenario, seed, synthetic_folder):
    k = study.protocol['segments_per_patient']
    if scenario['augmentation'] == 'gan':
        values, labels = load_synthetic(synthetic_folder)
        data = GanBags(X, table, ids, scaler, values, labels, k, sample_seed=seed + 10000)
    else:
        data = PatientBags(X, table, ids, scaler, k, sample_seed=seed + 10000)
    return DataLoader(data, batch_size=scenario['batch_size'] // k, shuffle=True, num_workers=0,
                      collate_fn=collate_bags, generator=torch.Generator().manual_seed(seed))


def run_case(study, model, scenario, device, fold=None, seed=42, arm=None, epochs=None):
    if device.type not in {'cpu', 'mps'}:
        raise ValueError('Resume epoch hanya divalidasi pada CPU/MPS')
    X, table = load_pool(study)
    if fold is not None:
        manifest = study.folds[fold - 1]
        train, _ = fold_indices(table, manifest)
        ids, scope = manifest['train_ids'], f'fold_{fold}'
        directory = cv_directory(study, model, scenario['id'], fold, seed)
        synthetic = study.gan.folder(fold) if scenario['augmentation'] == 'gan' else None
        effective = seed + 100 * fold
    else:
        if seed not in study.protocol['final_seeds'] or not epochs:
            raise ValueError('Final memerlukan seed terdaftar dan epoch dari CV')
        train, ids, scope = np.arange(len(X)), table.hospital_id.unique().tolist(), 'final'
        directory = final_directory(study, model, arm, seed)
        synthetic = gan_final.require_quality(study) if scenario['augmentation'] == 'gan' else None
        effective = seed
    s = settings_for(study, scenario, epochs)
    identity = {'experiment_hash': study.signature, 'device': str(device), 'model': model, 'scenario': scenario,
        'arm': arm, 'fold': fold, 'seed': seed, 'effective_seed': effective, 'settings': s,
        'train_patient_ids': sorted(ids),
        'validation_patient_ids': sorted(study.folds[fold - 1]['validation_ids']) if fold else [],
        'synthetic_generated_sha256': sha256_file(synthetic / 'generated.json') if synthetic else None,
        'patient_batch_size': s['batch_size'] // study.protocol['segments_per_patient']}
    with lock(directory / '.lock'):
        bind_case(directory, identity)
        if (directory / 'metrics.json').exists():
            return read_json(directory / 'metrics.json')
        disk_check(study.run, study.protocol)
        scaler = scaler_for(study.run, scope, X, table, train, ids)
        seed_everything(effective)
        loader = make_loader(study, X, table, ids, scaler, scenario, effective, synthetic)
        validation = PatientBags(X, table, study.folds[fold - 1]['validation_ids'], scaler) if fold else None
        network = build_model(model, study.protocol['pooling'], study.config['model'], {}).to(device)
        seed_everything(effective + 20000)
        print(f'V13 {"CV fold" + str(fold) if fold else "FINAL " + arm} {model}/{scenario["id"]} seed={seed} '
              f'LR={s["learning_rate"]} batch={s["batch_size"]} augmentasi={scenario["augmentation"]}', flush=True)
        try:
            result = fit(network, loader, validation, device, s, directory, study.class_names,
                         patient_class_weights(table.iloc[train], 3), study.protocol['evaluation_segment_chunk'])
        finally:
            del network, loader, validation; gc.collect()
            if device.type == 'mps': torch.mps.empty_cache()
        return result


def smoke(study, device):
    """Satu update sekali pakai per model x {baseline, GAN}; bobot dibuang."""
    X, table = load_pool(study)
    manifest = study.folds[0]
    train, _ = fold_indices(table, manifest)
    scaler = scaler_for(study.run, 'fold_1', X, table, train, manifest['train_ids'])
    records = []
    for model in study.protocol['models']:
        for scenario in (baseline(study.protocol), grid(study.protocol)[-1]):
            seed_everything(1342)
            loader = make_loader(study, X, table, manifest['train_ids'], scaler, scenario, 1342, study.gan.folder(1))
            values, labels, mask, _ = next(iter(loader))
            network = build_model(model, 'mean', study.config['model'], {}).to(device).train()
            optimizer = torch.optim.AdamW(network.parameters(), lr=scenario['learning_rate'])
            started = time.monotonic()
            loss = F.nll_loss(network(values.to(device), mask.to(device)), labels.to(device),
                              weight=patient_class_weights(table.iloc[train], 3).to(device))
            loss.backward(); optimizer.step()
            if not torch.isfinite(loss) or not all(torch.isfinite(p).all() for p in network.parameters()):
                raise RuntimeError(f'Smoke gagal {model}/{scenario["id"]}')
            replaced = getattr(loader.dataset, 'replacements', 0)
            if scenario['augmentation'] == 'gan' and replaced == 0:
                raise RuntimeError('Augmentasi GAN tidak mengganti window apa pun')
            records.append({'model': model, 'scenario': scenario['id'], 'batch_shape': list(values.shape),
                            'gan_replacements_in_batch': replaced, 'loss': float(loss.detach().cpu()),
                            'seconds': time.monotonic() - started})
            print(f'SMOKE PASS {model}/{scenario["id"]} batch={list(values.shape)} ganti_GAN={replaced}', flush=True)
            del network, optimizer, loader; gc.collect()
            if device.type == 'mps': torch.mps.empty_cache()
    write_json(study.run / 'smoke.json', {'experiment_hash': study.signature, 'device': str(device),
                                          'passed': True, 'cases': records, 'weights_discarded': True})


def require_smoke(study, device):
    path = study.run / 'smoke.json'
    if not path.exists() or read_json(path)['experiment_hash'] != study.signature or read_json(path)['device'] != str(device):
        raise RuntimeError('Jalankan smoke (notebook 69) pada device yang sama dahulu')


def cv(study, device, models=None, scenario_ids=None, folds=None, seeds=None):
    require_smoke(study, device)
    for model, scenario, fold, seed in cv_cases(study, models, scenario_ids, folds, seeds):
        run_case(study, model, scenario, device, fold, seed)


def cv_table(study):
    rows = []
    for model, scenario, fold, seed in cv_cases(study):
        path = cv_directory(study, model, scenario['id'], fold, seed) / 'metrics.json'
        if path.exists():
            r = read_json(path)
            rows.append({'model': model, 'scenario': scenario['id'], 'augmentation': scenario['augmentation'],
                         'learning_rate': scenario['learning_rate'], 'batch_size': scenario['batch_size'],
                         'fold': fold, 'seed': seed, 'best_epoch': r['best_epoch'], 'epochs_run': r['epochs_run'],
                         'missing_predicted_classes': r['validation_missing_predicted_classes'], **r['metrics'],
                         **{f'recall_{c}': r['classification_report'][c]['recall'] for c in study.class_names}})
    return pd.DataFrame(rows)


def summarize(study, frame=None):
    frame = cv_table(study) if frame is None else frame
    n = len(study.protocol['folds']) * len(study.protocol['cv_seeds'])
    keys = ['model', 'scenario', 'augmentation', 'learning_rate', 'batch_size']
    summary = (frame.groupby(keys).agg(cases=('f1', 'size'), f1_mean=('f1', 'mean'), f1_sd=('f1', 'std'),
               accuracy_mean=('accuracy', 'mean'), sensitivity_mean=('sensitivity', 'mean'),
               specificity_mean=('specificity', 'mean'), auc_mean=('auc', 'mean'),
               epochs=('best_epoch', 'median'), collapse_cases=('missing_predicted_classes', lambda x: int((x > 0).sum())))
               .reset_index())
    summary['complete'] = summary.cases == n
    return summary


def select(study):
    if progress(study)['cv_complete'] != progress(study)['cv_expected']:
        raise RuntimeError('Seleksi terkunci: seluruh fit CV (baseline + 9 grid x 3 model x 5 fold x seed) belum lengkap')
    frame = cv_table(study); summary = summarize(study, frame)
    out = study.run / 'reports'; out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'cv_cases.csv', index=False); summary.to_csv(out / 'cv_summary.csv', index=False)
    choices = {}
    for model in study.protocol['models']:
        base = summary[(summary.model == model) & (summary.scenario == 'baseline')].iloc[0]
        tuned = (summary[(summary.model == model) & (summary.augmentation == 'gan')]
                 .sort_values(['f1_mean', 'f1_sd', 'scenario'], ascending=[False, True, True]).iloc[0])
        choices[model] = {
            'baseline': {'scenario': 'baseline', 'learning_rate': float(base.learning_rate), 'batch_size': int(base.batch_size),
                         'cv_f1_mean': float(base.f1_mean), 'cv_f1_sd': float(base.f1_sd), 'epochs': int(round(base.epochs))},
            'finetune': {'scenario': tuned.scenario, 'learning_rate': float(tuned.learning_rate), 'batch_size': int(tuned.batch_size),
                         'cv_f1_mean': float(tuned.f1_mean), 'cv_f1_sd': float(tuned.f1_sd), 'epochs': int(round(tuned.epochs))}}
    decision = {'experiment_hash': study.signature, 'choices': choices,
                'selection_source': 'development CV only (5 fold x 3 seed)', 'test_used_for_selection': False,
                'cv_estimate_is_selection_optimistic_for_finetune': True}
    path = study.run / 'selection.json'
    if path.exists() and read_json(path) != decision:
        raise RuntimeError('Seleksi yang sudah dibekukan berubah')
    write_json(path, decision)
    for model, c in choices.items():
        print(f'{model}: baseline CV {c["baseline"]["cv_f1_mean"]:.4f}±{c["baseline"]["cv_f1_sd"]:.4f} '
              f'| fine-tuning {c["finetune"]["scenario"]} LR={c["finetune"]["learning_rate"]} batch={c["finetune"]["batch_size"]} '
              f'CV {c["finetune"]["cv_f1_mean"]:.4f}±{c["finetune"]["cv_f1_sd"]:.4f} epoch final={c["finetune"]["epochs"]}', flush=True)
    return decision


def scenario_of(study, choice):
    return next(s for s in all_scenarios(study.protocol) if s['id'] == choice['scenario'])


def final(study, device, models=None, arms=('baseline', 'finetune')):
    require_smoke(study, device)
    decision = read_json(study.run / 'selection.json')
    if decision['experiment_hash'] != study.signature:
        raise RuntimeError('Seleksi berasal dari run lain')
    for model in models or study.protocol['models']:
        for arm in arms:
            choice = decision['choices'][model][arm]
            for seed in study.protocol['final_seeds']:
                run_case(study, model, scenario_of(study, choice), device, None, seed, arm, choice['epochs'])


def require_test_unlocked(study):
    state = progress(study)
    if not (state['selection_frozen'] and state['final_complete'] == state['final_expected']
            and state['final_gan_quality_passed']):
        raise RuntimeError('Test 20% terkunci: seleksi, GAN final (QC) dan 30 model final harus lengkap dahulu')


def evaluate(study, device):
    require_test_unlocked(study)
    X, table = load_pool(study, 'test')
    scaler_path = study.run / 'scalers/final/scaler.joblib'
    if read_json(study.run / 'scalers/final/provenance.json')['sha256'] != sha256_file(scaler_path):
        raise RuntimeError('Scaler final berubah')
    dataset = PatientBags(X, table, table.hospital_id.unique(), joblib.load(scaler_path))
    cache = {n: sha256_file(study.source / 'data/test' / n) for n in ('X.npy', 'segments.csv', 'manifest.json')}
    for model in study.protocol['models']:
        for arm in ('baseline', 'finetune'):
            for seed in study.protocol['final_seeds']:
                trained = final_directory(study, model, arm, seed)
                directory = test_directory(study, model, arm, seed)
                if sha256_file(trained / 'model.pth') != read_json(trained / 'metrics.json')['model_sha256']:
                    raise RuntimeError('Bobot final berubah')
                with lock(directory / '.lock'):
                    bind_case(directory, {'experiment_hash': study.signature, 'test_cache': cache,
                                          'final_model_sha256': sha256_file(trained / 'model.pth'),
                                          'selection_hash': canonical_hash(read_json(study.run / 'selection.json'))})
                    if (directory / 'metrics.json').exists():
                        continue
                    network = build_model(model, 'mean', study.config['model'], {}).to(device)
                    network.load_state_dict(torch.load(trained / 'model.pth', map_location='cpu', weights_only=True))
                    ids, truth, probability, _ = predict(network, dataset, device, study.protocol['evaluation_segment_chunk'])
                    prediction_table(ids, truth, probability, study.class_names).to_csv(directory / 'patient_predictions.csv', index=False)
                    result = {**report(truth, probability, study.class_names), 'n_patients': len(ids),
                              'test_role': 'historical_holdout_previously_observed'}
                    write_json(directory / 'metrics.json', result)
                    print(f'TEST {model}/{arm} seed={seed} macro-F1={result["metrics"]["f1"]:.4f} '
                          f'acc={result["metrics"]["accuracy"]:.4f} AUC={result["metrics"]["auc"]:.4f}', flush=True)
                    del network; gc.collect()
                    if device.type == 'mps': torch.mps.empty_cache()
