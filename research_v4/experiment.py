"""Manual endpoints: native V3 diagnosis, controlled MIL grid, stability, final."""
import gc
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.artifacts import canonical_hash, lock, read_json, seed_everything, sha256_file, write_json
from src.evaluation import prediction_table, report
from src.experiment import bind_case, training_settings
from src.splits import fold_indices
from research_v3.sampling import patient_class_weights, training_loader
from .common import cv_directory, disk_check, progress, require_final_complete, scenarios
from .data import PatientBags, SegmentsWithIdentity, collate_bags, load_pool, scaler_for
from .models import build_model
from .training import compact_completed, fit, predict


def prepare(run, source, old, cfg, config):
    X, table = load_pool(run, source, cfg)
    folds = read_json(run / 'folds/manifest.json')
    rows = []
    for fold in folds:
        train, val = fold_indices(table, fold)
        scaler_for(run, f'fold_{fold["fold"]}', X, table, train, fold['train_ids'])
        rows.append({'fold': fold['fold'], 'train_patients': len(fold['train_ids']),
                     'validation_patients': len(fold['validation_ids']),
                     'train_segments': len(train), 'validation_segments': len(val)})
    scaler_for(run, 'final', X, table, np.arange(len(X)), table.hospital_id.unique().tolist())
    out = run / 'reports'; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'fold_inventory.csv', index=False)
    table.groupby(['hospital_id', 'label']).size().rename('n_segments').reset_index().to_csv(out / 'patient_inventory.csv', index=False)
    model_inventory = []
    for name in cfg['models']:
        for pooling in cfg['pooling']:
            network = build_model(name, pooling, config['model'], cfg['new_model'])
            model_inventory.append({'model': name, 'pooling': pooling,
                'parameters': sum(p.numel() for p in network.parameters()),
                'checkpoint_tensor_bytes': sum(t.numel() * t.element_size() for t in network.state_dict().values())})
            del network
    pd.DataFrame(model_inventory).to_csv(out / 'model_inventory.csv', index=False)
    final_weight_bytes = sum(max(r['checkpoint_tensor_bytes'] for r in model_inventory if r['model'] == m)
                             for m in cfg['models']) * len(cfg['final_seeds'])
    write_json(out / 'preparation.json', {'n_development_patients': table.hospital_id.nunique(),
        'n_development_segments': len(table), 'sampling_rate': cfg['sampling_rate'],
        'segment_seconds': cfg['segment_length'] / cfg['sampling_rate'], 'test_signal_read': False,
        'expected_cv_fits': progress(run, cfg)['expected'], 'expected_diagnostic_fits': 45,
        'expected_stability_cases': 75, 'additional_stability_fits': 50, 'expected_final_fits': 25,
        'no_new_waveform_cache': True, 'no_real_training_started': True,
        'estimated_final_tensor_storage_mib_upper': final_weight_bytes / 1024**2,
        'storage_estimate_excludes_logs_predictions_serialization_overhead_and_active_optimizer': True})
    print(f'Prepared {len(X)} shared windows, {table.hospital_id.nunique()} development patients. '
          'Historical test signal remains closed.', flush=True)


def fit_case(run, source, cfg, config, model, pooling, scenario, device, fold=None, seed=42,
             phase='cv', epochs=None, diagnostic_choice=None, recipe='Adam'):
    if device.type not in {'cpu', 'mps'}:
        raise ValueError('V4 epoch resume supports CPU/MPS; CUDA has not been validated')
    if model not in cfg['models'] or pooling not in cfg['pooling']:
        raise ValueError('Unknown model/pooling')
    X, table = load_pool(run, source, cfg)
    if fold is not None:
        if fold not in range(1, 6):
            raise ValueError('fold must be 1–5')
        manifest = read_json(run / 'folds/manifest.json')[fold - 1]
        train, val = fold_indices(table, manifest)
        ids, scope = manifest['train_ids'], f'fold_{fold}'
    else:
        if phase != 'final' or seed not in cfg['final_seeds'] or not epochs:
            raise ValueError('Final requires a preregistered seed and CV-derived epochs')
        train, val = np.arange(len(X)), np.asarray([], dtype=int)
        ids, scope = table.hospital_id.unique().tolist(), 'final'
    if phase == 'diagnostic':
        settings = training_settings(config, scenario)
        settings['optimizer'] = recipe
        settings['class_weighting'] = 'inverse_training_patient_class_frequency'
        directory = run / 'diagnostic' / recipe / model / scope / f'seed_{seed}'
        effective_seed = seed + 100 * fold + 17 * ['encoder', 'patchtst', 'hierarchical'].index(model)
        mode = 'segment'
    else:
        settings = {**cfg['training'], **scenario, 'scheduler_t_max': 100}
        directory = (cv_directory(run, model, pooling, scenario['id'], fold, seed) if phase == 'cv'
                     else run / 'final' / model / f'seed_{seed}' if phase == 'final'
                     else run / phase / model / scope / f'seed_{seed}')
        effective_seed = seed + (100 * fold if fold else 0)
        mode = 'bag'
    if epochs is not None:
        settings['epochs'] = int(epochs)
    identity = {'experiment_hash': read_json(run / 'protocol.json')['experiment_hash'],
        'device': str(device),
        'phase': phase, 'model': model, 'pooling': pooling, 'scenario': scenario, 'fold': fold,
        'seed': seed, 'effective_seed': effective_seed, 'settings': settings, 'training_mode': mode,
        'train_patient_ids': sorted(ids), 'validation_patient_ids': sorted(manifest['validation_ids']) if fold else [],
        'diagnostic_v3_choice': diagnostic_choice,
        'segment_budget': settings['batch_size'],
        'patient_batch_size': settings['batch_size'] // cfg['segments_per_patient'] if mode == 'bag' else None}
    with lock(directory / '.lock'):
        bind_case(directory, identity)
        if (directory / 'metrics.json').exists():
            result = read_json(directory / 'metrics.json')
            if (directory / 'best.pth').exists():
                compact_completed(directory, result)
            return result
        disk_check(run, cfg)
        scaler = scaler_for(run, scope, X, table, train, ids)
        seed_everything(effective_seed)
        if mode == 'bag':
            data = PatientBags(X, table, ids, scaler, cfg['segments_per_patient'], sample_seed=effective_seed + 10000)
            loader = DataLoader(data, batch_size=identity['patient_batch_size'], shuffle=True,
                generator=torch.Generator().manual_seed(effective_seed), num_workers=0, collate_fn=collate_bags)
        else:
            data = SegmentsWithIdentity(X, table, train, scaler)
            loader = training_loader(data, table.iloc[train], settings['batch_size'], effective_seed,
                                     diagnostic_choice['arm'].endswith('_patient'), num_samples=len(train))
        validation = PatientBags(X, table, manifest['validation_ids'], scaler) if fold else None
        network = build_model(model, pooling, config['model'], cfg['new_model']).to(device)
        if mode == 'bag':
            # Attention-gate initialization must not shift paired dropout RNG.
            seed_everything(effective_seed + 20000)
        print(f'V4 {phase} {model}/{pooling} {scenario["id"]} {scope} seed={seed} '
              f'effective={effective_seed} segment_budget={settings["batch_size"]} '
              f'patient_batch={identity["patient_batch_size"]} optimizer={settings["optimizer"]}', flush=True)
        try:
            result = fit(network, loader, validation, device, settings, directory, config['class_names'],
                         patient_class_weights(table.iloc[train], 3), cfg['evaluation_segment_chunk'], mode)
        except RuntimeError as error:
            if 'out of memory' in str(error).lower():
                raise RuntimeError('Memori perangkat penuh. Tutup proses GPU lain dan lanjutkan sel yang sama. '
                    'Batch/protokol tidak diubah otomatis; jika perlu recipe lebih kecil, buat protokol/run baru.') from error
            raise
        finally:
            del network, loader, validation, data, X
            gc.collect()
            if device.type == 'mps':
                torch.mps.empty_cache()
        return result


def diagnostic(run, source, old, cfg, config, device, models, folds, recipe='Adam'):
    choices = read_json(old / 'selection.json')['choices']
    for model in models:
        choice = next(c for c in choices if c['model'] == model)
        sc = next(s for s in scenarios(cfg) if s['id'] == choice['scenario'])
        for fold in folds:
            for seed in cfg['diagnostic_seeds']:
                fit_case(run, source, cfg, config, model, 'mean', sc, device, fold, seed,
                         'diagnostic', diagnostic_choice=choice, recipe=recipe)


def cv(run, source, cfg, config, device, models, pools, selected_scenarios, folds):
    for model in models:
        for pooling in pools:
            for sc in selected_scenarios:
                for fold in folds:
                    fit_case(run, source, cfg, config, model, pooling, sc, device, fold, cfg['cv_seed'])


def summarize(run, cfg, select=False):
    with lock(run / 'reports/.summary.lock'):
        return _summarize(run, cfg, select)


def _summarize(run, cfg, select=False):
    rows, summaries = [], []
    for m in cfg['models']:
        for p in cfg['pooling']:
            for sc in scenarios(cfg):
                cases = []
                for f in range(1, 6):
                    path = cv_directory(run, m, p, sc['id'], f, cfg['cv_seed']) / 'metrics.json'
                    if path.exists():
                        result = read_json(path)
                        row = {'model': m, 'pooling': p, 'scenario': sc['id'], 'fold': f,
                               'best_epoch': result['best_epoch'], 'epochs_run': result['epochs_run'],
                               'missing_predicted_classes': result['validation_missing_predicted_classes'],
                               **result['metrics']}
                        rows.append(row); cases.append(row)
                if len(cases) == 5:
                    summaries.append({'model': m, 'pooling': p, 'scenario': sc['id'],
                        'learning_rate': sc['learning_rate'], 'batch_size': sc['batch_size'],
                        'patient_batch_size': sc['batch_size'] // cfg['segments_per_patient'],
                        'f1_mean': float(np.mean([r['f1'] for r in cases])),
                        'f1_std': float(np.std([r['f1'] for r in cases], ddof=1)),
                        'accuracy_mean': float(np.mean([r['accuracy'] for r in cases])),
                        'epochs': int(np.median([r['best_epoch'] for r in cases])),
                        'collapse_folds': sum(r['missing_predicted_classes'] > 0 for r in cases)})
    directory = run / 'reports'; directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / 'cv_folds.csv', index=False)
    pd.DataFrame(summaries).to_csv(directory / 'cv_summary.csv', index=False)
    status = progress(run, cfg)
    write_json(directory / 'cv_progress.json', status)
    print(f'CV {status["complete"]}/{status["expected"]}', flush=True)
    if select:
        if not status['ready']:
            raise RuntimeError('Pilot belum cukup untuk final. Lengkapi 450 kasus grid pada protokol default.')
        missing = [f'{m}/fold_{f}/seed_{s}' for m in cfg['models'][:3] for f in range(1, 6)
                   for s in cfg['diagnostic_seeds'] if not (run / 'diagnostic/Adam' / m / f'fold_{f}' / f'seed_{s}/metrics.json').is_file()]
        if missing:
            raise RuntimeError('Selesaikan diagnosis native V3 terlebih dahulu: ' + ', '.join(missing[:5]))
        choices = [sorted([r for r in summaries if r['model'] == m],
                          key=lambda r: (-r['f1_mean'], r['f1_std'], r['pooling'], r['scenario']))[0] for m in cfg['models']]
        primary = sorted(choices, key=lambda r: (-r['f1_mean'], r['f1_std'], r['model']))[0]
        decision = {'choices': choices, 'primary': primary, 'primary_seed': cfg['primary_seed'],
            'selection_source': 'development five-fold CV only', 'test_used_for_selection': False,
            'cv_estimate_is_selection_optimistic': True, 'historical_test_previously_observed': True,
            'stability_used_to_pick_best_seed': False}
        if (run / 'selection.json').exists() and read_json(run / 'selection.json') != decision:
            raise RuntimeError('Frozen selection changed')
        write_json(run / 'selection.json', decision)
        print('Selection frozen:', primary, flush=True)
    return pd.DataFrame(summaries)


def stability(run, source, cfg, config, device, models):
    summarize(run, cfg, select=True)
    choices = read_json(run / 'selection.json')['choices']
    for model in models:
        choice = next(c for c in choices if c['model'] == model)
        sc = next(s for s in scenarios(cfg) if s['id'] == choice['scenario'])
        for f in range(1, 6):
            for seed in cfg['stability_seeds']:
                target = run / 'stability' / model / f'fold_{f}' / f'seed_{seed}'
                if seed == cfg['cv_seed']:
                    # Identical case: reuse its predictions/history instead of fitting again.
                    origin = cv_directory(run, model, choice['pooling'], sc['id'], f, seed)
                    with lock(target / '.lock'):
                        if not (target / 'metrics.json').exists():
                            write_json(target / 'config.json', {'reused_cv_case': str(origin.relative_to(run)),
                                'source_config_sha256': sha256_file(origin / 'config.json'),
                                'source_metrics_sha256': sha256_file(origin / 'metrics.json')})
                            write_json(target / 'metrics.json', read_json(origin / 'metrics.json'))
                else:
                    fit_case(run, source, cfg, config, model, choice['pooling'], sc, device, f, seed, 'stability')


def final(run, source, cfg, config, device, models):
    summarize(run, cfg, select=True)
    choices = read_json(run / 'selection.json')['choices']
    for model in models:
        choice = next(c for c in choices if c['model'] == model)
        for seed in cfg['final_seeds']:
            sc = next(s for s in scenarios(cfg) if s['id'] == choice['scenario'])
            fit_case(run, source, cfg, config, model, choice['pooling'], sc, device, seed=seed,
                     phase='final', epochs=choice['epochs'])


def evaluate(run, source, cfg, config, device):
    require_final_complete(run, cfg)
    X, table = load_pool(run, source, cfg, 'test')
    test_identity = {name: sha256_file(source / 'data/test' / name) for name in ('X.npy', 'segments.csv', 'manifest.json')}
    if (run / 'test_cache.json').exists() and read_json(run / 'test_cache.json') != test_identity:
        raise RuntimeError('Historical test cache changed')
    write_json(run / 'test_cache.json', test_identity)
    scaler = joblib.load(run / 'scalers/final/scaler.joblib')
    marker = read_json(run / 'scalers/final/provenance.json')
    if marker['sha256'] != sha256_file(run / 'scalers/final/scaler.joblib'):
        raise RuntimeError('Final scaler changed')
    dataset = PatientBags(X, table, table.hospital_id.unique(), scaler)
    for choice in read_json(run / 'selection.json')['choices']:
        for seed in cfg['final_seeds']:
            directory = run / 'test' / choice['model'] / f'seed_{seed}'
            trained = run / 'final' / choice['model'] / f'seed_{seed}'
            if sha256_file(trained / 'model.pth') != read_json(trained / 'metrics.json')['model_sha256']:
                raise RuntimeError('Final weights changed')
            with lock(directory / '.lock'):
                binding = {'experiment_hash': read_json(run / 'protocol.json')['experiment_hash'],
                    'selection_hash': canonical_hash(read_json(run / 'selection.json')),
                    'test_cache': test_identity, 'final_model_sha256': sha256_file(trained / 'model.pth'),
                    'final_scaler_sha256': marker['sha256']}
                bind_case(directory, binding)
                if (directory / 'metrics.json').exists():
                    continue
                network = build_model(choice['model'], choice['pooling'], config['model'], cfg['new_model']).to(device)
                network.load_state_dict(torch.load(trained / 'model.pth', map_location='cpu', weights_only=True))
                ids, truth, probabilities, contributions = predict(network, dataset, device, cfg['evaluation_segment_chunk'])
                prediction_table(ids, truth, probabilities, config['class_names']).to_csv(directory / 'patient_predictions.csv', index=False)
                pd.DataFrame(contributions).to_csv(directory / 'segment_weights.csv', index=False)
                write_json(directory / 'metrics.json', {**report(truth, probabilities, config['class_names']),
                    'test_role': 'historical_exploratory_not_external', 'n_patients': len(ids)})
                print(f'Historical test {choice["model"]} seed={seed} macro-F1={report(truth, probabilities, config["class_names"])["metrics"]["f1"]:.4f}', flush=True)
                del network; gc.collect()
                if device.type == 'mps':
                    torch.mps.empty_cache()
