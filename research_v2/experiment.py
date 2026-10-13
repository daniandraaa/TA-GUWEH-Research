"""Paired real-data ablation; selection exclusively on development patients."""
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.artifacts import lock, read_json, seed_everything, write_json
from src.data import ECGDataset, load_pool
from src.evaluation import predict_patient, prediction_table, report
from src.experiment import bind_case, training_settings
from src.models import build_model
from src.preprocessing import prepare_scaler
from src.splits import fold_indices
from src.training import fit_classifier
from .common import disk_check
from .data import prepare_pool, require_final_complete


def class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes)
    if len(counts) != n_classes or np.any(counts == 0):
        raise ValueError('Every training class must be present')
    values = 1.0 / counts
    return torch.tensor(values / values.sum() * n_classes, dtype=torch.float32)


def fit(run, cfg, config, model_name, arm, scenario, device, fold=None, seed=None, epochs=None):
    if arm not in cfg['arms'] or model_name not in config['models']:
        raise ValueError('Unknown arm/model')
    if device.type not in {'cpu', 'mps'}:
        raise ValueError('This resume implementation supports CPU/MPS; CUDA RNG support is not implemented.')
    X, segments = load_pool(run)
    if fold is not None:
        manifest = read_json(run / 'folds/manifest.json')[fold - 1]
        train, val = fold_indices(segments, manifest)
        ids = manifest['train_ids']
        seed = config['seed'] + 100 * fold + 17 * config['models'].index(model_name)
        scope = f'fold_{fold}'
        directory = run / 'cv' / arm / model_name / scenario['id'] / scope
    else:
        if seed not in cfg['final_seeds'] or epochs is None:
            raise ValueError('Final requires preregistered seed and CV-derived epochs')
        train, val = np.arange(len(segments)), np.array([], dtype=int)
        ids = segments.hospital_id.unique().tolist()
        scope = 'final'
        directory = run / 'final' / model_name / f'seed_{seed}'
    settings = training_settings(config, scenario)
    settings['class_weighting'] = 'inverse_frequency_training_fold' if arm == 'weighted_real' else 'none'
    if epochs is not None:
        settings['epochs'] = epochs
    identity = {'model': model_name, 'arm': arm, 'scenario': scenario, 'fold': fold, 'seed': seed,
                'settings': settings, 'train_patient_ids': ids,
                'experiment_hash': read_json(run / 'protocol.json')['experiment_hash']}
    with lock(directory / '.lock'):
        bind_case(directory, identity)
        if (directory / 'metrics.json').exists():
            return read_json(directory / 'metrics.json')
        disk_check(run, cfg, 128 * 1024**2)
        scaler = prepare_scaler(run / 'folds' / scope, X, train, ids)
        labels = segments.label.to_numpy()
        weights = class_weights(labels[train], len(config['class_names'])) if arm == 'weighted_real' else None
        seed_everything(seed)
        train_loader = DataLoader(ECGDataset(X, labels, train, scaler), batch_size=settings['batch_size'],
                                  shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
        val_loader = DataLoader(ECGDataset(X, labels, val, scaler), batch_size=settings['eval_batch_size']) if len(val) else None
        model = build_model(model_name, config['model']).to(device)
        print(f'{arm} {model_name} {scenario["id"]} {scope} seed={seed}', flush=True)
        result = fit_classifier(model, train_loader, val_loader,
                                segments.iloc[val].hospital_id.to_numpy() if len(val) else None,
                                device, settings, directory, config['class_names'], resume=True, class_weights=weights)
        if fold is not None and not cfg['retain_cv_checkpoints']:
            # Only this run's completed, reproducible CV weights; final weights stay available.
            (directory / 'best.pth').unlink(missing_ok=True)
        del model, train_loader, val_loader, X
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
        return result


def summarize(run, cfg, config, scenarios, select=False):
    rows, summaries = [], []
    for arm in cfg['arms']:
        for model in config['models']:
            for sc in scenarios:
                values = []
                for fold in range(1, config['n_folds'] + 1):
                    path = run / 'cv' / arm / model / sc['id'] / f'fold_{fold}' / 'metrics.json'
                    if path.exists():
                        row = {'arm': arm, 'model': model, 'scenario': sc['id'], 'fold': fold, **read_json(path)}
                        rows.append(row)
                        values.append(row)
                if len(values) == config['n_folds']:
                    summaries.append({'arm': arm, 'model': model, 'scenario': sc['id'],
                                      'f1_mean': float(np.mean([v['f1'] for v in values])),
                                      'f1_std': float(np.std([v['f1'] for v in values], ddof=1)),
                                      'epochs': int(np.median([v['best_epoch'] for v in values]))})
    report_dir = run / 'reports'
    report_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(report_dir / 'cv_folds.csv', index=False)
    pd.DataFrame(summaries).to_csv(report_dir / 'cv_summary.csv', index=False)
    expected = len(cfg['arms']) * len(config['models']) * len(scenarios) * config['n_folds']
    print(f'CV complete: {len(rows)}/{expected}', flush=True)
    if select:
        if len(rows) != expected:
            raise RuntimeError('Semua arm/model/scenario/fold harus lengkap sebelum seleksi.')
        choices = [sorted([s for s in summaries if s['model'] == model],
                          key=lambda s: (-s['f1_mean'], s['f1_std'], s['arm'], s['scenario']))[0]
                   for model in config['models']]
        primary = sorted(choices, key=lambda s: (-s['f1_mean'], s['f1_std'], s['model']))[0]
        decision = {'choices': choices, 'primary': primary, 'primary_seed': cfg['final_seeds'][0],
                    'test_used_for_selection': False, 'selection_source': 'development CV',
                    'cv_estimate_is_selection_optimistic': True}
        path = run / 'selection.json'
        if path.exists() and read_json(path) != decision:
            raise RuntimeError('Frozen selection changed')
        write_json(path, decision)
    return pd.DataFrame(summaries)


def train_finals(run, cfg, config, scenarios, model, device):
    summarize(run, cfg, config, scenarios, select=True)
    choice = next(c for c in read_json(run / 'selection.json')['choices'] if c['model'] == model)
    sc = next(s for s in scenarios if s['id'] == choice['scenario'])
    for seed in cfg['final_seeds']:
        fit(run, cfg, config, model, choice['arm'], sc, device, seed=seed, epochs=choice['epochs'])


def evaluate(run, source, cfg, config, device):
    import joblib
    require_final_complete(run, cfg, config)
    prepare_pool(run, source, cfg, config, 'test')
    X, segments = load_pool(run, 'test')
    scaler = joblib.load(run / 'folds/final/scaler.joblib')
    loader = DataLoader(ECGDataset(X, segments.label, scaler=scaler), batch_size=config['training']['eval_batch_size'])
    for name in config['models']:
        for seed in cfg['final_seeds']:
            out = run / 'test' / name / f'seed_{seed}'
            with lock(out / '.lock'):
                if (out / 'metrics.json').exists():
                    continue
                model = build_model(name, config['model']).to(device)
                model.load_state_dict(torch.load(run / 'final' / name / f'seed_{seed}/model.pth', map_location='cpu', weights_only=True))
                ids, truth, probs = predict_patient(model, loader, segments.hospital_id.to_numpy(), device)
                prediction_table(ids, truth, probs, config['class_names']).to_csv(out / 'patient_predictions.csv', index=False)
                write_json(out / 'metrics.json', {**report(truth, probs, config['class_names']), 'test_role': 'historical_not_external'})
                del model
    print('Historical test complete; no selection by test score.', flush=True)
