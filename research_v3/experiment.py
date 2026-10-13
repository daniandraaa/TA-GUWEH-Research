"""Paired 2x2 representation/sampling ablation and a classical comparator."""
import gc
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.artifacts import lock, read_json, seed_everything, write_json
from src.data import ECGDataset
from src.evaluation import aggregate_patients, compute_metrics, predict_patient, prediction_table, report
from src.experiment import bind_case, training_settings
from src.preprocessing import prepare_scaler
from src.splits import fold_indices
from src.training import fit_classifier
from . import classical
from .common import disk_check, require_final_complete
from .data import load_pool, representation, prepare
from .sampling import training_loader, patient_class_weights
from .models import build_model


def fit(run, source, cfg, config, model_name, arm, scenario, device,
        fold=None, seed=None, epochs=None):
    tree = model_name == 'extratrees'
    if arm not in (cfg['classical_arms'] if tree else cfg['arms']) or model_name not in cfg['models'] + ['extratrees']:
        raise ValueError('Unknown model/arm')
    if device.type not in {'cpu', 'mps'}:
        raise ValueError('Resume currently supports CPU and MPS only')
    X, segments = load_pool(run, source, arm)
    if fold is not None:
        if not 1 <= fold <= config['n_folds']:
            raise ValueError('Invalid fold')
        manifest = read_json(run / 'folds/manifest.json')[fold - 1]
        train, val = fold_indices(segments, manifest)
        ids = manifest['train_ids']
        names = cfg['models'] + ['extratrees']
        seed = config['seed'] + 100 * fold + 17 * names.index(model_name)
        scope = f'fold_{fold}'
        directory = run / 'cv' / arm / model_name / scenario['id'] / scope
    else:
        if seed not in cfg['final_seeds'] or (not tree and epochs is None):
            raise ValueError('Final requires preregistered seed and CV-derived epochs')
        train, val = np.arange(len(segments)), np.array([], dtype=int)
        ids, scope = segments.hospital_id.unique().tolist(), 'final'
        directory = run / 'final' / model_name / f'seed_{seed}'
    settings = None if tree else training_settings(config, scenario)
    if settings is not None:
        settings['class_weighting'] = cfg['class_weighting']
        if epochs is not None:
            settings['epochs'] = epochs
    identity = {'model': model_name, 'arm': arm, 'scenario': scenario, 'fold': fold,
                'seed': seed, 'train_patient_ids': ids, 'settings': settings,
                'experiment_hash': read_json(run / 'protocol.json')['experiment_hash']}
    with lock(directory / '.lock'):
        bind_case(directory, identity)
        if (directory / 'metrics.json').exists():
            return read_json(directory / 'metrics.json')
        disk_check(run, cfg, 256 * 1024**2)
        print(f'V3 {model_name} {arm} {scenario["id"]} {scope} seed={seed}', flush=True)
        if tree:
            values = classical.features(run, source, arm, config, cfg)
            model = classical.fit_tree(values[train], segments.iloc[train], cfg, scenario, seed)
            if len(val):
                pid, truth, probs = aggregate_patients(segments.iloc[val].label,
                                  model.predict_proba(values[val]), segments.iloc[val].hospital_id)
                prediction_table(pid, truth, probs, config['class_names']).to_csv(directory / 'validation_predictions.csv', index=False)
                result = {**compute_metrics(truth, probs), 'n_validation_patients': len(pid)}
            else:
                temporary = directory / 'model.tmp.joblib'
                joblib.dump(model, temporary, compress=3)
                temporary.replace(directory / 'model.joblib')
                result = {'fit_complete': True, 'n_training_patients': len(ids)}
            write_json(directory / 'metrics.json', result)
        else:
            scaler = prepare_scaler(run / 'scalers' / representation(arm) / scope, X, train, ids)
            seed_everything(seed)
            labels = segments.label.to_numpy()
            real = ECGDataset(X, labels, train, scaler)
            # Hold the number of optimizer steps constant between 2.5 s windows
            # and 0.5 s candidates for a matched fold/LR/batch/epoch budget.
            _, reference_segments = load_pool(run, source, 'window_segment')
            draws = int(reference_segments.hospital_id.isin(ids).sum())
            loader = training_loader(real, segments.iloc[train], settings['batch_size'], seed,
                                     arm.endswith('_patient'), num_samples=draws)
            validation = DataLoader(ECGDataset(X, labels, val, scaler),
                                    batch_size=settings['eval_batch_size']) if len(val) else None
            model = build_model(model_name, config['model']).to(device)
            result = fit_classifier(model, loader, validation,
                        segments.iloc[val].hospital_id.to_numpy() if len(val) else None,
                        device, settings, directory, config['class_names'], resume=True,
                        class_weights=patient_class_weights(segments.iloc[train], 3))
            if fold is not None and not cfg['retain_cv_checkpoints']:
                (directory / 'best.pth').unlink(missing_ok=True)
            del loader, validation, model
        del X
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
        return result


def summarize(run, cfg, config, scenarios, select=False):
    rows, summaries, expected = [], [], 0
    for model in cfg['models'] + ['extratrees']:
        tree = model == 'extratrees'
        grid = classical.scenarios(cfg) if tree else scenarios
        arms = cfg['classical_arms'] if tree else cfg['arms']
        expected += len(grid) * len(arms) * config['n_folds']
        for arm in arms:
            for sc in grid:
                cases = []
                for f in range(1, config['n_folds'] + 1):
                    p = run / 'cv' / arm / model / sc['id'] / f'fold_{f}/metrics.json'
                    if p.exists():
                        row = {'arm': arm, 'model': model, 'scenario': sc['id'], 'fold': f, **read_json(p)}
                        rows.append(row)
                        cases.append(row)
                if len(cases) == config['n_folds']:
                    summaries.append({'arm': arm, 'model': model, 'scenario': sc['id'],
                          **{k:v for k,v in sc.items() if k != 'id'},
                          'f1_mean': float(np.mean([r['f1'] for r in cases])),
                          'f1_std': float(np.std([r['f1'] for r in cases], ddof=1)),
                          'epochs': None if tree else int(np.median([r['best_epoch'] for r in cases]))})
    out = run / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'cv_folds.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'cv_summary.csv', index=False)
    print(f'CV complete: {len(rows)}/{expected}', flush=True)
    if select:
        if len(rows) != expected:
            raise RuntimeError('Pilot/bagian grid tidak cukup untuk seleksi; seluruh 630 kasus harus lengkap pada protokol default.')
        choices = [sorted([r for r in summaries if r['model'] == m],
                         key=lambda r: (-r['f1_mean'], r['f1_std'], r['arm'], r['scenario']))[0]
                   for m in cfg['models'] + ['extratrees']]
        # Primary deep model and classical comparator are reported separately.
        primary = sorted([r for r in choices if r['model'] != 'extratrees'],
                         key=lambda r: (-r['f1_mean'], r['f1_std'], r['model']))[0]
        decision = {'choices': choices, 'primary': primary, 'primary_seed': cfg['primary_seed'],
                    'test_used_for_selection': False, 'selection_source': 'development CV',
                    'cv_estimate_is_selection_optimistic': True,
                    'historical_test_previously_observed': True}
        p = run / 'selection.json'
        if p.exists() and read_json(p) != decision:
            raise RuntimeError('Frozen selection changed')
        write_json(p, decision)
    return pd.DataFrame(summaries)


def final(run, source, cfg, config, scenarios, model, device):
    summarize(run, cfg, config, scenarios, select=True)
    choice = next(c for c in read_json(run / 'selection.json')['choices'] if c['model'] == model)
    grid = classical.scenarios(cfg) if model == 'extratrees' else scenarios
    sc = next(s for s in grid if s['id'] == choice['scenario'])
    for seed in cfg['final_seeds']:
        fit(run, source, cfg, config, model, choice['arm'], sc, device,
            seed=seed, epochs=choice['epochs'])


def evaluate(run, source, cfg, config, v2cfg, device):
    require_final_complete(run, cfg)
    prepare(run, source, cfg, config, v2cfg, 'test')
    for choice in read_json(run / 'selection.json')['choices']:
        name, arm = choice['model'], choice['arm']
        X, segments = load_pool(run, source, arm, 'test')
        for seed in cfg['final_seeds']:
            out = run / 'test' / name / f'seed_{seed}'
            with lock(out / '.lock'):
                if (out / 'metrics.json').exists():
                    continue
                directory = run / 'final' / name / f'seed_{seed}'
                if name == 'extratrees':
                    model = joblib.load(directory / 'model.joblib')
                    values = classical.features(run, source, arm, config, cfg, 'test')
                    ids, truth, probs = aggregate_patients(segments.label, model.predict_proba(values), segments.hospital_id)
                else:
                    scaler = joblib.load(run / 'scalers' / representation(arm) / 'final/scaler.joblib')
                    loader = DataLoader(ECGDataset(X, segments.label, scaler=scaler), batch_size=config['training']['eval_batch_size'])
                    model = build_model(name, config['model']).to(device)
                    model.load_state_dict(torch.load(directory / 'model.pth', map_location='cpu', weights_only=True))
                    ids, truth, probs = predict_patient(model, loader, segments.hospital_id.to_numpy(), device)
                prediction_table(ids, truth, probs, config['class_names']).to_csv(out / 'patient_predictions.csv', index=False)
                write_json(out / 'metrics.json', {**report(truth, probs, config['class_names']),
                           'test_role': 'historical_exploratory_not_external', 'n_patients': len(ids)})
                del model
                gc.collect()
                if device.type == 'mps':
                    torch.mps.empty_cache()
    print('Historical test complete; primary seed 42, no best-test-seed selection.', flush=True)
