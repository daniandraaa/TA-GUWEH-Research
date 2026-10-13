"""Manual preparation, disposable memory check, then paired development CV."""
import gc
import shutil
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.artifacts import canonical_hash, lock, read_json, seed_everything, sha256_file, write_json
from research_v3.sampling import patient_class_weights
from research_v4.common import disk_check
from research_v4.data import PatientBags, collate_bags, load_pool, scaler_for
from research_v4.models import build_model
from research_v4.experiment import fit_case
from .common import case_directory, check_case, require_prepared, require_smoke, scenario, progress


def prepare(study):
    table = pd.read_csv(study.source / 'data/development/segments.csv', dtype={'hospital_id': str})
    if set(table.hospital_id) != {r['hospital_id'] for r in study.records}:
        raise RuntimeError('Development cache cohort changed')
    with lock(study.run / '.prepare.lock'):
        for fold in study.folds:
            ids = fold['train_ids']
            indices = np.flatnonzero(table.hospital_id.isin(ids).to_numpy())
            relative = f'scalers/fold_{fold["fold"]}'
            marker = read_json(study.baseline / relative / 'provenance.json')
            expected = {'training_patient_ids': sorted(ids), 'indices_hash': canonical_hash(indices.tolist()),
                        'n_segments': len(indices), 'development_cache': study.identity['development_cache']}
            if marker['spec'] != expected:
                raise RuntimeError('V4 scaler was not fitted on this training fold')
            target = study.run / relative; target.mkdir(parents=True, exist_ok=True)
            for name in ('scaler.joblib', 'provenance.json'):
                origin = study.baseline / relative / name
                if (target / name).exists():
                    if sha256_file(target / name) != sha256_file(origin):
                        raise RuntimeError('Existing copied scaler changed')
                else:
                    shutil.copy2(origin, target / name)
        write_json(study.run / 'preparation.json', {'experiment_hash': study.signature,
            'development_patients': 230, 'development_segments': len(table),
            'controls_reused': 30, 'new_fits_expected': 30, 'scaler_fit_performed': False,
            'training_started': False, 'test_signals_read': False,
            'patient_batch_size': 8, 'training_windows_per_patient': 8, 'segment_budget': 64})
    require_prepared(study)
    print('Prepared: 230 development patients; 30 verified controls; 30 new fits planned. No training.', flush=True)


def smoke(study, models, device):
    require_prepared(study)
    if str(device) != study.identity['paired_device']:
        raise RuntimeError('Gunakan device yang sama dengan kontrol V4')
    with lock(study.run / '.training.lock'):
        for model in models:
            path = study.run / 'smoke' / f'{model}.json'
            if path.exists():
                require_smoke(study, model, device)
                print(f'SMOKE cached: {model}', flush=True)
                continue
            disk_check(study.run, study.training_cfg)
            X, table = load_pool(study.run, study.source, study.training_cfg)
            fold = study.folds[0]
            indices = np.flatnonzero(table.hospital_id.isin(fold['train_ids']).to_numpy())
            scaler = scaler_for(study.run, 'fold_1', X, table, indices, fold['train_ids'])
            counts = table[table.hospital_id.isin(fold['train_ids'])].groupby('hospital_id').size()
            ids = sorted(counts[counts >= 8].index)[:8]
            if len(ids) != 8:
                raise RuntimeError('Need eight training patients with eight windows for memory check')
            seed_everything(142)
            dataset = PatientBags(X, table, ids, scaler, 8, sample_seed=10142)
            loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=collate_bags)
            network = build_model(model, study.choices[model]['pooling'],
                                  study.inherited['model'], study.training_cfg['new_model']).to(device)
            optimizer = torch.optim.AdamW(network.parameters(), lr=study.choices[model]['learning_rate'],
                                         weight_decay=study.training_cfg['training']['weight_decay'])
            seed_everything(20142)
            began = time.monotonic()
            values = labels = mask = logp = loss = weights = norm = None
            try:
                values, labels, mask, batch_ids = next(iter(loader))
                if list(values.shape) != [8, 8, 12, 5000] or int(mask.sum()) != 64:
                    raise RuntimeError('Smoke must exercise the full 64-segment budget')
                network.train(); optimizer.zero_grad(set_to_none=True)
                weights = patient_class_weights(table.iloc[indices], 3).to(device)
                logp = network(values.to(device), mask.to(device))
                loss = F.nll_loss(logp, labels.to(device), weight=weights)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite smoke loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(network.parameters(),
                    study.training_cfg['training']['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in network.parameters()):
                    raise RuntimeError('Nonfinite smoke parameters')
                if device.type == 'mps': torch.mps.synchronize()
                record = {'experiment_hash': study.signature, 'device': str(device), 'model': model,
                    'batch_shape': list(values.shape), 'valid_segments': int(mask.sum()),
                    'training_patient_ids': batch_ids, 'disposable_optimizer_steps': 1,
                    'loss': float(loss.detach().cpu()), 'gradient_norm': float(norm.detach().cpu()),
                    'seconds': time.monotonic() - began, 'passed': True,
                    'classifier_checkpoint_created': False, 'test_signals_read': False}
                write_json(path, {'record': record, 'record_hash': canonical_hash(record)})
                print(f'SMOKE PASS {model}: 8 patients × 8 windows; disposable weights discarded.', flush=True)
            except RuntimeError as error:
                if 'out of memory' in str(error).lower():
                    raise RuntimeError('K8 memory check failed. Close other GPU processes and repeat. '
                        'Batch/protocol is not reduced automatically.') from error
                raise
            finally:
                del network, optimizer, dataset, loader, X, values, labels, mask, logp, loss, weights, norm
                gc.collect()
                if device.type == 'mps': torch.mps.empty_cache()


def train(study, models, folds, seeds, device, pilot=False):
    require_prepared(study)
    for model in models:
        require_smoke(study, model, device)
        if not pilot:
            initial = case_directory(study, model, 1, 42)
            if not (initial / 'metrics.json').exists():
                raise RuntimeError(f'Jalankan pilot {model} fold1 seed42 dahulu')
            check_case(study, model, 1, 42)
    with lock(study.run / '.training.lock'):
        for model in models:
            for fold in ([1] if pilot else folds):
                for seed in ([42] if pilot else seeds):
                    directory = case_directory(study, model, fold, seed)
                    if (directory / 'metrics.json').exists():
                        check_case(study, model, fold, seed)
                        print(f'REUSE completed candidate: {model}/fold{fold}/seed{seed}', flush=True)
                        continue
                    print(f'K8: {model}/fold{fold}/seed{seed}; patient_batch=8, segment_budget=64', flush=True)
                    fit_case(study.run, study.source, study.training_cfg, study.inherited, model,
                             study.choices[model]['pooling'], scenario(study, model), device,
                             fold=fold, seed=seed, phase='ablation')
                    check_case(study, model, fold, seed)
    p = progress(study)
    print({k: v for k, v in p.items() if k != 'missing'}, flush=True)
