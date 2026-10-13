"""Preparation is read-only on V8; real classifier operations are manual endpoints."""
import gc
import shutil
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from src.artifacts import ROOT, lock, read_json, seed_everything, sha256_file, write_json
from src.evaluation import prediction_table, report
from research_v4.common import disk_check
from research_v4.training import predict, class_diagnostics
from research_v8.common import directory as v8_directory
from research_v8.experiment import setup
from research_v8.training import fit
from .common import (directory, case_for, checked_complete, entry_for, settings_for,
    require_prepared, storage_check, validate_reuse, verify_files, write_manifest)


def prepare(study):
    with lock(study.run / '.prepare.lock'):
        if (study.run / 'preparation.json').exists():
            require_prepared(study); print('PREPARE cached; classifier steps=0', flush=True); return
        storage = storage_check(study)
        files = {}
        for fold in study.protocol['folds']:
            for name in ('scaler.joblib', 'provenance.json'):
                origin = study.parent.run / f'scalers/outer_{fold}' / name
                target = study.run / f'scalers/outer_{fold}' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and sha256_file(target) != sha256_file(origin): raise RuntimeError('Copied scaler changed')
                if not target.exists(): shutil.copy2(origin, target)
                files[str(target.relative_to(study.run))] = sha256_file(target)
        pd.DataFrame(study.plan).to_csv(study.run / 'epoch_plan.csv', index=False)
        files['epoch_plan.csv'] = sha256_file(study.run / 'epoch_plan.csv')
        for row in study.plan:
            if not row['reuse']: continue
            folder = directory(study, row['model'], row['fold'], row['seed'])
            case = case_for(study, row, row['source_device'])
            old = validate_reuse(study, row, case)
            if (folder / 'completed.json').exists():
                checked_complete(study, row['model'], row['fold'], row['seed']); continue
            write_json(folder / 'case.json', case)
            write_json(folder / 'reuse.json', {'source_folder': str(old.relative_to(ROOT)),
                'completed_sha256': sha256_file(old / 'completed.json')})
            write_manifest(folder / 'completed.json', {'experiment_hash': study.signature,
                'case_hash': case['case_hash'], 'kind': 'v8_reference',
                'files': {n: sha256_file(folder / n) for n in ('case.json', 'reuse.json')}})
        write_manifest(study.run / 'preparation.json', {'experiment_hash': study.signature, 'files': files,
            'classifier_optimizer_steps': 0, 'reused_cases': sum(r['reuse'] for r in study.plan),
            'new_fits_expected': sum(not r['reuse'] for r in study.plan), 'storage_estimate': storage,
            'test_signals_read': False, 'new_waveform_cache': False, 'new_inner_fits': 0})
    print('Prepared: 10 exact V8 references + 20 NEW real-only fits; classifier steps=0.', flush=True)


def require_smoke(study, model, device):
    path = study.run / f'smoke/{model}.json'
    if not path.exists(): raise RuntimeError(f'Jalankan smoke notebook 46 untuk {model} dahulu')
    saved = read_json(path); verify_files(study.run, saved)
    if saved['experiment_hash'] != study.signature or saved['device'] != str(device) or not saved['passed']:
        raise RuntimeError('Smoke must pass on the same device')


def smoke(study, models, device):
    require_prepared(study)
    with lock(study.run / '.training.lock'):
        for model in models:
            if str(device) != entry_for(study, model, 1, 42)['source_device']:
                raise RuntimeError('Use the same device as the V8 control for this paired comparison')
            path = study.run / f'smoke/{model}.json'
            if path.exists(): require_smoke(study, model, device); print(f'SMOKE cached {model}', flush=True); continue
            disk_check(study.run, study.protocol); seed_everything(142)
            network, loader, validation, weights, _ = setup(study, model, 1, 42, 'real_only')
            optimizer = values = labels = mask = loss = None
            try:
                eligible = [i for i, ix in enumerate(loader.dataset.indices) if len(ix) >= 4][:8]
                if len(eligible) != 8: raise RuntimeError('Eight K4 training bags required')
                values, labels, mask, ids, metadata = loader.collate_fn([loader.dataset[i] for i in eligible])
                network = network.to(device); network.train()
                optimizer = torch.optim.AdamW(network.parameters(), lr=study.choices[model]['learning_rate'],
                    weight_decay=study.training['weight_decay'])
                began = time.monotonic(); optimizer.zero_grad(set_to_none=True)
                loss = F.nll_loss(network(values.to(device), mask.to(device)), labels.to(device), weight=weights.to(device))
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite smoke loss')
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(network.parameters(), study.training['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in network.parameters()): raise RuntimeError('Nonfinite smoke parameters')
                if any(r['replacements'] for r in metadata): raise RuntimeError('Real-only smoke replaced a window')
                write_manifest(path, {'experiment_hash': study.signature, 'model': model, 'device': str(device),
                    'passed': True, 'shape': list(values.shape), 'loss': float(loss.detach().cpu()),
                    'gradient_norm': float(norm.detach().cpu()), 'seconds': time.monotonic() - began,
                    'disposable_optimizer_steps': 1, 'weights_discarded': True,
                    'test_signals_read': False, 'outer_validation_signals_read': False, 'files': {}})
                print(f'SMOKE PASS {model}: {list(values.shape)}; disposable weights discarded', flush=True)
            finally:
                # Only disposable training weights; no validation inference here.
                del network, loader, validation, optimizer, values, labels, mask, loss
                gc.collect()
                if device.type == 'mps': torch.mps.empty_cache()


def fit_case(study, model, fold, seed, device):
    require_prepared(study); row = entry_for(study, model, fold, seed)
    if str(device) != row['source_device']: raise RuntimeError('Use the same device as the V8 real-only control')
    folder = directory(study, model, fold, seed)
    if (folder / 'completed.json').exists():
        checked_complete(study, model, fold, seed); print(f'REUSE {model}/fold{fold}/seed{seed}', flush=True); return
    if row['reuse']: raise RuntimeError('Run prepare to register the exact V8 reference')
    require_smoke(study, model, device); storage_check(study)
    case = case_for(study, row, device)
    if (folder / 'case.json').exists() and read_json(folder / 'case.json') != case: raise RuntimeError('Case/device changed')
    write_json(folder / 'case.json', case)
    seed_everything(case['effective_seed'])
    network, loader, validation, weights, effective = setup(study, model, fold, seed, 'real_only')
    network = network.to(device); seed_everything(effective + 20000)
    try:
        print(f'TRAIN {model}/fold{fold}/seed{seed}; median inner epoch={row["epochs"]}; no outer callbacks', flush=True)
        fitted = fit(network, loader, None, device, settings_for(study, row), folder,
            case['case_hash'], weights, study.protocol['evaluation_segment_chunk'])
        network.load_state_dict(torch.load(folder / 'model.pth', map_location='cpu', weights_only=True))
        ids, truth, probability, contributions = predict(network, validation, device, study.protocol['evaluation_segment_chunk'])
        prediction_table(ids, truth, probability, study.inherited['class_names']).to_csv(folder / 'predictions.csv', index=False)
        pd.DataFrame(contributions).to_csv(folder / 'segment_weights.csv', index=False)
        result = {**report(truth, probability, study.inherited['class_names']), **class_diagnostics(truth, probability),
            **fitted, 'evaluation_role': 'outer_real_patient_scoring', 'test_signals_read': False}
        write_json(folder / 'metrics.json', result)
        names = ('case.json', 'metrics.json', 'predictions.csv', 'segment_weights.csv', 'history.csv',
                 'fit_result.json', 'fit_finished.json', 'model.pth', 'last.pt')
        write_manifest(folder / 'completed.json', {'experiment_hash': study.signature, 'case_hash': case['case_hash'],
            'kind': 'new_fixed_epoch_fit', 'files': {n: sha256_file(folder / n) for n in names}})
    finally:
        del network, loader, validation; gc.collect()
        if device.type == 'mps': torch.mps.empty_cache()


def train(study, models, folds, seeds, device, pilot=False):
    require_prepared(study)
    if not pilot:
        for model in models: checked_complete(study, model, 1, 42)
    with lock(study.run / '.training.lock'):
        for model in models:
            for fold in ([1] if pilot else folds):
                for seed in ([42] if pilot else seeds): fit_case(study, model, fold, seed, device)
