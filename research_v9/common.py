"""Bind a new epoch rule to immutable V8 selectors and real-only controls."""
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import json
import shutil
import yaml
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from research_v4.common import disk_check
from research_v6.common import verify_files, write_manifest
from research_v8.common import context as v8_context, directory as v8_directory
from research_v8.common import checked_complete as v8_complete, require_prepared as v8_prepared

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = '20261006_inner_epoch_median_real_only_v9'


def median_epoch(values):
    if len(values) != 3 or any(type(v) is not int or not 1 <= v <= 100 for v in values):
        raise ValueError('Exactly three integer inner-selected epochs in [1,100] required')
    return int(median(values))


@dataclass
class Study:
    run: Path
    parent: object
    protocol: dict
    plan: list
    identity: dict

    @property
    def signature(self): return canonical_hash(self.identity)
    @property
    def source(self): return self.parent.source
    @property
    def records(self): return self.parent.records
    @property
    def folds(self): return self.parent.folds
    @property
    def choices(self): return self.parent.choices
    @property
    def inherited(self): return self.parent.inherited
    @property
    def training(self): return self.parent.training
    @property
    def gan(self): return self.parent.gan


def context(run_id=DEFAULT_RUN, initialize=False):
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('One run-id directory name required')
    p = yaml.safe_load((PACKAGE / 'protocol.yaml').read_text())
    expected = {
        'source_v8_run': '20261005_paired_classifier_augmentation_v8',
        'models': ['hierarchical', 'encoder'], 'folds': [1, 2, 3, 4, 5], 'seeds': [42, 43, 44],
        'arm': 'real_only', 'epoch_rule': 'median_of_three_existing_inner_selected_epochs_per_model_fold',
        'inner_split_rule': 'unchanged_v8_gan_fit_and_reference_patients',
        'outer_checkpoint': 'fixed_last_epoch_no_outer_selection',
        'hyperparameters': 'unchanged_v8_lr_pooling_optimizer_scheduler_scaler_k4',
        'patient_batch_size': 8, 'segments_per_patient': 4, 'sampling_rate': 2000,
        'segment_length': 5000, 'evaluation_segment_chunk': 16,
        'reuse_rule': 'exact_matching_v8_real_only_recipe_device_and_epoch_by_reference',
        'final_training_enabled': False, 'test_signals_enabled': False,
        'interpretation': 'exploratory_development_ablation_prior_hyperparameter_selection_not_fully_nested'}
    if set(p) != set(expected) | {'minimum_free_gib'} or any(p[k] != v for k, v in expected.items()) or p['minimum_free_gib'] < 3:
        raise ValueError('Only the frozen real-only epoch ablation is supported')
    old = v8_context(p['source_v8_run']); v8_prepared(old)
    audit_path = old.run / 'analysis_20261006_complete/audit.json'
    audit = read_json(audit_path)
    if (audit['experiment_hash'] != old.signature or audit['inner_cases_verified'] != 30
            or audit['outer_cases_verified'] != 90 or not audit['v4_common_training_trajectories_identical']
            or audit['audit_code_sha256'] != sha256_file(ROOT / 'analysis_v8/audit_completed.py')
            or audit['test_signals_read']):
        raise RuntimeError('Complete the unchanged V8 audit first')
    plan = []; sources = {}
    for model in p['models']:
        for fold in p['folds']:
            epochs = []
            for seed in p['seeds']:
                result = v8_complete(old, model, fold, seed)
                epochs.append(result['selected_epoch'])
                for arm in ('inner_selection', 'real_only'):
                    v8_complete(old, model, fold, seed, arm)
                    folder = v8_directory(old, model, fold, seed, arm)
                    marker = folder / 'completed.json'
                    name = str(marker.relative_to(ROOT)); digest = sha256_file(marker)
                    if audit['source_sha256'].get(name) != digest:
                        raise RuntimeError('V8 completion manifest differs from the completed audit')
                    sources[name] = digest
            budget = median_epoch(epochs)
            for seed, epoch in zip(p['seeds'], epochs):
                folder = v8_directory(old, model, fold, seed, 'real_only')
                plan.append({'model': model, 'fold': fold, 'seed': seed,
                    'v8_selected_epoch': epoch, 'epochs': budget, 'reuse': epoch == budget,
                    'source_device': read_json(folder / 'case.json')['device']})
    identity = {'protocol': p, 'source_v8_hash': old.signature, 'source_completed_manifests': sources,
        'source_audit_sha256': sha256_file(audit_path), 'new_model': old.identity['new_model'],
        'runtime': old.identity['runtime'], 'plan': plan,
        'code': {q.name: sha256_file(q) for q in sorted(PACKAGE.glob('*.py'))},
        'runner_sha256': sha256_file(ROOT / 'notebooks/_v9_runner.py'),
        'test_signals_read': False, 'fully_nested_estimate': False}
    study = Study(ROOT / 'results/research_v9' / run_id, old, p, plan, identity)
    expected_saved = json.loads(json.dumps({**identity, 'experiment_hash': study.signature}))
    path = study.run / 'protocol.json'
    if not path.exists():
        if not initialize: raise RuntimeError('Jalankan notebook 46 / prepare dahulu')
        disk_check(study.run, p)
        with lock(study.run / '.initialize.lock'):
            if not path.exists(): write_json(path, expected_saved)
    if read_json(path) != expected_saved:
        raise RuntimeError('Kode/protokol/data/runtime V9 berubah. Pertahankan hasil; gunakan run-id baru.')
    return study


def entry_for(study, model, fold, seed):
    values = [r for r in study.plan if (r['model'], r['fold'], r['seed']) == (model, fold, seed)]
    if len(values) != 1: raise ValueError('Unsupported model/fold/seed')
    return values[0]


def directory(study, model, fold, seed):
    return study.run / 'cases' / model / f'fold_{fold}' / f'seed_{seed}'


def settings_for(study, row):
    return {**study.training, 'learning_rate': study.choices[row['model']]['learning_rate'], 'epochs': row['epochs']}


def case_for(study, row, device):
    m, f, seed = row['model'], row['fold'], row['seed']; split = study.folds[f - 1]
    selectors = {str(s): sha256_file(v8_directory(study.parent, m, f, s) / 'completed.json') for s in study.protocol['seeds']}
    case = {'experiment_hash': study.signature, 'model': m, 'fold': f, 'seed': seed,
        'arm': 'real_only', 'device': str(device), 'effective_seed': seed + 100 * f,
        'settings': settings_for(study, row), 'pooling': study.choices[m]['pooling'],
        'train_ids': sorted(split['gan_fit_ids'] + split['gan_reference_ids']),
        'evaluation_ids': sorted(split['outer_validation_ids']),
        'epoch_rule': study.protocol['epoch_rule'], 'inner_selection_manifests_sha256': selectors,
        'scaler_sha256': sha256_file(study.run / f'scalers/outer_{f}/scaler.joblib'),
        'source_gan_generated_sha256': sha256_file(study.gan.folder(f) / 'generated.json'),
        'test_signals_read': False, 'outer_epoch_selection': False}
    return {**case, 'case_hash': canonical_hash(case)}


def validate_reuse(study, row, case):
    if not row['reuse'] or row['epochs'] != row['v8_selected_epoch']:
        raise RuntimeError('A different selected epoch requires a new fit')
    old_folder = v8_directory(study.parent, row['model'], row['fold'], row['seed'], 'real_only')
    old = read_json(old_folder / 'case.json')
    keys = ('model', 'fold', 'seed', 'arm', 'device', 'effective_seed', 'settings', 'pooling',
            'train_ids', 'evaluation_ids', 'scaler_sha256', 'source_gan_generated_sha256',
            'test_signals_read', 'outer_epoch_selection')
    if any(old[k] != case[k] for k in keys): raise RuntimeError('V8 reuse recipe/device is not identical')
    result = v8_complete(study.parent, row['model'], row['fold'], row['seed'], 'real_only')
    if result['selected_epoch'] != row['epochs'] or result['epochs_run'] != row['epochs'] or result['inner_selection']:
        raise RuntimeError('V8 reuse must be the exact fixed last-epoch checkpoint')
    return old_folder


def require_prepared(study):
    saved = read_json(study.run / 'preparation.json'); verify_files(study.run, saved)
    if saved['experiment_hash'] != study.signature: raise RuntimeError('Preparation changed')
    return saved


def checked_complete(study, model, fold, seed):
    row = entry_for(study, model, fold, seed); folder = directory(study, model, fold, seed)
    if not (folder / 'completed.json').exists(): raise RuntimeError(f'Lengkapi {model}/fold{fold}/seed{seed}')
    saved = read_json(folder / 'completed.json'); verify_files(folder, saved)
    case = read_json(folder / 'case.json')
    if (saved['experiment_hash'] != study.signature or saved['case_hash'] != case['case_hash']
            or case != case_for(study, row, case['device'])): raise RuntimeError('Case identity changed')
    if row['reuse']:
        if saved['kind'] != 'v8_reference': raise RuntimeError('Expected immutable V8 reference')
        old = validate_reuse(study, row, case); ref = read_json(folder / 'reuse.json')
        if ref != {'source_folder': str(old.relative_to(ROOT)), 'completed_sha256': sha256_file(old / 'completed.json')}:
            raise RuntimeError('V8 reference changed')
        return old, read_json(old / 'metrics.json')
    if saved['kind'] != 'new_fixed_epoch_fit': raise RuntimeError('Expected a new fit')
    result = read_json(folder / 'metrics.json')
    if result['epochs_run'] != row['epochs'] or result['selected_epoch'] != row['epochs'] or result['inner_selection']:
        raise RuntimeError('Fixed median epoch budget changed')
    return folder, result


def progress(study):
    completed = reused = new = 0; missing = []
    for row in study.plan:
        m, f, s = row['model'], row['fold'], row['seed']
        if (directory(study, m, f, s) / 'completed.json').exists():
            checked_complete(study, m, f, s); completed += 1
            if row['reuse']: reused += 1
            else: new += 1
        else: missing.append(f'{m}/fold{f}/seed{s}')
    return {'experiment_hash': study.signature, 'comparison_complete': completed, 'comparison_expected': len(study.plan),
        'referenced_v8_complete': reused, 'referenced_v8_expected': sum(r['reuse'] for r in study.plan),
        'new_fits_complete': new, 'new_fits_expected': sum(not r['reuse'] for r in study.plan),
        'ready_for_comparison': not missing, 'missing': missing, 'final_enabled': False, 'test_enabled': False}


def remaining_storage(study):
    estimate = 0
    for row in study.plan:
        folder = directory(study, row['model'], row['fold'], row['seed'])
        if row['reuse'] or (folder / 'completed.json').exists(): continue
        old = v8_directory(study.parent, row['model'], row['fold'], row['seed'], 'real_only')
        total = sum(q.stat().st_size for q in old.iterdir() if q.is_file())
        present = sum(q.stat().st_size for q in folder.iterdir() if q.is_file()) if folder.exists() else 0
        estimate += max(0, total - present)
    return int(estimate * 1.1)


def storage_check(study):
    estimate = remaining_storage(study)
    disk_check(study.run, study.protocol, peak_bytes=estimate + 256 * 1024**2)
    return {'free_gib': shutil.disk_usage(study.run).free / 1024**3,
        'estimated_remaining_artifacts_gib': estimate / 1024**3,
        'reserve_gib': study.protocol['minimum_free_gib'], 'peak_margin_gib': .25}
