"""Separate frozen protocol, verified V4 controls, and development-only gates."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import yaml
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from research_v4.common import context as v4_context, cv_directory, disk_check
from research_v4_mps_compat.runtime import audit_run

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = '20261003_window_bag_ablation_v5'
RESULT_FILES = ('config.json', 'metrics.json', 'history.csv',
                'validation_predictions.csv', 'validation_segment_weights.csv')


@dataclass
class Study:
    run: Path
    baseline: Path
    source: Path
    protocol: dict
    training_cfg: dict
    inherited: dict
    choices: dict
    records: list
    folds: list
    controls: list
    identity: dict

    @property
    def signature(self):
        return canonical_hash(self.identity)


def validate_folds(records, folds):
    ids = {r['hospital_id'] for r in records}
    if len(ids) != 230 or len(records) != 230 or any(r['pool'] != 'development' for r in records):
        raise ValueError('V5 accepts exactly the unchanged 230 development patients')
    if [f['fold'] for f in folds] != [1, 2, 3, 4, 5]:
        raise ValueError('Five fixed folds required')
    validation = []
    for f in folds:
        train, valid = set(f['train_ids']), set(f['validation_ids'])
        if train & valid or train | valid != ids:
            raise ValueError('Fold overlap or cohort mismatch')
        if len(train) != len(f['train_ids']) or len(valid) != len(f['validation_ids']):
            raise ValueError('Duplicate fold patient')
        validation.extend(f['validation_ids'])
    if sorted(validation) != sorted(ids):
        raise ValueError('OOF must cover each development patient once')


def validate_control(case, baseline_hash, base_cfg, choice, fold, seed):
    unsigned = {k: v for k, v in case.items() if k != 'case_hash'}
    scenario = {'id': choice['scenario'], 'learning_rate': choice['learning_rate'],
                'batch_size': choice['batch_size']}
    expected = {'experiment_hash': baseline_hash, 'model': choice['model'],
        'pooling': choice['pooling'], 'scenario': scenario, 'fold': fold['fold'], 'seed': seed,
        'effective_seed': seed + 100 * fold['fold'], 'training_mode': 'bag',
        'train_patient_ids': sorted(fold['train_ids']),
        'validation_patient_ids': sorted(fold['validation_ids']),
        'settings': {**base_cfg['training'], **scenario, 'scheduler_t_max': 100},
        'segment_budget': 32, 'patient_batch_size': 8, 'diagnostic_v3_choice': None,
        'phase': 'cv' if seed == base_cfg['cv_seed'] else 'stability'}
    if case.get('case_hash') != canonical_hash(unsigned) or any(case.get(k) != v for k, v in expected.items()):
        raise RuntimeError('V4 control recipe/fold/seed changed; cannot reuse it')
    if base_cfg['segments_per_patient'] != 4 or case['device'] not in {'mps', 'cpu'}:
        raise RuntimeError('Unsupported control sampling/device')


def control_records(baseline, base_cfg, choices, folds, seeds):
    signature = read_json(baseline / 'protocol.json')['experiment_hash']
    records = []
    for model, choice in choices.items():
        for fold in folds:
            for seed in seeds:
                stability = baseline / 'stability' / model / f'fold_{fold["fold"]}' / f'seed_{seed}'
                if seed == base_cfg['cv_seed']:
                    origin = cv_directory(baseline, model, choice['pooling'], choice['scenario'], fold['fold'], seed)
                    reuse = read_json(stability / 'config.json')
                    if (reuse != {'reused_cv_case': str(origin.relative_to(baseline)),
                                  'source_config_sha256': sha256_file(origin / 'config.json'),
                                  'source_metrics_sha256': sha256_file(origin / 'metrics.json')}
                            or read_json(stability / 'metrics.json') != read_json(origin / 'metrics.json')):
                        raise RuntimeError('V4 seed42 reuse provenance changed')
                else:
                    origin = stability
                case = read_json(origin / 'config.json')
                validate_control(case, signature, base_cfg, choice, fold, seed)
                files = {str((origin / n).relative_to(ROOT)): sha256_file(origin / n) for n in RESULT_FILES}
                records.append({'model': model, 'fold': fold['fold'], 'seed': seed,
                    'device': case['device'], 'origin': str(origin.relative_to(ROOT)), 'sha256': files,
                    'stability_reference': str(stability.relative_to(ROOT)),
                    'stability_config_sha256': sha256_file(stability / 'config.json'),
                    'stability_metrics_sha256': sha256_file(stability / 'metrics.json')})
    if len({r['device'] for r in records}) != 1:
        raise RuntimeError('Controls mix devices; a paired device is required')
    return records


def context(run_id=DEFAULT_RUN, initialize=False):
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run-id must be one directory name')
    protocol = yaml.safe_load((PACKAGE / 'protocol.yaml').read_text())
    if (protocol['models'] != ['hierarchical', 'encoder'] or protocol['folds'] != [1, 2, 3, 4, 5]
            or protocol['seeds'] != [42, 43, 44] or protocol['patient_batch_size'] != 8
            or protocol['control_segments_per_patient'] != 4 or protocol['candidate_segments_per_patient'] != 8
            or protocol['historical_test_enabled'] or protocol['final_training_enabled']):
        raise ValueError('This endpoint implements only the preregistered development K4/K8 ablation')
    baseline, source, _, base_cfg, inherited = v4_context(protocol['source_v4_run'])
    compatibility = audit_run(baseline)
    selections = read_json(baseline / 'selection.json')
    if selections['test_used_for_selection'] or selections['selection_source'] != 'development five-fold CV only':
        raise RuntimeError('V4 choices must come from development CV')
    choices = {m: next(c for c in selections['choices'] if c['model'] == m) for m in protocol['models']}
    if any(c['batch_size'] != 32 or c['patient_batch_size'] != 8 for c in choices.values()):
        raise RuntimeError('Selected controls must use batch eight patients / budget32')
    records = [r for r in read_json(baseline / 'data/patients.json') if r['pool'] == 'development']
    folds = read_json(baseline / 'folds/manifest.json')
    validate_folds(records, folds)
    audit = baseline / 'development_input_audit_20261003'
    status = read_json(audit / 'status.json')
    if not status['complete'] or status['completed_patients'] != 230 or status['test_signals_read']:
        raise RuntimeError('Complete notebook28 development audit first')
    controls = control_records(baseline, base_cfg, choices, folds, protocol['seeds'])
    training_cfg = deepcopy(base_cfg)
    training_cfg.update(models=protocol['models'], segments_per_patient=8, batch_sizes=[64],
                        minimum_free_gib=protocol['minimum_free_gib'])
    baseline_protocol = read_json(baseline / 'protocol.json')
    if inherited['sampling_rate'] != 2000 or inherited['segment_length'] != 5000:
        raise ValueError('Native window specification changed')
    scalers = {str(p.relative_to(baseline)): sha256_file(p)
               for f in folds for p in [baseline / f'scalers/fold_{f["fold"]}/scaler.joblib',
                                       baseline / f'scalers/fold_{f["fold"]}/provenance.json']}
    identity = {'protocol': protocol, 'v4_experiment_hash': baseline_protocol['experiment_hash'],
        'v4_compatibility_hash': compatibility['compatibility_hash'],
        'v4_selection_sha256': sha256_file(baseline / 'selection.json'),
        'choices': choices, 'training_cfg': training_cfg, 'inherited_config': inherited,
        'development_cache': baseline_protocol['development_cache'],
        'development_patient_hash': canonical_hash(records), 'fold_hash': canonical_hash(folds),
        'input_audit_hash': read_json(audit / 'manifest.json')['audit_hash'],
        'quality_table_sha256': sha256_file(audit / 'patient_signal_quality.csv'),
        'controls_hash': canonical_hash(controls), 'fold_scaler_sha256': scalers,
        'paired_device': controls[0]['device'],
        'code': {p.name: sha256_file(p) for p in sorted(PACKAGE.glob('*.py'))},
        'runner_sha256': sha256_file(ROOT / 'notebooks/_v5_runner.py'),
        'selection_uses_development_only': True, 'test_signals_read': False}
    run = ROOT / 'results/research_v5' / run_id
    study = Study(run, baseline, source, protocol, training_cfg, inherited, choices, records, folds, controls, identity)
    path = run / 'protocol.json'
    if not path.exists():
        if not initialize:
            raise RuntimeError('Jalankan notebook29 / endpoint prepare terlebih dahulu')
        disk_check(run, training_cfg)
        with lock(run / '.init.lock'):
            if not path.exists():
                write_json(run / 'data/patients.json', records)
                write_json(run / 'folds/manifest.json', folds)
                write_json(run / 'control_manifest.json', controls)
                write_json(path, {**identity, 'experiment_hash': study.signature})
    saved = read_json(path)
    if saved != {**identity, 'experiment_hash': study.signature}:
        raise RuntimeError('Kode/protokol/runtime/input/control V5 berubah; pertahankan run lama dan gunakan run-id baru')
    for relative, value in [('data/patients.json', records), ('folds/manifest.json', folds),
                            ('control_manifest.json', controls)]:
        if read_json(run / relative) != value:
            raise RuntimeError(f'Frozen V5 input changed: {relative}')
    return study


def case_directory(study, model, fold, seed):
    return study.run / 'ablation' / model / f'fold_{fold}' / f'seed_{seed}'


def scenario(study, model):
    return {'id': study.protocol['candidate_scenario_id'],
            'learning_rate': study.choices[model]['learning_rate'], 'batch_size': 64}


def case_identity(study, model, fold, seed, device):
    spec = scenario(study, model)
    manifest = study.folds[fold - 1]
    return {'experiment_hash': study.signature, 'device': str(device), 'phase': 'ablation',
        'model': model, 'pooling': study.choices[model]['pooling'], 'scenario': spec,
        'fold': fold, 'seed': seed, 'effective_seed': seed + 100 * fold,
        'settings': {**study.training_cfg['training'], **spec, 'scheduler_t_max': 100},
        'training_mode': 'bag', 'train_patient_ids': sorted(manifest['train_ids']),
        'validation_patient_ids': sorted(manifest['validation_ids']), 'diagnostic_v3_choice': None,
        'segment_budget': 64, 'patient_batch_size': 8}


def check_case(study, model, fold, seed):
    directory = case_directory(study, model, fold, seed)
    saved = read_json(directory / 'config.json')
    expected = case_identity(study, model, fold, seed, study.identity['paired_device'])
    if saved != {**expected, 'case_hash': canonical_hash(expected)}:
        raise RuntimeError(f'Candidate identity changed: {directory}')
    for name in RESULT_FILES:
        if not (directory / name).is_file():
            raise RuntimeError(f'Completed case missing artifact: {directory / name}')
    return directory


def require_prepared(study):
    preparation = read_json(study.run / 'preparation.json')
    if preparation['experiment_hash'] != study.signature or preparation['scaler_fit_performed']:
        raise RuntimeError('V5 preparation mismatch')
    for relative, digest in study.identity['fold_scaler_sha256'].items():
        if sha256_file(study.run / relative) != digest:
            raise RuntimeError('Copied fold scaler changed')


def require_smoke(study, model, device):
    if str(device) != study.identity['paired_device']:
        raise RuntimeError(f'Paired controls use {study.identity["paired_device"]}; device must stay identical')
    saved = read_json(study.run / 'smoke' / f'{model}.json')
    record = saved['record']
    if (saved['record_hash'] != canonical_hash(record) or record['experiment_hash'] != study.signature
            or record['device'] != str(device) or record['model'] != model
            or record['batch_shape'] != [8, 8, 12, 5000] or record['valid_segments'] != 64
            or record['disposable_optimizer_steps'] != 1 or not record['passed']):
        raise RuntimeError('Jalankan smoke notebook29 pada model/device yang sama')


def progress(study):
    complete, missing = [], []
    for model in study.protocol['models']:
        for fold in study.protocol['folds']:
            for seed in study.protocol['seeds']:
                path = case_directory(study, model, fold, seed)
                if (path / 'metrics.json').exists():
                    check_case(study, model, fold, seed)
                    complete.append((model, fold, seed))
                else:
                    missing.append(f'{model}/fold_{fold}/seed_{seed}')
    return {'complete': len(complete), 'expected': 30, 'ready_for_comparison': not missing,
        'per_model': {m: sum(x[0] == m for x in complete) for m in study.protocol['models']},
        'controls_reused': len(study.controls), 'missing': missing,
        'test_enabled': False, 'final_training_enabled': False}
