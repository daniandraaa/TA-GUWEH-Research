"""Immutable development-only transfer experiment, separate from V2–V9."""
from dataclasses import dataclass
from pathlib import Path
import importlib.metadata
import json
import platform
import shutil
import yaml
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from research_v9.common import context as v9_context
from .assets import ASSETS
from .compatibility import require_inspection

PACKAGE = Path(__file__).parent
DEFAULT_RUN = '20261006_ecg_fm_transfer_pilot_v10'


def grid(protocol, arm):
    rates = protocol['tail_learning_rates' if arm == 'pretrained_tail' else 'head_learning_rates']
    rows = [{'scenario': f'S{i:02}', 'learning_rate': float(lr), 'batch_size': int(bs)}
        for i, (lr, bs) in enumerate(((lr, bs) for lr in rates for bs in protocol['patient_batch_sizes']), 1)]
    if len(rows) != 9 or len({(r['learning_rate'], r['batch_size']) for r in rows}) != 9:
        raise ValueError('Exactly nine distinct LR × patient batch configurations required')
    return rows


@dataclass
class Study:
    run: Path
    parent: object
    protocol: dict
    inspection: dict
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
    def classes(self): return self.parent.inherited['class_names']
    @property
    def cache(self): return ASSETS / 'experiments' / self.run.name


def storage_check(study, peak_bytes=0):
    free = shutil.disk_usage(ROOT).free
    needed = study.protocol['minimum_free_gib'] * 1024**3 + int(peak_bytes)
    if free < needed:
        raise RuntimeError(f'Ruang kosong {free/1024**3:.2f} GiB; perlu {needed/1024**3:.2f} GiB '
            '(cadangan 3 GiB + artefak tahap ini). Kosongkan ruang di luar hasil riset lalu ulang sel.')
    return {'free_gib': free / 1024**3, 'minimum_reserve_gib': study.protocol['minimum_free_gib'],
        'additional_peak_gib': peak_bytes / 1024**3}


def context(run_id=DEFAULT_RUN, initialize=False):
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}: raise ValueError('Single run-id required')
    p = yaml.safe_load((PACKAGE / 'protocol.yaml').read_text())
    if (p['probe_arms'] != ['random_probe', 'pretrained_probe'] or p['folds'] != [1,2,3,4,5]
        or p['outer_seeds'] != [42,43,44] or p['cv_seed'] != 42 or p['minimum_free_gib'] < 3
        or p['historical_test_enabled'] or p['final_training_enabled'] or p['cache_dtype'] != 'float32'
        or p['pretraining_patient_independence_verified']
        or p['native_samples'] != 5000 or p['sampling_rate'] != 2000
        or p['adapted_samples'] != 1250 or p['adapted_sampling_rate'] != 500
        or p['tail_scope'] != 'fold1_seed42_pilot_only'
        or p['outer_checkpoint'] != 'refit_fixed_inner_selected_epoch_no_outer_callbacks'):
        raise ValueError('Unsupported development-only transfer protocol')
    grid(p, 'pretrained_probe'); grid(p, 'pretrained_tail')
    inspection = require_inspection()
    parent = v9_context(p['source_v9_run'])
    audit_path = parent.run / 'analysis_20261006_complete/audit.json'; audit = read_json(audit_path)
    if (audit['experiment_hash'] != parent.signature or audit['comparison_cases_verified'] != 30
        or audit['test_signals_read'] or audit['audit_code_sha256'] != sha256_file(ROOT/'analysis_v9/audit_completed.py')):
        raise RuntimeError('Require verified completed V9 audit')
    # Bind every audited source manifest, without executing any old endpoint.
    for name, digest in audit['source_sha256'].items():
        if sha256_file(ROOT / name) != digest: raise RuntimeError(f'Frozen V9 audit source changed: {name}')
    ids = {r['hospital_id'] for r in parent.records}
    for split in parent.folds:
        fit, val, outer = map(set, (split['gan_fit_ids'], split['gan_reference_ids'], split['outer_validation_ids']))
        if fit & val or (fit | val) & outer or fit | val | outer != ids:
            raise RuntimeError('Patient overlap/incomplete development split')
    identity = {'protocol': p, 'source_v9_hash': parent.signature, 'v9_audit_sha256': sha256_file(audit_path),
        'records_hash': canonical_hash(parent.records), 'folds': parent.folds, 'classes': parent.inherited['class_names'],
        'inspection_hash': canonical_hash(inspection),
        'code': {str(q.relative_to(PACKAGE)): sha256_file(q) for q in sorted(PACKAGE.rglob('*.py'))},
        'runner_sha256': sha256_file(ROOT/'notebooks/_v10_runner.py'),
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
            **{n: importlib.metadata.version(n) for n in ('torch','numpy','pandas','scipy','scikit-learn','PyYAML','matplotlib')}},
        'test_signals_read': False, 'historical_test_previously_accessed': True,
        'pretraining_patient_independence_verified': False,
        'fully_nested_unbiased_confirmatory_estimate': False}
    study = Study(ROOT/'results/research_v10'/run_id, parent, p, inspection, identity)
    expected = json.loads(json.dumps({**identity, 'experiment_hash': study.signature}))
    path = study.run/'protocol.json'
    if not path.exists():
        if not initialize: raise RuntimeError('Jalankan notebook 51 / prepare dahulu')
        storage_check(study)
        with lock(study.run/'.initialize.lock'):
            if not path.exists(): write_json(path, expected)
    if read_json(path) != expected: raise RuntimeError('Kode/data/protokol/runtime V10 berubah; pertahankan hasil dan gunakan run-id baru')
    return study


def split_for(study, fold):
    if fold not in study.protocol['folds']: raise ValueError('Unsupported fold')
    return study.folds[fold-1]


def case_directory(study, arm, fold, seed, scenario=None):
    phase = f'inner_{scenario}' if scenario else f'outer_seed_{seed}'
    return study.run/'cases'/arm/f'fold_{fold}'/phase


def select_best(rows):
    if len(rows) != 9: raise RuntimeError('Selection requires all nine completed inner scenarios')
    return sorted(rows, key=lambda r: (-r['f1'], r['learning_rate'], r['batch_size']))[0]
