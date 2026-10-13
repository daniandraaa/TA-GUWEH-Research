"""Bind V7 to unchanged V6 split/scaler/QC and frozen comparator artifacts."""
from dataclasses import dataclass
from pathlib import Path
import importlib.metadata
import math
import json
import platform
import yaml
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from research_v4.common import disk_check
from research_v6.common import Study as BaseStudy, context as v6_context, verify_files, write_manifest, require_prepared, require_quality

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = '20261004_fit_distribution_gan_v7'
PREPARATION_FILES = ('fit_rows.csv', 'reference_rows.csv', 'fit_features.csv', 'reference_features.csv',
    'feature_families.json', 'amplitude_scale.json')


@dataclass
class Study(BaseStudy):
    baseline: Path


def baseline_identity(old):
    for fold in old.protocol['folds']: require_prepared(old, fold)
    directory = old.folder(1)
    inputs = {'protocol.json': sha256_file(old.run / 'protocol.json'),
        'folds.json': sha256_file(old.run / 'folds.json')}
    for fold in old.protocol['folds']:
        for name in ('preparation.json', *PREPARATION_FILES):
            relative = f'fold_{fold}/{name}'; inputs[relative] = sha256_file(old.run / relative)
    for marker in ('completed.json', 'generated.json', 'quality.json'):
        saved = read_json(directory / marker); verify_files(directory, saved)
        if saved['experiment_hash'] != old.signature: raise RuntimeError('Wrong baseline identity')
        inputs[f'fold_1/{marker}'] = sha256_file(directory / marker)
        inputs.update({f'fold_1/{name}': digest for name, digest in saved['files'].items()})
    quality = read_json(directory / 'quality.json')
    if quality['generated_manifest_sha256'] != sha256_file(directory / 'generated.json'):
        raise RuntimeError('Wrong baseline quality source')
    if read_json(directory / 'generated.json')['generator_sha256'] != sha256_file(directory / 'generator.pth'):
        raise RuntimeError('Wrong baseline generator source')
    return inputs


def validate_protocol(p, baseline):
    inherited = {k: v for k, v in p.items() if k not in ('source_v6_run', 'fidelity')}
    expected = {k: v for k, v in baseline.items() if k != 'source_v5_run'}
    expected['training'] = {k: v for k, v in expected['training'].items() if k != 'auxiliary_moment_weight'}
    if inherited != expected:
        raise ValueError('V7 changes only objective; all V6 split/architecture/training/QC settings must stay identical')
    f = p['fidelity']
    if (f['backend'] != 'cpu_with_autograd_to_generator' or f['spectral_window_samples'] != [500, 1000, 2000]
            or f['quantiles'] != [.1, .25, .5, .75, .9]
            or f['target_source'] != 'gan_fit_patients_only_patient_balanced'
            or f['power_floor'] != 1e-6 or f['target_updates_during_training']
            or set(f['weights']) != {'amplitude', 'roughness', 'correlation', 'spectrum'}
            or set(f['iqr_floor']) != set(f['weights'])
            or any(not math.isfinite(v) or v <= 0 for v in [*f['weights'].values(), *f['iqr_floor'].values()])):
        raise ValueError('Unsupported fixed V7 distribution objective')


def context(run_id=DEFAULT_RUN, initialize=False):
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}: raise ValueError('One run-id directory required')
    p = yaml.safe_load((PACKAGE / 'protocol.yaml').read_text())
    old = v6_context(p['source_v6_run'])
    validate_protocol(p, old.protocol)
    inputs = baseline_identity(old)
    identity = {'protocol': p, 'source_v6_hash': old.signature,
        'source_v6_files': inputs, 'development_cache': old.identity['development_cache'],
        'inner_folds': old.folds, 'records_hash': canonical_hash(old.records),
        'code': {f.name: sha256_file(f) for f in sorted(PACKAGE.glob('*.py'))},
        'runner_sha256': sha256_file(ROOT / 'notebooks/_v7_runner.py'),
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
            **{n: importlib.metadata.version(n) for n in ('torch','numpy','pandas','scipy','scikit-learn','PyYAML','matplotlib')}},
        'test_signals_read': False, 'outer_validation_signals_read': False,
        'development_recipe_informed_by_v6_quality_failure': True}
    study = Study(ROOT / 'results/research_v7' / run_id, old.source, p, old.records, old.folds, identity, old.run)
    path = study.run / 'protocol.json'
    expected = json.loads(json.dumps({**identity, 'experiment_hash': study.signature}))
    if not path.exists():
        if not initialize: raise RuntimeError('Jalankan notebook37 / prepare dahulu')
        disk_check(study.run, p)
        with lock(study.run / '.initialize.lock'):
            if not path.exists():
                write_json(study.run / 'folds.json', old.folds); write_json(path, expected)
    if (read_json(path) != expected
            or canonical_hash(read_json(study.run / 'folds.json')) != canonical_hash(expected['inner_folds'])):
        raise RuntimeError('Kode/protokol/input/runtime V7 berubah. Pertahankan run lama; gunakan run-id baru.')
    return study
