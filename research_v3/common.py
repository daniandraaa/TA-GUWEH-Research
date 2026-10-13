"""Versioned identity, inherited patient splits, and conservative disk checks."""
from pathlib import Path
import shutil
import yaml
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from research_v2.common import context as v2_context

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = '20261001_patient_r_candidates_v3'


def context(run_id=DEFAULT_RUN, initialize=False):
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run-id must be a single directory name')
    cfg = yaml.safe_load((PACKAGE / 'protocol.yaml').read_text())
    source, _, v2cfg, config, scenarios = v2_context(cfg['source_run'])
    records = read_json(source / 'data/patients.json')
    folds = read_json(source / 'folds/manifest.json')
    development = {r['hospital_id'] for r in records if r['pool'] == 'development'}
    if len(folds) != config['n_folds'] or sorted(h for f in folds for h in f['validation_ids']) != sorted(development):
        raise ValueError('Invalid inherited fold coverage')
    for f in folds:
        a,b = set(f['train_ids']),set(f['validation_ids'])
        if a & b or a | b != development:
            raise ValueError('Invalid inherited patient split')
    identity = {'protocol': cfg, 'config': config, 'scenarios': scenarios,
                'source_hash': read_json(source / 'protocol.json')['experiment_hash'],
                'patient_metadata_hash': canonical_hash(records), 'fold_manifest_hash': canonical_hash(folds),
                'source_development_cache': {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                     for p in sorted((source / 'data/development').glob('*')) if p.is_file() and not p.name.startswith('.')},
                'code': {p.name: sha256_file(p) for p in sorted(PACKAGE.glob('*.py'))}}
    signature = canonical_hash(identity)
    run = ROOT / 'results/research_v3' / run_id
    path = run / 'protocol.json'
    if not path.exists():
        if not initialize:
            raise RuntimeError('Jalankan prepare dari notebook 11 dahulu.')
        disk_check(run, cfg)
        with lock(run / '.init.lock'):
            if not path.exists():
                write_json(path, {**identity, 'experiment_hash': signature,
                                 'test_role': 'historical_exploratory_not_external',
                                 'expert_ecg_review_available': False,
                                 'candidate_pvc_verified': False})
                write_json(run / 'data/patients.json', read_json(source / 'data/patients.json'))
                write_json(run / 'folds/manifest.json', read_json(source / 'folds/manifest.json'))
    if read_json(path)['experiment_hash'] != signature:
        raise RuntimeError('Kode/protokol V3 berubah; buat run-id baru, jangan campur hasil.')
    if canonical_hash(read_json(run / 'data/patients.json')) != canonical_hash(records) or canonical_hash(read_json(run / 'folds/manifest.json')) != canonical_hash(folds):
        raise RuntimeError('Metadata/fold V3 berubah; cohort tidak boleh berubah diam-diam.')
    # V2 validates source-code, configuration and raw development input identity.
    # Reuse its preprocessing specification without changing its results.
    return run, source, cfg, config, scenarios, v2cfg


def disk_check(run, cfg, extra_bytes=0):
    free = shutil.disk_usage(run if run.exists() else ROOT).free
    needed = cfg['minimum_free_gib'] * 1024**3 + extra_bytes
    if free < needed:
        raise RuntimeError(f'Disk bebas {free/1024**3:.2f} GiB; perlu {needed/1024**3:.2f} GiB.')


def require_final_complete(run, cfg):
    if not (run / 'selection.json').exists():
        raise RuntimeError('Test terkunci: seleksi seluruh CV belum lengkap.')
    names = cfg['models'] + ['extratrees']
    missing = [f'{m}/seed_{s}' for m in names for s in cfg['final_seeds']
               if not (run / 'final' / m / f'seed_{s}/metrics.json').exists()]
    if missing:
        raise RuntimeError('Test terkunci; selesaikan final: ' + ', '.join(missing))
