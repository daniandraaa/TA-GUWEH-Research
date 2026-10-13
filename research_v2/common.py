"""Independent run identity; legacy code, configs, results and inputs stay read-only."""
from pathlib import Path
import copy
import platform
import shutil
import torch
import yaml
from src.artifacts import ROOT, canonical_hash, open_run, read_json, sha256_file, write_json, lock

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = '20260930_anatomy_fs2000_v2'


def settings():
    return yaml.safe_load((PACKAGE / 'protocol.yaml').read_text())


def disk_check(run, cfg, extra_bytes=0):
    free = shutil.disk_usage(run if run.exists() else ROOT).free
    required = cfg['minimum_free_gib'] * 1024**3 + extra_bytes
    if free < required:
        raise RuntimeError(f'Disk bebas {free/1024**3:.2f} GiB, diperlukan {required/1024**3:.2f} GiB.')
    return free / 1024**3


def context(run_id=DEFAULT_RUN, initialize=False):
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run-id must be one directory name')
    cfg = settings()
    source, legacy, scenarios = open_run(cfg['source_run'])
    run = ROOT / 'results/research_v2' / run_id
    config = copy.deepcopy(legacy)
    for key in ['class_names', 'sampling_rate', 'segment_length', 'overlap']:
        config[key] = cfg[key]
    identity = {'extension': cfg, 'config': config, 'scenarios': scenarios,
                'source_experiment_hash': read_json(source / 'protocol.json')['experiment_hash'],
                'code': {p.name: sha256_file(p) for p in sorted(PACKAGE.glob('*.py'))}}
    signature = canonical_hash(identity)
    path = run / 'protocol.json'
    if not path.exists():
        if not initialize:
            raise RuntimeError('Jalankan prepare dari notebook 06 dahulu.')
        disk_check(run, cfg)
        with lock(run / '.init.lock'):
            if not path.exists():
                records = read_json(source / 'data/patients.json')
                assets = [ROOT / 'Diagnosis.xlsx', source / 'data/patients.json', source / 'folds/manifest.json']
                # Hold-out signals are not read until all final fits are complete.
                assets += [ROOT / cfg['preprocessing']['input'] / f"{r['hospital_id']}.csv"
                           for r in records if r['pool'] == 'development']
                hashes = {str(p.relative_to(ROOT)): {'sha256': sha256_file(p), 'bytes': p.stat().st_size,
                          'mtime_ns': p.stat().st_mtime_ns} for p in assets}
                write_json(path, {**identity, 'experiment_hash': signature, 'assets': hashes,
                                  'test_role': 'historical_not_external',
                                  'environment': {'python': platform.python_version(), 'torch': torch.__version__}})
    previous = read_json(path)
    if previous['experiment_hash'] != signature:
        raise RuntimeError('Kode/konfigurasi berubah; gunakan run ID baru.')
    for name, expected in previous['assets'].items():
        p = ROOT / name
        if not p.exists() or (p.stat().st_size, p.stat().st_mtime_ns) != (expected['bytes'], expected['mtime_ns']):
            raise RuntimeError(f'Input berubah: {name}')
    return run, source, cfg, config, scenarios


def verify_content(run):
    for name, expected in read_json(run / 'protocol.json')['assets'].items():
        if sha256_file(ROOT / name) != expected['sha256']:
            raise RuntimeError(f'Checksum input berubah: {name}')
