"""Frozen experiment identity, readiness gates, and shared-cache provenance."""
from pathlib import Path
import itertools
import shutil
import importlib.metadata
import yaml
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from research_v3.common import context as v3_context

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = '20261002_patient_mil_hybrids_v4'


def scenarios(cfg):
    return [{'id': f'S{i:02}', 'learning_rate': float(lr), 'batch_size': int(bs)}
            for i, (lr, bs) in enumerate(itertools.product(cfg['learning_rates'], cfg['batch_sizes']), 1)]


def disk_check(run, cfg, peak_bytes=256 * 1024**2):
    free = shutil.disk_usage(run if run.exists() else ROOT).free
    needed = cfg['minimum_free_gib'] * 1024**3 + peak_bytes
    if free < needed:
        raise RuntimeError(f'Disk bebas {free / 1024**3:.2f} GiB; perlu {needed / 1024**3:.2f} GiB. '
                           'Pindahkan arsip yang sudah diverifikasi secara manual, lalu lanjutkan sel yang sama.')


def context(run_id=DEFAULT_RUN, initialize=False):
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run-id must be one directory name')
    cfg = yaml.safe_load((PACKAGE / 'protocol.yaml').read_text())
    old, source, _, config, _, _ = v3_context(cfg['source_v3_run'])
    records, folds = read_json(old / 'data/patients.json'), read_json(old / 'folds/manifest.json')
    if config['sampling_rate'] != cfg['sampling_rate'] or config['segment_length'] != cfg['segment_length']:
        raise ValueError('Waveform specification changed')
    if any(bs % cfg['segments_per_patient'] for bs in cfg['batch_sizes']):
        raise ValueError('Segment budget must be divisible by segments_per_patient')
    identity = {'protocol': cfg, 'inherited_config': config,
        'source_v3_signature': read_json(old / 'protocol.json')['experiment_hash'],
        'source_v3_selection_hash': canonical_hash(read_json(old / 'selection.json')),
        'source_v2_signature': read_json(source / 'protocol.json')['experiment_hash'],
        'development_cache': {name: sha256_file(source / 'data/development' / name)
                              for name in ('X.npy', 'segments.csv', 'manifest.json')},
        'patient_metadata_hash': canonical_hash(records), 'fold_manifest_hash': canonical_hash(folds),
        'code': {p.name: sha256_file(p) for p in sorted(PACKAGE.glob('*.py'))},
        'runtime': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scikit-learn')}}
    signature = canonical_hash(identity)
    run = ROOT / 'results/research_v4' / run_id
    path = run / 'protocol.json'
    if not path.exists():
        if not initialize:
            raise RuntimeError('Jalankan notebook 20 / endpoint prepare dahulu.')
        disk_check(run, cfg)
        with lock(run / '.init.lock'):
            if not path.exists():
                write_json(run / 'data/patients.json', records)
                write_json(run / 'folds/manifest.json', folds)
                write_json(path, {**identity, 'experiment_hash': signature,
                                  'test_role': 'historical_exploratory_not_external'})
    if read_json(path)['experiment_hash'] != signature:
        raise RuntimeError('Kode/protokol/runtime/input V4 berubah. Gunakan run-id baru; hasil tidak boleh dicampur.')
    if canonical_hash(read_json(run / 'data/patients.json')) != canonical_hash(records) or canonical_hash(read_json(run / 'folds/manifest.json')) != canonical_hash(folds):
        raise RuntimeError('Metadata/fold V4 berubah')
    return run, source, old, cfg, config


def cv_directory(run, model, pooling, scenario, fold, seed):
    return run / 'cv' / model / pooling / scenario / f'fold_{fold}' / f'seed_{seed}'


def progress(run, cfg, folds=5):
    expected = [cv_directory(run, m, p, s['id'], f, cfg['cv_seed']) / 'metrics.json'
                for m in cfg['models'] for p in cfg['pooling'] for s in scenarios(cfg)
                for f in range(1, folds + 1)]
    missing = [str(p.relative_to(run)) for p in expected if not p.is_file()]
    return {'complete': len(expected) - len(missing), 'expected': len(expected),
            'ready': not missing, 'missing': missing}


def require_final_complete(run, cfg):
    if not (run / 'selection.json').is_file() or not progress(run, cfg)['ready']:
        raise RuntimeError('Historical test terkunci: full grid dan seleksi belum lengkap.')
    for model in cfg['models']:
        for seed in cfg['final_seeds']:
            folder = run / 'final' / model / f'seed_{seed}'
            if not (folder / 'metrics.json').is_file() or not (folder / 'model.pth').is_file():
                raise RuntimeError(f'Historical test terkunci: final {model}/seed_{seed} belum lengkap.')
    missing = [f'{m}/fold_{f}/seed_{s}' for m in cfg['models'] for f in range(1, 6)
               for s in cfg['stability_seeds'] if not (run / 'stability' / m / f'fold_{f}' / f'seed_{s}' / 'metrics.json').is_file()]
    if missing:
        raise RuntimeError('Historical test terkunci: stability belum lengkap: ' + ', '.join(missing[:5]))
