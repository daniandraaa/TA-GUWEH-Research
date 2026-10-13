"""New manifests and stage locks, without entering frozen parent contexts."""
from dataclasses import dataclass
from pathlib import Path
import importlib.metadata
import platform
import shutil
import sys
import numpy as np
import yaml
from src.artifacts import ROOT, canonical_hash, read_json, write_json, sha256_file

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = '20261007_beat_identity_v12'
LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']


def wfdb_module():
    # Import the frozen training runtime first, then the isolated QA package.
    import pandas, scipy
    folder = ROOT / '.ecgqa'
    if folder.is_dir() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    try:
        import wfdb
    except ImportError as error:
        raise RuntimeError('WFDB belum tersedia di .ecgqa; gunakan dependency QA notebook 17. .venv tidak diubah otomatis.') from error
    return wfdb


def valid_run(value):
    if not isinstance(value,str) or not value or Path(value).name!=value or value in {'.','..'} or '\\' in value:
        raise ValueError('Run-id harus satu nama folder')
    return value


@dataclass
class Study:
    run: Path
    protocol: dict
    identity: dict

    @property
    def signature(self):
        return canonical_hash(self.identity)


def context(run_id=DEFAULT_RUN, initialize=False):
    valid_run(run_id)
    p = yaml.safe_load((PACKAGE/'protocol.yaml').read_text())
    if any(p[name] for name in ['origin_training_enabled','historical_origin_test_enabled',
            'external_threshold_tuning_enabled','local_exclusion_allowed','reference_anchor_training_enabled']):
        raise RuntimeError('V12 hanya identitas beat; origin/test/exclusion/tuning eksternal dinonaktifkan')
    if p['folds']!=5 or p['classes']!=['normal_N','PVC_V_or_r','other_known_beat']:
        raise RuntimeError('Skema beat/fold berubah; review protokol baru dahulu')
    wfdb = wfdb_module()
    identity = {'protocol':p, 'code':{q.name:sha256_file(q) for q in sorted(PACKAGE.glob('*.py'))},
        'protocol_sha256':sha256_file(PACKAGE/'protocol.yaml'),
        'runner_sha256':sha256_file(ROOT/'notebooks/_v12_runner.py'),
        'shared_code':{name:sha256_file(ROOT/name) for name in ['src/artifacts.py','src/data.py','research_v3/candidates.py']},
        'runtime':{'python':platform.python_version(),'wfdb':wfdb.__version__,
            **{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','PyYAML','joblib','threadpoolctl','matplotlib']}},
        'local_records_sha256':sha256_file(ROOT/p['local_records']),
        'clinical_local_beat_validation':False,'origin_training':False}
    study = Study(ROOT/'results/research_v12'/run_id,p,identity)
    path = study.run/'protocol.json'
    expected = {**identity,'experiment_hash':study.signature}
    if not path.exists():
        if not initialize:
            raise RuntimeError('Mulai dari notebook 62 / prepare')
        disk_check(study)
        write_json(path,expected)
    if read_json(path)!=expected:
        raise RuntimeError('Source/runtime/protokol V12 berubah; pertahankan hasil, gunakan run-id baru')
    return study


def disk_check(study, additional=0):
    free = shutil.disk_usage(ROOT).free
    required = study.protocol['minimum_free_gib']*1024**3 + additional
    if free < required:
        raise RuntimeError(f'Ruang kosong {free/1024**3:.2f} GiB; perlu cadangan 3 GiB + {additional/1024**2:.0f} MiB. Hasil lama tidak dihapus.')


def complete(folder, identity, names, **extra):
    payload = {'identity':identity,'files':{name:sha256_file(folder/name) for name in names},**extra}
    write_json(folder/'completed.json',{**payload,'manifest_hash':canonical_hash(payload)})


def verified(folder, identity=None):
    path = folder/'completed.json'
    if not path.exists():
        return None
    m = read_json(path)
    payload = {k:v for k,v in m.items() if k!='manifest_hash'}
    if canonical_hash(payload)!=m['manifest_hash']:
        raise RuntimeError(f'Manifest berubah: {folder}')
    if identity is not None and m['identity']!=identity:
        raise RuntimeError(f'Identitas tahap berubah: {folder}')
    for name,digest in m['files'].items():
        if sha256_file(folder/name)!=digest:
            raise RuntimeError(f'Artefak berubah: {folder/name}')
    return m


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    temp.replace(path)
