"""Bind new source to frozen V10 without entering any parent context/lock."""
from dataclasses import dataclass
from pathlib import Path
import importlib.metadata
import platform
import shutil
import numpy as np
import pandas as pd
import yaml
from src.artifacts import ROOT, canonical_hash, read_json, sha256_file, write_json
from research_v6.common import verify_files, write_manifest

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = '20261006_morphology_regularized_features_v11'
LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
FEATURE_ARMS = {'qrs_rr','morphology','morphology_rr','fusion_random','fusion_pretrained'}


def validate_run_id(run_id):
    if not isinstance(run_id,str) or not run_id or Path(run_id).name!=run_id or run_id in {'.','..'} or '\\' in run_id:
        raise ValueError('Single run-id required; path separators prohibited')
    return run_id


@dataclass
class Study:
    run: Path
    protocol: dict
    records: list
    folds: list
    classes: list
    parent_run: Path
    feature_folder: Path
    identity: dict

    @property
    def signature(self): return canonical_hash(self.identity)


def disk_check(study, additional_bytes=0):
    free=shutil.disk_usage(ROOT).free
    if free < study.protocol['minimum_free_gib']*1024**3 + additional_bytes:
        raise RuntimeError(f'Ruang kosong {free/1024**3:.2f} GiB. Perlu cadangan 3 GiB + '
            f'{additional_bytes/1024**2:.0f} MiB tahap ini. Kosongkan file non-riset lalu ulang; hasil lama dipertahankan.')
    return {'free_gib':free/1024**3,'additional_mib':additional_bytes/1024**2}


def context(run_id=DEFAULT_RUN, initialize=False):
    validate_run_id(run_id)
    p=yaml.safe_load((PACKAGE/'protocol.yaml').read_text())
    if p['historical_test_enabled'] or p['final_training_enabled'] or p['minimum_free_gib']<3:
        raise ValueError('Development-only experiment required')
    if p['folds']!=[1,2,3,4,5] or p['sampling_rate']!=2000 or len(p['inverse_regularization_C'])!=9:
        raise ValueError('Fixed five-fold/native/9-C protocol required')
    grid=p['inverse_regularization_C']
    if len(set(grid))!=9 or not all(np.isfinite(c) and c>0 for c in grid):raise ValueError('Nine unique positive C values required')
    parent=ROOT/'results/research_v10'/p['source_v10_run']
    old=read_json(parent/'protocol.json');audit=read_json(parent/'analysis_20261006_complete/audit.json')
    if canonical_hash({k:v for k,v in old.items() if k!='experiment_hash'})!=old['experiment_hash']:
        raise RuntimeError('V10 identity changed')
    if old['experiment_hash']!=audit['experiment_hash'] or audit['cases_verified']!=130 or not audit['all_saved_metrics_reproduced']:
        raise RuntimeError('Require complete audited V10')
    for name,digest in old['code'].items():
        if sha256_file(ROOT/'research_v10'/name)!=digest: raise RuntimeError('Frozen V10 code changed')
    if sha256_file(ROOT/'notebooks/_v10_runner.py')!=old['runner_sha256']: raise RuntimeError('Frozen V10 runner changed')
    # Existing feature arrays are reused; no encoder inference or parent initializer.
    cache_ref=read_json(parent/'feature_cache.json');folder=Path(cache_ref['cache'])
    if sha256_file(folder/'completed.json')!=cache_ref['manifest_sha256'] or cache_ref['manifest_sha256']!=audit['feature_cache_sha256']:
        raise RuntimeError('V10 feature manifest changed')
    done=read_json(folder/'completed.json')
    if canonical_hash({k:v for k,v in done.items() if k!='record_hash'})!=done['record_hash']:
        raise RuntimeError('Feature manifest content changed')
    used=['patients.csv','random_probe.npy','pretrained_probe.npy']
    for name in used:
        if sha256_file(folder/name)!=done['files'][name]: raise RuntimeError(f'Used V10 feature changed: {name}')
    patients=pd.read_csv(folder/'patients.csv',dtype={'hospital_id':str})
    records=[r for r in read_json(ROOT/p['records_source']) if r['pool']=='development']
    records=sorted(records,key=lambda r:r['hospital_id'])
    if len(records)!=230 or patients.hospital_id.tolist()!=[r['hospital_id'] for r in records]:
        raise RuntimeError('Unchanged 230-patient development required')
    if patients.label.tolist()!=[r['label'] for r in records]: raise RuntimeError('Labels changed')
    ids={r['hospital_id'] for r in records}
    for split in old['folds']:
        a,b,c=map(set,(split['gan_fit_ids'],split['gan_reference_ids'],split['outer_validation_ids']))
        if a&b or (a|b)&c or a|b|c!=ids: raise RuntimeError('Incomplete/overlapping split')
    identity={'protocol':p,'classes':old['classes'],'folds':old['folds'],'records_hash':canonical_hash(records),
        'source_v10_hash':old['experiment_hash'],'source_audit_sha256':sha256_file(parent/'analysis_20261006_complete/audit.json'),
        'feature_manifest_sha256':cache_ref['manifest_sha256'],
        'code':{q.name:sha256_file(q) for q in sorted(PACKAGE.glob('*.py'))},
        'frozen_detector_sha256':sha256_file(ROOT/'research_v3/candidates.py'),
        'frozen_detector_protocol_sha256':sha256_file(ROOT/'research_v3/protocol.yaml'),
        'shared_code':{name:sha256_file(ROOT/name) for name in ('src/artifacts.py','src/evaluation.py','src/data.py','research_v6/common.py')},
        'runner_sha256':sha256_file(ROOT/'notebooks/_v11_runner.py'),
        'runtime':{'python':platform.python_version(),**{n:importlib.metadata.version(n) for n in ('numpy','pandas','scipy','scikit-learn','PyYAML','matplotlib','joblib','threadpoolctl')}},
        'test_signals_read':False,'pvc_beat_annotations_available':False,'pretraining_patient_independence_verified':False}
    study=Study(ROOT/'results/research_v11'/run_id,p,records,old['folds'],old['classes'],parent,folder,identity)
    saved={**identity,'experiment_hash':study.signature};path=study.run/'protocol.json'
    if not path.exists():
        if not initialize: raise RuntimeError('Jalankan notebook 57 / prepare dahulu')
        disk_check(study,16*1024**2);write_json(path,saved)
        write_json(study.run/'data/patients.json',records)
    if read_json(path)!=saved: raise RuntimeError('V11 source/runtime/protocol changed; preserve old run and use a new run-id')
    return study


def complete(directory, identity, names, **extra):
    write_manifest(directory/'completed.json',{'identity':identity,
        'files':{name:sha256_file(directory/name) for name in names},**extra})


def verified(directory, identity):
    path=directory/'completed.json'
    if not path.exists():return None
    m=read_json(path);verify_files(directory,m)
    if m['identity']!=identity:raise RuntimeError(f'Completed identity changed: {directory}')
    return m
