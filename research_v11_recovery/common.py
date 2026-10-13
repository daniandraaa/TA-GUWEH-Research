"""Verify frozen metadata without requiring embeddings for unrelated stages."""
from pathlib import Path
import importlib.metadata
import os
import platform
import pandas as pd
import yaml
from src.artifacts import ROOT,canonical_hash,read_json,sha256_file,write_json
from research_v11.common import Study,disk_check,validate_run_id

PACKAGE=Path(__file__).resolve().parent
SOURCE_RUN='20261006_morphology_regularized_features_v11'
DEFAULT_RUN='20261007_morphology_regularized_features_v11_r1'
CACHE_ARMS={'random_l2','pretrained_l2','fusion_random','fusion_pretrained'}


def load_origin():
    """Validate original source/runtime/cohort and audited parent; no cache access."""
    source=ROOT/'results/research_v11'/SOURCE_RUN
    saved=read_json(source/'protocol.json');identity={k:v for k,v in saved.items() if k!='experiment_hash'}
    if canonical_hash(identity)!=saved['experiment_hash']:raise RuntimeError('Original V11 identity changed')
    p=yaml.safe_load((ROOT/'research_v11/protocol.yaml').read_text())
    if p!=saved['protocol']:raise RuntimeError('Frozen V11 YAML changed')
    for name,digest in saved['code'].items():
        if sha256_file(ROOT/'research_v11'/name)!=digest:raise RuntimeError(f'Frozen V11 source changed: {name}')
    for name,digest in saved['shared_code'].items():
        if sha256_file(ROOT/name)!=digest:raise RuntimeError(f'Shared source changed: {name}')
    for path,key in [('research_v3/candidates.py','frozen_detector_sha256'),('research_v3/protocol.yaml','frozen_detector_protocol_sha256'),
                     ('notebooks/_v11_runner.py','runner_sha256')]:
        if sha256_file(ROOT/path)!=saved[key]:raise RuntimeError(f'Frozen dependency changed: {path}')
    runtime={'python':platform.python_version(),**{name:importlib.metadata.version(name) for name in saved['runtime'] if name!='python'}}
    if runtime!=saved['runtime']:raise RuntimeError('Runtime changed; preserve original run and review new protocol')
    parent=ROOT/'results/research_v10'/p['source_v10_run'];old=read_json(parent/'protocol.json')
    if canonical_hash({k:v for k,v in old.items() if k!='experiment_hash'})!=old['experiment_hash'] or old['experiment_hash']!=saved['source_v10_hash']:
        raise RuntimeError('Frozen V10 identity changed')
    for name,digest in old['code'].items():
        if sha256_file(ROOT/'research_v10'/name)!=digest:raise RuntimeError('Frozen V10 source changed')
    if sha256_file(ROOT/'notebooks/_v10_runner.py')!=old['runner_sha256']:raise RuntimeError('Frozen V10 runner changed')
    audit_path=parent/'analysis_20261006_complete/audit.json'
    if sha256_file(audit_path)!=saved['source_audit_sha256']:raise RuntimeError('Original V10 audit changed')
    audit=read_json(audit_path);ref=read_json(parent/'feature_cache.json')
    if audit['experiment_hash']!=old['experiment_hash'] or audit['cases_verified']!=130 or not audit['all_saved_metrics_reproduced']:
        raise RuntimeError('Complete audited V10 required')
    if ref['manifest_sha256']!=audit['feature_cache_sha256'] or ref['manifest_sha256']!=saved['feature_manifest_sha256']:
        raise RuntimeError('Original feature manifest reference changed')
    records=sorted([r for r in read_json(ROOT/p['records_source']) if r['pool']=='development'],key=lambda r:r['hospital_id'])
    if len(records)!=230 or canonical_hash(records)!=saved['records_hash'] or read_json(source/'data/patients.json')!=records:
        raise RuntimeError('Original development cohort changed')
    if old['folds']!=saved['folds'] or old['classes']!=saved['classes']:raise RuntimeError('Original folds/classes changed')
    ids={r['hospital_id'] for r in records}
    for split in old['folds']:
        a,b,c=map(set,(split['gan_fit_ids'],split['gan_reference_ids'],split['outer_validation_ids']))
        if a&b or (a|b)&c or a|b|c!=ids:raise RuntimeError('Split changed')
    return identity,records,parent,ref


def context(run_id=DEFAULT_RUN,initialize=False):
    validate_run_id(run_id)
    if run_id==SOURCE_RUN:raise ValueError('Preserve original V11 run; use the recovery run-id')
    origin,records,parent,ref=load_origin()
    identity={**origin,'execution_repair':{'source_run':SOURCE_RUN,
        'source_protocol_sha256':sha256_file(ROOT/'results/research_v11'/SOURCE_RUN/'protocol.json'),
        'code':{q.name:sha256_file(q) for q in sorted(PACKAGE.glob('*.py'))},
        'runner_sha256':sha256_file(ROOT/'notebooks/_v11_recovery_runner.py'),
        'cache_required_only_for_embedding_inputs':True,
        'embedding_inputs_must_match_original_v10_audit':True,
        'delineation_and_fit_algorithm_unchanged':True,'gate_thresholds_unchanged':True,
        'annotation_decoder':'retain_all_qrs_peaks_unavailable_reference_boundaries_remain_unavailable'}}
    folder=Path(os.environ.get('TA_V11_FEATURE_CACHE',ref['cache'])).expanduser()
    study=Study(ROOT/'results/research_v11'/run_id,origin['protocol'],records,origin['folds'],origin['classes'],parent,folder,identity)
    expected={**identity,'experiment_hash':study.signature};path=study.run/'protocol.json'
    if not path.exists():
        if not initialize:raise RuntimeError('Jalankan notebook 57 yang sudah diperbaiki / prepare dahulu')
        disk_check(study,16*1024**2);write_json(path,expected);write_json(study.run/'data/patients.json',records)
    if read_json(path)!=expected:raise RuntimeError('Recovery source/runtime/protocol changed; preserve run and use a new run-id')
    return study


def expected_feature_files(study):
    audit=read_json(study.parent_run/'analysis_20261006_complete/audit.json')
    return {Path(name).name:digest for name,digest in audit['feature_files_sha256'].items()}


def verify_cached_files(study,names):
    expected=expected_feature_files(study);used={}
    for name in names:
        path=study.feature_folder/name
        if not path.is_file():raise FileNotFoundError(f'Cache fitur V10 belum tersedia: {path}. '
            'Pulihkan dari backup; TA_V11_FEATURE_CACHE dapat menunjuk folder salinan. '
            'Benchmark LUDB dan arm qrs_rr/morphology/morphology_rr tidak memerlukan embedding. '
            'Tidak ada regenerasi encoder otomatis.')
        if name not in expected or sha256_file(path)!=expected[name]:raise RuntimeError(f'Cache berbeda dari audit V10: {name}; jangan mengganti fitur secara diam-diam')
        used[name]=expected[name]
    return used


def cache_status(study):
    status={'folder':str(study.feature_folder),'available':False,'arms':{},'verification':'original_v10_audited_file_hashes'}
    for encoder in ['random','pretrained']:
        try:
            verify_cached_files(study,['patients.csv',encoder+'_probe.npy']);current={'ready':True,'reason':None}
        except (FileNotFoundError,RuntimeError) as error:current={'ready':False,'reason':str(error)}
        status['arms'][encoder+'_l2']=current;status['arms']['fusion_'+encoder]=current
    status['available']=all(value['ready'] for value in status['arms'].values())
    return status


def load_bundle(study,arms):
    """Read only inputs required by requested arms; retain all patient labels/order."""
    import numpy as np
    from research_v11.common import FEATURE_ARMS
    from research_v11.benchmark import require_pass
    from research_v11.features import read_features
    from research_v11.morphology import columns_for
    if not arms or any(arm not in study.protocol['arms'] for arm in arms):raise ValueError('Unsupported arm')
    encoders={arm.split('_')[-1] if arm.startswith('fusion_') else arm.split('_')[0] for arm in arms if arm in CACHE_ARMS}
    patients=pd.DataFrame([{'hospital_id':r['hospital_id'],'label':r['label']} for r in study.records])
    sources={'v10_source_experiment_hash':study.identity['source_v10_hash']};arrays={}
    if encoders:
        names=['patients.csv',*[encoder+'_probe.npy' for encoder in sorted(encoders)]]
        sources['verified_used_v10_file_sha256']=verify_cached_files(study,names)
        sources['original_v10_feature_manifest_sha256']=study.identity['feature_manifest_sha256']
        cached=pd.read_csv(study.feature_folder/'patients.csv',dtype={'hospital_id':str})
        if cached.hospital_id.tolist()!=patients.hospital_id.tolist() or cached.label.tolist()!=patients.label.tolist():raise RuntimeError('Cached patient order/labels changed')
        for encoder in encoders:
            a=np.load(study.feature_folder/f'{encoder}_probe.npy',mmap_mode='r')
            if a.shape!=(230,768) or a.dtype!=np.float32 or not np.isfinite(a).all():raise RuntimeError('Invalid frozen features')
            arrays[encoder]=a
    morphology=None
    if any(arm in FEATURE_ARMS for arm in arms):
        sources['ludb_gate']=require_pass(study);morphology,sources['morphology_manifest']=read_features(study)
    matrices={}
    for arm in arms:
        if arm.endswith('_l2'):matrix=np.asarray(arrays[arm.split('_')[0]],dtype=float)
        else:
            matrix=morphology[columns_for(arm)].to_numpy(dtype=float)
            if arm.startswith('fusion_'):matrix=np.concatenate([np.asarray(arrays[arm.split('_')[1]],dtype=float),matrix],axis=1)
        if np.isinf(matrix).any():raise RuntimeError('Infinite features prohibited')
        matrices[arm]=matrix
    return patients,matrices,sources
