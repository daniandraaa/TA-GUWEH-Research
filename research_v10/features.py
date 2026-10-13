"""Manual, label-free frozen extraction on the 230 development patients only."""
import gc
import numpy as np
import pandas as pd
import torch
from src.artifacts import canonical_hash, read_json, sha256_file, write_json
from research_v6.common import verify_files, write_manifest
from research_v8.data import load_development
from .assets import checkpoint_path
from .common import storage_check
from .data import adapt, patient_indices
from .model import load_encoder


def feature_paths(study):
    return study.cache/'features'


def feature_identity(study):
    return {'experiment_hash': study.signature, 'dtype': 'float32', 'patient_ids': sorted(r['hospital_id'] for r in study.records),
        'random_backbone_seed': study.protocol['random_backbone_seed'],
        'checkpoint': study.inspection['model_only_sha256'], 'no_training': True, 'test_signals_read': False}


def require_features(study):
    folder = feature_paths(study)
    if not (folder/'completed.json').exists(): raise RuntimeError('Jalankan notebook 52 / extract dahulu')
    done = read_json(folder/'completed.json'); verify_files(folder, done)
    if done['identity'] != feature_identity(study): raise RuntimeError('Feature cache identity changed')
    return folder, done


def extract(study, device):
    folder = feature_paths(study); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completed.json').exists():
        require_features(study); print('Frozen feature cache verified; skip', flush=True); return
    identity = feature_identity(study); marker = folder/'progress.json'
    if marker.exists():
        progress = read_json(marker)
        if progress['identity'] != identity or progress['device'] != str(device): raise RuntimeError('Feature resume identity/device changed')
    else: progress = {'identity': identity, 'device': str(device), 'next_patient': {'random_probe':0, 'pretrained_probe':0}}
    X, table = load_development(study); groups = patient_indices(table, identity['patient_ids'])
    required = {'random_probe':['random_probe.npy'],
        'pretrained_probe':['pretrained_probe.npy','prelast_tokens.npy']}
    for arm,names in required.items():
        if progress['next_patient'][arm] and any(not (folder/n).exists() for n in names):
            raise RuntimeError('Incomplete resumed cache; preserve it and use a new run-id rather than skipping missing patients')
    dim = study.inspection['encoder_model_config']['encoder_embed_dim']; tokens = 78
    total = len(table)*tokens*dim*4 + len(groups)*dim*4*2
    existing = sum(q.stat().st_size for q in folder.glob('*.npy'))
    storage_check(study, max(0,total-existing)+64*1024**2)
    rows = pd.DataFrame([{'hospital_id':hid,'label':label,'windows':len(indices)} for hid,label,indices in groups])
    rows.to_csv(folder/'patients.csv',index=False)
    pd.DataFrame({'segment_index':np.arange(len(table)), 'hospital_id':table.hospital_id, 'label':table.label}).to_csv(folder/'windows.csv',index=False)
    for arm in study.protocol['probe_arms']:
        path = folder/f'{arm}.npy'
        means = np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='float32',shape=(len(groups),dim))
        if means.shape != (len(groups),dim) or means.dtype != np.float32: raise RuntimeError('Feature shape/dtype changed')
        prelast = None
        if arm == 'pretrained_probe':
            path = folder/'prelast_tokens.npy'
            prelast = np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='float32',shape=(len(table),tokens,dim))
            if prelast.shape != (len(table),tokens,dim) or prelast.dtype != np.float32: raise RuntimeError('Token shape/dtype changed')
        model = load_encoder(checkpoint_path(),study.inspection['embedded_model_config'],
            random_seed=study.protocol['random_backbone_seed'] if arm == 'random_probe' else None).to(device)
        with torch.no_grad():
            for pi in range(progress['next_patient'][arm],len(groups)):
                hid, label, indices = groups[pi]; parts = []
                for start in range(0,len(indices),study.protocol['feature_chunk']):
                    chosen = indices[start:start+study.protocol['feature_chunk']]
                    values = torch.from_numpy(adapt(X[chosen])).to(device)
                    if prelast is None: out = model(values)
                    else:
                        before = model.until_last(values); out = model.finish(before)
                        prelast[chosen] = before.cpu().numpy()
                    if out.shape[1:] != (tokens,dim) or not torch.isfinite(out).all(): raise RuntimeError('Invalid frozen features')
                    parts.append(out.mean(1).cpu().numpy())
                means[pi] = np.concatenate(parts).astype(np.float64).mean(0).astype(np.float32)
                means.flush()
                if prelast is not None: prelast.flush()
                progress['next_patient'][arm] = pi+1; write_json(marker,progress)
                print(f'{arm} patient {pi+1}/{len(groups)} windows={len(indices)}',flush=True)
        del model, means, prelast; gc.collect()
        if device.type == 'mps': torch.mps.empty_cache()
    files = ['random_probe.npy','pretrained_probe.npy','prelast_tokens.npy','patients.csv','windows.csv']
    write_manifest(folder/'completed.json',{'identity':identity,'device':str(device),
        'files':{n:sha256_file(folder/n) for n in files},'native_windows':len(table),'patients':len(groups),
        'cache_bytes':total,'genuine_development_inference':True,'classifier_training':False,'test_signals_read':False})
    write_json(study.run/'feature_cache.json',{'cache':str(folder),'manifest_sha256':sha256_file(folder/'completed.json')})
    print('Frozen real-only feature extraction complete',flush=True)


def open_features(study):
    folder, done = require_features(study)
    patients = pd.read_csv(folder/'patients.csv',dtype={'hospital_id':str})
    windows = pd.read_csv(folder/'windows.csv',dtype={'hospital_id':str})
    arrays = {arm:np.load(folder/f'{arm}.npy',mmap_mode='r') for arm in study.protocol['probe_arms']}
    arrays['tokens'] = np.load(folder/'prelast_tokens.npy',mmap_mode='r')
    if patients.hospital_id.tolist() != feature_identity(study)['patient_ids'] or len(windows) != done['native_windows']:
        raise RuntimeError('Patient/window metadata changed')
    labels = {r['hospital_id']:r['label'] for r in study.records}
    if any(labels[r.hospital_id] != r.label for r in patients.itertuples()): raise RuntimeError('Feature labels changed')
    return patients, windows, arrays, sha256_file(folder/'completed.json')
