"""User-run genuine inference diagnostics of saved weights; never train/test."""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.artifacts import ROOT, canonical_hash, choose_device, lock, read_json, sha256_file, write_json
from src.evaluation import compute_metrics, prediction_table
from research_v6.common import verify_files, write_manifest
from research_v8.data import load_development
from research_v10.analysis import verify_case
from research_v10.assets import checkpoint_path
from research_v10.common import DEFAULT_RUN, context, case_directory
from research_v10.data import adapt, patient_indices
from research_v10.experiment import make_datasets
from research_v10.features import open_features
from research_v10.model import load_encoder, TailClassifier
from research_v10.training import FeatureClassifier, predict

DESTINATION='diagnostics_20261006_v1'


def compare_arrays(actual, reference, rtol=1e-3, atol=1e-4):
    actual=np.asarray(actual,dtype=np.float64);reference=np.asarray(reference,dtype=np.float64)
    if actual.shape!=reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise RuntimeError('Shape/nonfinite mismatch in inference replay')
    error=np.abs(actual-reference)
    return {'shape':list(actual.shape),'max_absolute_error':float(error.max()),'mean_absolute_error':float(error.mean()),
        'max_relative_error_with_atol_floor':float((error/np.maximum(np.abs(reference),atol)).max()),
        'rtol':rtol,'atol':atol,'within_tolerance':bool(np.allclose(actual,reference,rtol=rtol,atol=atol))}


def identity_for(study,device,stage):
    audit_path=study.run/'analysis_20261006_complete/audit.json';audit=read_json(audit_path)
    if audit['experiment_hash']!=study.signature or audit['cases_verified']!=130 or not audit['all_saved_metrics_reproduced']:
        raise RuntimeError('Require the completed independent V10 audit')
    return {'experiment_hash':study.signature,'completed_audit_sha256':sha256_file(audit_path),
        'diagnostic_code_sha256':sha256_file(Path(__file__)),'device':str(device),'stage':stage,
        'policy':'manual_inference_only_no_fit_no_checkpoint_selection_no_test'}


def replay_case(study,folder,device,bundle,encoder=None):
    case,result,stored=verify_case(study,folder)
    patients,windows,arrays,cache_hash=bundle
    if case['feature_cache_manifest']!=cache_hash:raise RuntimeError('Replay feature cache changed')
    train,evaluation,mean,scale=make_datasets(study,case['arm'],case['train_ids'],case['evaluation_ids'],patients,windows,arrays)
    model=(TailClassifier(encoder,mean,scale) if case['arm']=='pretrained_tail' else FeatureClassifier(mean,scale)).to(device)
    state=torch.load(folder/'selected.pth',map_location='cpu',weights_only=True)
    if sha256_file(folder/'selected.pth')!=result['selected_checkpoint_sha256']:raise RuntimeError('Selected weights changed')
    model.load_state_dict(state,strict=True);del state
    ids,truth,probabilities=predict(model,evaluation,device)
    reference=stored[[f'probability_{name}' for name in study.classes]].to_numpy()
    parity=compare_arrays(probabilities,reference,rtol=1e-4,atol=1e-5)
    if ids!=stored.hospital_id.tolist():raise RuntimeError('Replay patient order changed')
    parity['argmax_identical']=bool(np.array_equal(probabilities.argmax(1),reference.argmax(1)))
    train_ids,train_truth,train_probabilities=predict(model,train,device)
    evaluation_metrics=compute_metrics(truth,probabilities);train_metrics=compute_metrics(train_truth,train_probabilities)
    row={'arm':case['arm'],'fold':case['fold'],'seed':case['seed'],'phase':'inner_selected' if case['inner_selection'] else 'outer_fixed_refit',
        'case':str(folder.relative_to(study.run)),'selected_epoch':result['selected_epoch'],
        'training_patients':len(train),'evaluation_patients':len(evaluation),
        'train_macro_f1':train_metrics['f1'],'evaluation_macro_f1':evaluation_metrics['f1'],
        'f1_gap_train_minus_evaluation':train_metrics['f1']-evaluation_metrics['f1'],
        'train_accuracy':train_metrics['accuracy'],'evaluation_accuracy':evaluation_metrics['accuracy'],
        'saved_evaluation_f1':result['f1'],'replay_max_probability_error':parity['max_absolute_error'],
        'replay_probabilities_within_tolerance':parity['within_tolerance'],'replay_argmax_identical':parity['argmax_identical'],
        'selected_checkpoint_sha256':sha256_file(folder/'selected.pth')}
    frames={'train':prediction_table(train_ids,train_truth,train_probabilities,study.classes),
        'evaluation':prediction_table(ids,truth,probabilities,study.classes)}
    del model;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    return row,frames


def fit_replay(study,device,destination):
    bundle=open_features(study);encoder=None;rows=[];files=[]
    try:
        for arm in [*study.protocol['probe_arms'],'pretrained_tail']:
            choice=read_json(study.run/'choices'/arm/'fold_1.json')
            if arm=='pretrained_tail':encoder=load_encoder(checkpoint_path(),study.inspection['embedded_model_config'])
            for scenario in [choice['scenario'],None]:
                folder=case_directory(study,arm,1,42,scenario)
                print(f'REPLAY SAVED {arm}: {folder.name} ({device})',flush=True)
                row,frames=replay_case(study,folder,device,bundle,encoder);rows.append(row)
                for split,frame in frames.items():
                    name=f'{arm}_{row["phase"]}_{split}_{device.type}.csv'
                    frame.to_csv(destination/name,index=False);files.append(name)
        name=f'fold1_saved_fit_replay_{device.type}.csv';pd.DataFrame(rows).to_csv(destination/name,index=False);files.append(name)
    finally:
        del encoder,bundle;gc.collect()
    table=pd.DataFrame(rows)
    print(table.to_string(index=False),flush=True)
    passed=bool(table.replay_probabilities_within_tolerance.all() and table.replay_argmax_identical.all())
    return files,{'saved_models_replayed':len(rows),'outer_models':3,'inner_models':3,
        'all_stored_evaluation_probabilities_reproduced':passed,
        'genuine_inference_performed':True,'training_performed':False,'test_signals_read':False,
        'interpretation':'posthoc_training_evaluation_gap_diagnostic_not_new_model_selection'}


@torch.no_grad()
def encode_patient(model,X,indices,device,chunk,include_tokens):
    pooled=[];upstream=[]
    model.to(device);model.eval()
    for start in range(0,len(indices),chunk):
        selected=indices[start:start+chunk]
        values=torch.from_numpy(adapt(X[selected])).to(device)
        if include_tokens:
            before=model.until_last(values);out=model.finish(before);upstream.append(before.cpu().numpy())
        else:out=model(values)
        if not torch.isfinite(out).all():raise RuntimeError('Nonfinite genuine encoder output')
        pooled.append(out.mean(1).cpu().numpy())
    mean=np.concatenate(pooled).astype(np.float64).mean(0).astype(np.float32)
    return mean,np.concatenate(upstream) if include_tokens else None


def encoder_replay(study,device,destination):
    patients,windows,arrays,cache_hash=open_features(study)
    fit_ids=study.folds[0]['gan_fit_ids'];labels={r['hospital_id']:r['label'] for r in study.records}
    # Anchors are chosen by deterministic ID within each training class, not by score/error.
    anchors=[next(hid for hid in sorted(fit_ids) if labels[hid]==c) for c in range(3)]
    X,table=load_development(study);groups=patient_indices(table,anchors)
    lookup={hid:i for i,hid in enumerate(patients.hospital_id)};rows=[]
    for arm in study.protocol['probe_arms']:
        model=load_encoder(checkpoint_path(),study.inspection['embedded_model_config'],
            random_seed=study.protocol['random_backbone_seed'] if arm=='random_probe' else None)
        include=arm=='pretrained_probe'
        for hid,label,indices in groups:
            print(f'ENCODER REPLAY {arm}: class={study.classes[label]}, windows={len(indices)}, CPU → {device}',flush=True)
            cpu,cpu_tokens=encode_patient(model,X,indices,torch.device('cpu'),study.protocol['feature_chunk'],include)
            target,target_tokens=(cpu,cpu_tokens) if device.type=='cpu' else encode_patient(model,X,indices,device,study.protocol['feature_chunk'],include)
            reference=np.asarray(arrays[arm][lookup[hid]])
            comparisons=[('cpu_vs_target_mean',cpu,target),('target_vs_cached_patient_mean',target,reference),
                ('cpu_vs_cached_patient_mean',cpu,reference)]
            if include:
                cached_tokens=np.asarray(arrays['tokens'][indices])
                comparisons.extend([('cpu_vs_target_upstream_tokens',cpu_tokens,target_tokens),
                    ('target_vs_cached_upstream_tokens',target_tokens,cached_tokens)])
            for name,a,b in comparisons:
                rows.append({'arm':arm,'hospital_id':hid,'label':label,'windows':len(indices),'comparison':name,**compare_arrays(a,b)})
        model.to('cpu');del model;gc.collect()
        if device.type=='mps':torch.mps.empty_cache()
    frame=pd.DataFrame(rows);name=f'encoder_cache_replay_{device.type}.csv';frame.to_csv(destination/name,index=False)
    print(frame[['arm','hospital_id','comparison','max_absolute_error','within_tolerance']].to_string(index=False),flush=True)
    return [name],{'training_anchor_patients':anchors,'anchor_selection':'first_sorted_training_id_per_class_fold1',
        'all_cache_and_device_comparisons_within_tolerance':bool(frame.within_tolerance.all()),
        'cpu_mps_comparison_performed':device.type=='mps','genuine_development_waveform_inference_performed':True,
        'training_performed':False,'test_signals_read':False,
        'scope':'three_training_patients_only_not_all_patient_backend_parity_or_clinical_validation'}


def run(stage,device_name='mps',run_id=DEFAULT_RUN):
    study=context(run_id);device=choose_device(device_name)
    destination=study.run/DESTINATION;destination.mkdir(parents=True,exist_ok=True)
    identity=identity_for(study,device,stage);marker=destination/f'{stage}_{device.type}_completed.json'
    with lock(study.run/'.v10_stage.lock'):
        if marker.exists():
            saved=read_json(marker);verify_files(destination,saved)
            if saved['identity']!=identity:raise RuntimeError('Diagnostic source/device changed; preserve existing diagnostic and use a new diagnostic version')
            passed=saved['result'].get('all_stored_evaluation_probabilities_reproduced',
                saved['result'].get('all_cache_and_device_comparisons_within_tolerance',False))
            if not passed:raise RuntimeError('Saved diagnostic did not pass; inspect its results before new training')
            print('SKIP verified saved diagnostic',flush=True);return
        files,result=(fit_replay(study,device,destination) if stage=='fits' else encoder_replay(study,device,destination))
        result_path=destination/f'{stage}_{device.type}_result.json';write_json(result_path,result);files.append(result_path.name)
        write_manifest(marker,{'identity':identity,'result':result,'files':{name:sha256_file(destination/name) for name in files}})
        if stage=='fits' and not result['all_stored_evaluation_probabilities_reproduced']:
            raise RuntimeError('Inference replay differs from stored probabilities; read diagnostic CSV before new training')
        if stage=='encoder' and not result['all_cache_and_device_comparisons_within_tolerance']:
            raise RuntimeError('Encoder/cache/backend comparison differs; read diagnostic CSV before new training')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['fits','encoder'])
    parser.add_argument('--device',default='mps',choices=['cpu','mps']);parser.add_argument('--run-id',default=DEFAULT_RUN)
    args=parser.parse_args();run(args.stage,args.device,args.run_id)


if __name__=='__main__':main()
