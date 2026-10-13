"""Optional gated candidate export; all development patients remain present."""
import joblib
import numpy as np
import pandas as pd
from src.artifacts import ROOT,read_json,write_json,sha256_file
from .common import LEADS,complete,verified,disk_check,atomic_npz
from .features import choose_leads,detect,extract_array
from .experiment import probabilities,require_deployment
from .metrics import pvc_decisions


def selected_indices(prob,threshold,normal_threshold,k):
    prob=np.asarray(prob);pvc=np.flatnonzero(pvc_decisions(prob,threshold))
    normal=np.flatnonzero((prob[:,0]>=normal_threshold)&(prob.argmax(axis=1)==0))
    def top(indices,col):
        # Prespecified score rank, stable tie-break index; return time order.
        return sorted(sorted(indices,key=lambda i:(-prob[i,col],int(i)))[:k])
    return top(pvc,1),top(normal,0)


def export(study):
    p=study.protocol;deployment=study.run/'deployment';external=study.run/'external_evaluation'
    if not verified(deployment) or not verified(external):
        raise RuntimeError('Deployment dan evaluasi eksternal lengkap diperlukan')
    evaluation=read_json(external/'summary.json');choice=read_json(deployment/'choice.json')
    if evaluation['choice']!=choice:
        raise RuntimeError('External evaluation must score the same frozen deployment')
    if not evaluation['engineering_gate_pass']:
        raise RuntimeError('Gate eksternal FAIL: jangan ekspor input dua irama lokal. Analisis kegagalan dari 67; threshold/gate tidak diturunkan untuk melewati kegagalan.')
    require_deployment(study)
    verified(external,{'experiment_hash':study.signature,
        'deployment_manifest_sha256':sha256_file(deployment/'completed.json'),
        'external_feature_manifest_sha256':sha256_file(study.run/'features/mitdb/completed.json')})
    if not verified(study.run/'data',{'experiment_hash':study.signature}):
        raise RuntimeError('Fixed development patient manifest required')
    records=read_json(study.run/'data/local_development.json')
    if len(records)!=230:
        raise RuntimeError('All 230 development patients required')
    fs=p['local_sampling_rate'];before=round(fs*p['crop_before_seconds']);after=round(fs*p['crop_after_seconds']);length=before+after
    k=p['local_maximum_beats_per_type'];model=joblib.load(deployment/'pipeline.joblib')
    folder=study.run/'local_candidates'
    identity={'experiment_hash':study.signature,'deployment_manifest_sha256':sha256_file(deployment/'completed.json'),
        'external_evaluation_manifest_sha256':sha256_file(external/'completed.json'),
        'patient_records_sha256':sha256_file(study.run/'data/local_development.json'),
        'raw_sha256':{r['hospital_id']:sha256_file(ROOT/p['local_raw_directory']/f'{r["hospital_id"]}.csv') for r in records}}
    if verified(folder,identity):
        print('Kandidat lokal lengkap; dilewati.',flush=True);return
    disk_check(study,230*1024**2);summary=[];names=[]
    for index,r in enumerate(records,1):
        hid=r['hospital_id'];sub=folder/hid;current={'experiment_hash':study.signature,'patient':hid,
            'raw_sha256':identity['raw_sha256'][hid],'deployment':identity['deployment_manifest_sha256'],
            'external_gate_manifest':identity['external_evaluation_manifest_sha256']}
        if not verified(sub,current):
            raw=pd.read_csv(ROOT/p['local_raw_directory']/f'{hid}.csv',usecols=LEADS)[LEADS].to_numpy(dtype=float)
            indices=choose_leads(LEADS,'local',p);selected_raw=raw[:,indices]
            centers,qa=detect(selected_raw,fs,p);x=extract_array(selected_raw,centers,fs,p)
            prob=probabilities(model,[{'features':x}],p,choice['arm'])[0]
            a,b=selected_indices(prob,choice['threshold'],p['external_gate']['normal_N_threshold'],k)
            crops=np.zeros((2,k,12,length),dtype=np.float32);mask=np.zeros((2,k),dtype=bool)
            locations=np.full((2,k),-1,dtype=np.int64)
            for kind,items in enumerate([a,b]):
                for j,i in enumerate(items):
                    center=int(centers[i]);crops[kind,j]=raw[center-before:center+after].T
                    mask[kind,j]=True;locations[kind,j]=center
            # Mid-record fallback retains the patient, with an explicit separate
            # slot that is never labelled PVC or sinus.
            start=max(0,(len(raw)-length)//2);valid=min(length,len(raw)-start)
            fallback=np.zeros((12,length),dtype=np.float32);fallback[:,:valid]=raw[start:start+valid].T
            atomic_npz(sub/'candidate_crops.npz',crops=crops,valid_beat_mask=mask,centers=locations,
                fallback_window=fallback,fallback_valid_samples=np.array(valid))
            pd.DataFrame({'center_sample':centers,'score_normal_N':prob[:,0],'score_PVC':prob[:,1],'score_other':prob[:,2]}).to_csv(sub/'candidate_scores.csv',index=False)
            row={'hospital_id':hid,'candidate_count':len(centers),'PVC_candidate_selected':len(a),'normal_N_candidate_selected':len(b),
                'both_types_available':bool(a and b),'needs_untyped_fallback':not bool(a and b),
                'pvc_verified':False,'sinus_verified':False,'patient_excluded':False,
                'crop_shape':[2,k,12,length],'raw_units':'native_unverified_not_assumed_mV',
                'score_calibrated':False,'detector':qa,'anatomical_label_used_for_selection':False}
            write_json(sub/'selection.json',row)
            complete(sub,current,['candidate_crops.npz','candidate_scores.csv','selection.json'])
        summary.append(read_json(sub/'selection.json'));names.append(f'{hid}/completed.json')
        print(f'Lokal {index}/230: kandidat tersimpan, identitas klinis belum terverifikasi',flush=True)
    flat=pd.DataFrame([{key:value for key,value in row.items() if not isinstance(value,(list,dict))} for row in summary])
    flat.to_csv(folder/'coverage.csv',index=False)
    write_json(folder/'summary.json',{'patients':230,'patients_excluded':0,
        'both_types_available':int(flat.both_types_available.sum()),'needs_untyped_fallback':int(flat.needs_untyped_fallback.sum()),
        'ground_truth_local_beat_labels':False,'origin_training_performed':False,
        'sinus_claim_allowed':False,'interpretation':'candidate PVC / candidate normal-N, masked padding and separate untyped fallback'})
    complete(folder,identity,[*names,'coverage.csv','summary.json'])
    print(read_json(folder/'summary.json'),flush=True)
