"""Manual native-continuous extraction, retaining every development patient."""
import json
import numpy as np
import pandas as pd
from src.artifacts import ROOT, canonical_hash, read_json, sha256_file, write_json
from .common import LEADS, complete, verified, disk_check
from .morphology import measure_record, columns_for


def raw_manifest(study):
    files={}
    for record in study.records:
        path=ROOT/study.protocol['raw_directory']/f"{record['hospital_id']}.csv"
        files[record['hospital_id']]=sha256_file(path)
    return files


def extract(study):
    destination=study.run/'features';destination.mkdir(parents=True,exist_ok=True)
    hashes=raw_manifest(study);identity={'experiment_hash':study.signature,'raw_development_sha256':hashes,
        'patient_ids':[r['hospital_id'] for r in study.records],'pvc_verified':False}
    if verified(destination,identity):print('Feature extraction complete and verified; skip',flush=True);return
    disk_check(study,32*1024**2);rows=[];quality=[];shards=[]
    write_json(destination/'identity.json',identity)
    for index,record in enumerate(study.records,1):
        hid=record['hospital_id'];folder=destination/'patients'/hid;folder.mkdir(parents=True,exist_ok=True)
        current={'experiment_hash':study.signature,'hospital_id':hid,'label':record['label'],'raw_sha256':hashes[hid]}
        if verified(folder,current):
            row=read_json(folder/'features.json');qa=read_json(folder/'quality.json')
        else:
            raw=pd.read_csv(ROOT/study.protocol['raw_directory']/f'{hid}.csv',usecols=LEADS)[LEADS].to_numpy(dtype=float)
            row,qa,beats,_=measure_record(raw,study.protocol['sampling_rate'],study.protocol)
            row={name:None if not np.isfinite(value) else value for name,value in row.items()}
            write_json(folder/'features.json',row);write_json(folder/'quality.json',qa)
            pd.DataFrame(beats).to_csv(folder/'candidate_measurements.csv',index=False)
            complete(folder,current,['features.json','quality.json','candidate_measurements.csv'])
        rows.append({'hospital_id':hid,'label':record['label'],**row});quality.append({'hospital_id':hid,'label':record['label'],**qa})
        shards.append(str((folder/'completed.json').relative_to(destination)))
        print(f'Morphology {index}/230 candidates={qa["complete_candidates"]} missing={qa["feature_missing_fraction"]:.1%}',flush=True)
    pd.DataFrame(rows).to_csv(destination/'patient_features.csv',index=False)
    pd.DataFrame(quality).to_csv(destination/'patient_quality.csv',index=False)
    write_json(destination/'schema.json',{'units':'native amplitude units unverified; relative amplitudes share per-record global scale',
        'groups':{arm:columns_for(arm) for arm in ['qrs_rr','morphology','morphology_rr']},
        'aggregation':'median/IQR of all complete QRS candidates; no PVC selection',
        'patients_excluded':0,'quality_fields_not_classifier_inputs':list(quality[0])})
    complete(destination,identity,['identity.json','patient_features.csv','patient_quality.csv','schema.json',*shards],patients=230,training_performed=False,test_signals_read=False)


def read_features(study):
    folder=study.run/'features'
    if not (folder/'completed.json').exists():raise RuntimeError('Jalankan notebook 59 / extract dahulu')
    identity=read_json(folder/'identity.json');verified(folder,identity)
    if identity['experiment_hash']!=study.signature:raise RuntimeError('Feature protocol changed')
    frame=pd.read_csv(folder/'patient_features.csv',dtype={'hospital_id':str})
    if frame.hospital_id.tolist()!=[r['hospital_id'] for r in study.records] or frame.label.tolist()!=[r['label'] for r in study.records]:
        raise RuntimeError('Morphology patient order/labels changed')
    # Check per-patient shard files and source raw identity on resume/fit.
    for record in study.records:
        hid=record['hospital_id'];current={'experiment_hash':study.signature,'hospital_id':hid,'label':record['label'],'raw_sha256':identity['raw_development_sha256'][hid]}
        if sha256_file(ROOT/study.protocol['raw_directory']/f'{hid}.csv')!=current['raw_sha256']:raise RuntimeError('Raw source changed')
        verified(folder/'patients'/hid,current)
    return frame,sha256_file(folder/'completed.json')


def qa(study):
    frame,_=read_features(study);folder=study.run/'features';quality=pd.read_csv(folder/'patient_quality.csv',dtype={'hospital_id':str})
    summary={'patients':len(frame),'patients_excluded':0,'no_complete_candidate_patients':int(quality.complete_candidates.eq(0).sum()),
        'median_feature_missing_fraction':float(quality.feature_missing_fraction.median()),'pvc_verified':False,
        'clinical_measurement_validation':False,'no_training':True,'test_signals_read':False}
    write_json(folder/'quality_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    print(quality.groupby('label')[['complete_candidates','feature_missing_fraction']].median().to_string(),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].hist(quality.feature_missing_fraction,bins=20);axes[0].set(xlabel='Fraction of missing candidate features',ylabel='Patients',title='All 230 patients retained')
    axes[1].hist(quality.complete_candidates,bins=20);axes[1].set(xlabel='Complete candidate QRS per continuous recording',ylabel='Patients',title='Candidate counts are not verified PVC counts')
    fig.tight_layout();fig.savefig(folder/'feature_quality.png',dpi=150);plt.close(fig)
    # Fixed first inner-training patient in each class, never selected by score.
    inner=set(study.folds[0]['gan_fit_ids']);anchors=[next(r for r in study.records if r['label']==label and r['hospital_id'] in inner) for label in range(3)]
    for record in anchors:
        hid=record['hospital_id'];raw=pd.read_csv(ROOT/study.protocol['raw_directory']/f'{hid}.csv',usecols=LEADS)[LEADS].to_numpy(dtype=float)
        _,_,beats,clean=measure_record(raw,study.protocol['sampling_rate'],study.protocol)
        if not beats:continue
        # First complete candidate, not the widest or visually preferred beat.
        center=beats[0]['center_sample'];fs=study.protocol['sampling_rate'];lo=center-round(.25*fs);hi=center+round(.3*fs)
        fig,axes=plt.subplots(4,3,figsize=(11,8),sharex=True)
        for j,ax in enumerate(axes.flat):
            ax.plot((np.arange(lo,hi)-center)*1000/fs,clean[lo:hi,j],lw=.8);ax.axvline(0,color='gray',ls='--',lw=.6);ax.set_title(LEADS[j])
            for edge,color in [('onset','#17865c'),('offset','#bb4f23')]:
                value=beats[0][f'{LEADS[j]}_{edge}_sample']
                if value is not None:ax.axvline((value-center)*1000/fs,color=color,lw=.8)
        fig.suptitle(f'Training anchor class {record["label"]}: first candidate (PVC unverified)');fig.supxlabel('Time relative to candidate center (ms)');fig.supylabel('Native signal units (not assumed mV)')
        fig.tight_layout();fig.savefig(folder/f'anchor_class_{record["label"]}.png',dpi=130);plt.close(fig)
    return summary
