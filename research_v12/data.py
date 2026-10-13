"""Read annotations/splits separately from manual waveform extraction."""
from collections import Counter
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.artifacts import ROOT, read_json, write_json, sha256_file
from .common import wfdb_module, complete, verified, disk_check, atomic_npz
from .labels import annotation_rows, match_labels
from .features import choose_leads, detect, extract_array, feature_names

INCART_RECORDS=[f'I{i:02}' for i in range(1,76)]
MIT_RECORDS=['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']


def record_files(folder, name):
    return {f'{name}.{ext}':sha256_file(folder/f'{name}.{ext}') for ext in ['hea','dat','atr']}


def source_group(header, dataset, name):
    if dataset=='mitdb':
        # PhysioNet explicitly identifies 201/202 as the same subject.
        return 'MIT:201_202' if name in {'201','202'} else 'MIT:'+name
    numbers=[int(m.group(1)) for c in header.comments if (m:=re.fullmatch(r'patient\s+(\d+)',c.strip(),re.I))]
    if len(numbers)!=1:
        raise RuntimeError(f'INCART source-Holter group unavailable: {name}')
    return f'INCART:{numbers[0]:02}'


def metadata(folder, names, dataset, p):
    wfdb=wfdb_module();rows=[];label_arrays=[];group_arrays=[]
    for name in names:
        if any(not (folder/f'{name}.{ext}').is_file() for ext in ['hea','dat','atr']):
            raise FileNotFoundError(f'Dataset belum lengkap: {folder/name}. Download hanya lewat endpoint manual.')
        header=wfdb.rdheader(str(folder/name));annotation=wfdb.rdann(str(folder/name),'atr')
        if dataset=='incart' and (header.fs!=257 or header.n_sig!=12):
            raise RuntimeError(f'INCART header tidak sesuai: {name}')
        refs,edge=annotation_rows(annotation.sample,annotation.symbol,header.sig_len,header.fs,p)
        group=source_group(header,dataset,name)
        y=np.array([r[1] for r in refs if r[1]>=0],dtype=int)
        label_arrays.append(y);group_arrays.append(np.repeat(group,len(y)))
        rows.append({'record':name,'group':group,'fs':float(header.fs),'samples':int(header.sig_len),
            'lead_names':header.sig_name,'selected_leads':[header.sig_name[i] for i in choose_leads(header.sig_name,dataset,p)],
            'units':header.units,'assets':record_files(folder,name),
            'eligible_reference_counts':{str(k):int(np.sum(y==k)) for k in range(3)},
            'eligible_unknown_Q':sum(r[1]<0 for r in refs),'edge_reference_excluded':edge,
            'annotation_symbols':dict(Counter(annotation.symbol))})
    return rows,np.concatenate(label_arrays),np.concatenate(group_arrays)


def group_folds(y, groups, p):
    y=np.asarray(y);groups=np.asarray(groups)
    if set(np.unique(y))!={0,1,2}:
        raise ValueError('All three annotated beat classes required')
    cv=StratifiedGroupKFold(n_splits=p['folds'],shuffle=True,random_state=p['seed'])
    result=[];seen=set()
    for fold,(train,test) in enumerate(cv.split(np.zeros((len(y),1)),y,groups),1):
        inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=p['seed']+fold)
        inner_fit,inner_val=next(inner.split(np.zeros((len(train),1)),y[train],groups[train]))
        fit=train[inner_fit];val=train[inner_val]
        a,b,c=[set(groups[idx]) for idx in [fit,val,test]]
        if a&b or (a|b)&c or a|b|c!=set(groups):
            raise RuntimeError('Overlapping/incomplete source groups')
        if any(set(y[idx])!={0,1,2} for idx in [fit,val,train,test]):
            raise RuntimeError('A split lacks a beat class; do not silently discard a class')
        if seen&c:
            raise RuntimeError('Outer group repeated')
        seen|=c
        result.append({'fold':fold,'inner_training_groups':sorted(a),'inner_validation_groups':sorted(b),
            'outer_training_groups':sorted(a|b),'outer_validation_groups':sorted(c)})
    if seen!=set(groups):
        raise RuntimeError('Missing outer source group')
    return result


def prepare(study):
    folder=study.run/'data';identity={'experiment_hash':study.signature}
    if verified(folder,identity):
        saved=read_json(folder/'incart.json')
        verify_source(ROOT/study.protocol['incart_directory'],saved)
        print('Persiapan INCART dan split lengkap; dilewati.',flush=True)
        return saved
    disk_check(study,16*1024**2)
    rows,y,g=metadata(ROOT/study.protocol['incart_directory'],INCART_RECORDS,'incart',study.protocol)
    if len(set(g))!=32:
        raise RuntimeError('Expected 32 INCART source-Holter groups')
    saved={'dataset':'INCART 1.0.0','records':rows,'groups':sorted(set(g)),
        'folds':group_folds(y,g,study.protocol),'annotation_class_counts':{str(k):int(np.sum(y==k)) for k in range(3)},
        'waveforms_processed':False,'training_performed':False}
    write_json(folder/'incart.json',saved)
    local=[r for r in read_json(ROOT/study.protocol['local_records']) if r['pool']=='development']
    if len(local)!=230 or len({r['hospital_id'] for r in local})!=230:
        raise RuntimeError('Expected fixed 230 local development patients')
    write_json(folder/'local_development.json',sorted(local,key=lambda r:r['hospital_id']))
    complete(folder,identity,['incart.json','local_development.json'])
    print(f'INCART: 75 rekaman, 32 grup; kelas anotasi {saved["annotation_class_counts"]}; lima outer fold. Tidak ada training.',flush=True)
    return saved


def verify_source(folder, saved):
    for row in saved['records']:
        if record_files(folder,row['record'])!=row['assets']:
            raise RuntimeError(f'Sumber dataset berubah: {row["record"]}')


def extraction(study,dataset='incart',download=False):
    if dataset not in {'incart','mitdb'}:
        raise ValueError('Dataset harus incart atau mitdb')
    p=study.protocol
    if dataset=='incart':
        if download:
            raise ValueError('INCART sudah tersedia; tidak ditimpa/download ulang')
        if not verified(study.run/'data',{'experiment_hash':study.signature}):
            raise RuntimeError('Prepare dahulu')
        saved=read_json(study.run/'data/incart.json');source=ROOT/p['incart_directory']
    else:
        # Freeze the complete INCART selection before reading/downloading the
        # external annotations. This endpoint does not fit or tune a model.
        from .experiment import require_deployment
        require_deployment(study)
        source=ROOT/p['external_directory'];snapshot=study.run/'external_source.json'
        if snapshot.exists():
            saved=read_json(snapshot);verify_source(source,saved)
        else:
            missing=[name for name in MIT_RECORDS if any(not (source/f'{name}.{ext}').is_file() for ext in ['hea','dat','atr'])]
            if missing:
                if not download:
                    raise RuntimeError('MIT-BIH belum lengkap. Di notebook 67 set DOWNLOAD_MITBIH=True lalu jalankan sel download/extract manual.')
                disk_check(study,140*1024**2);source.mkdir(parents=True,exist_ok=True)
                wfdb_module().dl_database('mitdb',str(source),records=missing,annotators=['atr'],keep_subdirs=False,overwrite=False)
            rows,_,groups=metadata(source,MIT_RECORDS,'mitdb',p)
            saved={'dataset':'MIT-BIH 1.0.0','records':rows,'groups':sorted(set(groups)),
                'role':'external_detector_evaluation_only_no_tuning','origin_labels_available':False}
            if len(saved['groups'])!=47:
                raise RuntimeError('Expected 47 MIT-BIH subject groups')
            write_json(snapshot,saved)
    verify_source(source,saved)
    destination=study.run/'features'/dataset
    identity={'experiment_hash':study.signature,'dataset':dataset,'source':saved,'feature_names':feature_names(p,True)}
    if verified(destination,identity):
        print(f'{dataset} features lengkap dan terverifikasi; dilewati.',flush=True);return
    disk_check(study,150*1024**2);names=[]
    wfdb=wfdb_module()
    for index,row in enumerate(saved['records'],1):
        name=row['record'];folder=destination/name
        current={'experiment_hash':study.signature,'source':row,'dataset':dataset}
        if not verified(folder,current):
            record=wfdb.rdrecord(str(source/name));ann=wfdb.rdann(str(source/name),'atr')
            indices=choose_leads(record.sig_name,dataset,p)
            raw=record.p_signal[:,indices]
            centers,qa=detect(raw,record.fs,p)
            features=extract_array(raw,centers,record.fs,p)
            refs,edge=annotation_rows(ann.sample,ann.symbol,len(raw),record.fs,p)
            labels,matched=match_labels(refs,centers,record.fs,p['matching_tolerance_seconds'])
            # One available lead is kept as one; no invented missing channel.
            atomic_npz(folder/'features.npz',features=features,centers=centers,labels=labels,matched_reference=matched)
            pd.DataFrame(refs,columns=['center_sample','beat_class','symbol']).to_csv(folder/'reference.csv',index=False)
            write_json(folder/'metadata.json',{**row,'detector':qa,'eligible_reference_excluded_at_edge':edge,
                'candidate_known_labeled':int(np.sum(labels>=0)),'candidate_unmatched':int(np.sum(labels==-2)),
                'candidate_unknown_Q':int(np.sum(labels==-1)),'reference_anchor_training':False})
            complete(folder,current,['features.npz','reference.csv','metadata.json'])
        names.append(f'{name}/completed.json')
        print(f'{dataset} fitur {index}/{len(saved["records"])}: {name}',flush=True)
    write_json(destination/'schema.json',{'names':feature_names(p,True),'source_leads_retained':True,
        'lead_pooling':'mean probabilities, not mean waveforms','RR':'detected neighbors, offline before/after',
        'amplitude':'per-channel record RMS; native unit scale cancels; not origin features'})
    complete(destination,identity,[*names,'schema.json'])


def load_bundle(study,dataset='incart'):
    p=study.protocol;source=ROOT/p['incart_directory' if dataset=='incart' else 'external_directory']
    saved=read_json(study.run/'data/incart.json' if dataset=='incart' else study.run/'external_source.json')
    verify_source(source,saved)
    folder=study.run/'features'/dataset
    identity={'experiment_hash':study.signature,'dataset':dataset,'source':saved,'feature_names':feature_names(p,True)}
    if not verified(folder,identity):
        raise RuntimeError(f'Ekstraksi {dataset} belum lengkap')
    blocks=[]
    for row in saved['records']:
        sub=folder/row['record'];verified(sub,{'experiment_hash':study.signature,'source':row,'dataset':dataset})
        with np.load(sub/'features.npz') as a:
            block={key:a[key].copy() for key in ['features','centers','labels','matched_reference']}
        block.update(record=row['record'],group=row['group'],
            reference=list(pd.read_csv(sub/'reference.csv').itertuples(index=False,name=None)),
            metadata=read_json(sub/'metadata.json'))
        if block['features'].shape!=(len(block['centers']),len(row['selected_leads']),len(feature_names(p,True))):
            raise RuntimeError('Feature shape/order changed')
        blocks.append(block)
    return blocks,saved
