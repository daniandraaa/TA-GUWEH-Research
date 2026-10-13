"""Manual grouped beat-classifier fits; never train an anatomical origin model."""
from collections import Counter
from pathlib import Path
import hashlib
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.artifacts import read_json,write_json,sha256_file,canonical_hash
from .common import complete,verified,disk_check
from .data import load_bundle
from .features import feature_names
from .metrics import evaluate


def training_rows(blocks,p,arm):
    if arm not in p['arms']:
        raise ValueError('Unknown input arm')
    width=len(feature_names(p,arm=='morphology_rr'))
    rows=[];x=[];y=[];weights=[]
    groups=sorted({b['group'] for b in blocks})
    for group in groups:
        members=[(b,i) for b in blocks if b['group']==group for i,label in enumerate(b['labels']) if label>=0]
        present=sorted({int(b['labels'][i]) for b,i in members})
        if not present:
            raise RuntimeError(f'No matched annotated training beat in group {group}')
        for label in present:
            candidates=[(b,i) for b,i in members if b['labels'][i]==label]
            salt=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)
            rng=np.random.default_rng(p['seed']+salt+label)
            take=min(p['training_cap_per_group_class'],len(candidates))
            chosen=np.sort(rng.choice(len(candidates),size=take,replace=False))
            beat_weight=1/(len(groups)*len(present)*take)
            for j in chosen:
                b,i=candidates[j];n_leads=b['features'].shape[1]
                for lead in range(n_leads):
                    x.append(b['features'][i,lead,:width]);y.append(label)
                    weights.append(beat_weight/n_leads)
                    rows.append({'record':b['record'],'group':group,'center_sample':int(b['centers'][i]),
                        'class':label,'lead_index':lead,'weight_raw':beat_weight/n_leads})
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=int);weights=np.asarray(weights)
    if set(y)!={0,1,2}:
        raise RuntimeError('All three classes required in matched training candidates')
    # Mean weight one keeps C comparable to an ordinary sklearn sample weight.
    weights*=len(weights)/weights.sum()
    return x,y,weights,pd.DataFrame(rows)


def fit_pipeline(x,y,weights,C,p):
    model=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scaler',StandardScaler()),('head',LogisticRegression(C=float(C),solver='lbfgs',
            max_iter=p['maximum_iterations'],tol=1e-4,random_state=p['seed']))])
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,y,head__sample_weight=weights)
    if model.classes_.tolist()!=[0,1,2]:
        raise RuntimeError('Unexpected classifier class order')
    return model


def probabilities(model,blocks,p,arm):
    width=len(feature_names(p,arm=='morphology_rr'));result=[]
    with threadpool_limits(limits=2):
        for b in blocks:
            n,leads,_=b['features'].shape
            if not n:
                result.append(np.empty((0,3)));continue
            # Each source beat is scored once after available-lead soft voting.
            pred=model.predict_proba(b['features'][:,:,:width].reshape(n*leads,width))
            result.append(pred.reshape(n,leads,3).mean(axis=1))
    return result


def prediction_frame(blocks,probs):
    rows=[]
    for b,prob in zip(blocks,probs):
        frame=pd.DataFrame({'record':b['record'],'group':b['group'],'center_sample':b['centers'],
            'matched_reference_sample':b['matched_reference'],'reference_class':b['labels'],
            'prob_normal_N':prob[:,0],'prob_PVC':prob[:,1],'prob_other':prob[:,2]})
        rows.append(frame)
    return pd.concat(rows,ignore_index=True)


def case_identity(study,arm,train_groups,eval_groups,C,thresholds,role):
    return {'experiment_hash':study.signature,'arm':arm,'training_groups':sorted(train_groups),
        'evaluation_groups':sorted(eval_groups),'C':float(C),'thresholds':list(map(float,thresholds)),
        'role':role,'features_manifest_sha256':sha256_file(study.run/'features/incart/completed.json')}


def fit_case(study,blocks,arm,train_groups,eval_groups,C,thresholds,folder,role):
    if set(train_groups)&set(eval_groups):
        raise ValueError('Training/evaluation source groups overlap')
    identity=case_identity(study,arm,train_groups,eval_groups,C,thresholds,role)
    if verified(folder,identity):
        print(f'Fit lengkap, dilewati: {folder.relative_to(study.run)}',flush=True)
        return read_json(folder/'metrics.json')
    if (study.run/'external_source.json').exists():
        raise RuntimeError('Evaluasi eksternal sudah dibuka; fitting INCART baru pada run ini dibekukan')
    disk_check(study,16*1024**2)
    training=[b for b in blocks if b['group'] in set(train_groups)]
    evaluation=[b for b in blocks if b['group'] in set(eval_groups)]
    x,y,w,samples=training_rows(training,study.protocol,arm)
    print(f'FIT {role} {arm} C={C:g}: {len(train_groups)} grup, {len(y)} row lead; {len(eval_groups)} grup evaluasi',flush=True)
    model=fit_pipeline(x,y,w,C,study.protocol)
    train_prob=probabilities(model,training,study.protocol,arm)
    train_metrics,_=evaluate(training,train_prob,thresholds[0],study.protocol['external_gate']['normal_N_threshold'])
    prob=probabilities(model,evaluation,study.protocol,arm)
    scores=[]
    for threshold in thresholds:
        m,records=evaluate(evaluation,prob,threshold,study.protocol['external_gate']['normal_N_threshold'])
        scores.append({'C':float(C),'threshold':float(threshold),'metrics':m,'per_record':records})
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,folder/'pipeline.joblib.tmp');(folder/'pipeline.joblib.tmp').replace(folder/'pipeline.joblib')
    samples.to_csv(folder/'training_samples.csv',index=False)
    prediction_frame(evaluation,prob).to_csv(folder/'predictions.csv',index=False)
    result={'identity':identity,'scores':scores,'train_metrics_at_first_threshold':train_metrics,
        'training_lead_rows':len(y),'unique_training_beats':len(samples[['record','center_sample']].drop_duplicates()),
        'scaler_fit_groups':sorted(train_groups),'n_iter':model.named_steps['head'].n_iter_.tolist(),
        'probability_calibrated':False,'origin_training':False}
    write_json(folder/'metrics.json',result)
    complete(folder,identity,['pipeline.joblib','training_samples.csv','predictions.csv','metrics.json'])
    return result


def choose_score(scores):
    if not scores:
        raise ValueError('No inner scores')
    return max(scores,key=lambda s:(s['metrics']['f1'],s['metrics']['precision'],-s['C'],s['threshold']))


def run(study,folds,arms):
    blocks,saved=load_bundle(study)
    for fold in folds:
        split=next(r for r in saved['folds'] if r['fold']==fold)
        for arm in arms:
            scores=[]
            for index,C in enumerate(study.protocol['inverse_regularization_C'],1):
                result=fit_case(study,blocks,arm,split['inner_training_groups'],split['inner_validation_groups'],C,
                    study.protocol['pvc_thresholds'],study.run/'cases'/arm/f'fold_{fold}'/f'inner_C{index}','inner')
                scores.extend(result['scores'])
            chosen=choose_score(scores)
            folder=study.run/'cases'/arm/f'fold_{fold}'
            write_json(folder/'selection.json',{'arm':arm,'fold':fold,'C':chosen['C'],'threshold':chosen['threshold'],
                'source':'inner validation groups only','inner_metrics':chosen['metrics'],'experiment_hash':study.signature})
            outer=fit_case(study,blocks,arm,split['outer_training_groups'],split['outer_validation_groups'],
                chosen['C'],[chosen['threshold']],folder/'outer','outer')
            m=outer['scores'][0]['metrics']
            print(f'Outer fold {fold} {arm}: PVC P/R/F1={m["precision"]:.4f}/{m["recall"]:.4f}/{m["f1"]:.4f}; '
                f'missed detector PVC={m["missed_reference_PVC"]}',flush=True)


def completed_choices(study):
    if not verified(study.run/'data',{'experiment_hash':study.signature}):
        raise RuntimeError('Prepare dan split INCART lengkap diperlukan')
    saved=read_json(study.run/'data/incart.json')
    rows=[]
    for arm in study.protocol['arms']:
        for fold in range(1,6):
            folder=study.run/'cases'/arm/f'fold_{fold}'
            split=next(s for s in saved['folds'] if s['fold']==fold)
            if not (folder/'outer/completed.json').exists():
                raise RuntimeError('Lima fold kedua arm harus lengkap sebelum seleksi deployment')
            selected=read_json(folder/'selection.json');outer=read_json(folder/'outer/metrics.json')
            identity=case_identity(study,arm,split['outer_training_groups'],split['outer_validation_groups'],
                selected['C'],[selected['threshold']],'outer')
            verified(folder/'outer',identity)
            if selected['experiment_hash']!=study.signature or selected['arm']!=arm or selected['fold']!=fold or outer['identity']!=identity:
                raise RuntimeError('Selection/outer identity inconsistent')
            for index,C in enumerate(study.protocol['inverse_regularization_C'],1):
                inner_identity=case_identity(study,arm,split['inner_training_groups'],split['inner_validation_groups'],
                    C,study.protocol['pvc_thresholds'],'inner')
                if not verified(folder/f'inner_C{index}',inner_identity):
                    raise RuntimeError('Inner case incomplete')
                if read_json(folder/f'inner_C{index}/metrics.json')['identity']!=inner_identity:
                    raise RuntimeError('Inner metric identity differs')
            all_scores=[s for index in range(1,len(study.protocol['inverse_regularization_C'])+1)
                for s in read_json(folder/f'inner_C{index}/metrics.json')['scores']]
            expected=choose_score(all_scores)
            if (expected['C'],expected['threshold'])!=(selected['C'],selected['threshold']):
                raise RuntimeError('Cannot reproduce inner selection')
            rows.append({'arm':arm,'fold':fold,'C':selected['C'],'threshold':selected['threshold'],
                **outer['scores'][0]['metrics']})
    return rows


def deployment_choice(rows,arms):
    mean={arm:float(np.mean([r['f1'] for r in rows if r['arm']==arm])) for arm in arms}
    arm=max(arms,key=lambda a:(mean[a],-arms.index(a)))
    pairs=Counter((r['C'],r['threshold']) for r in rows if r['arm']==arm)
    C,threshold=max(pairs,key=lambda pair:(pairs[pair],-pair[0],pair[1]))
    return {'arm':arm,'C':float(C),'threshold':float(threshold),'development_mean_PVC_F1':mean,
        'selection_uses_external_data':False,'score_probability_calibrated':False}


def deployment_identity(study):
    rows=completed_choices(study);choice=deployment_choice(rows,study.protocol['arms'])
    return {'experiment_hash':study.signature,'choice':choice,
        'cases_manifest':{str(p.relative_to(study.run)):sha256_file(p) for p in sorted((study.run/'cases').glob('*/*/*/completed.json'))},
        'feature_manifest_sha256':sha256_file(study.run/'features/incart/completed.json')}


def require_deployment(study):
    identity=deployment_identity(study);folder=study.run/'deployment'
    if not verified(folder,identity):
        raise RuntimeError('Deploy detector dari INCART dahulu; MIT-BIH belum boleh dibuka')
    choice=read_json(folder/'choice.json')
    if any(choice.get(k)!=v for k,v in identity['choice'].items()) or choice.get('experiment_hash')!=study.signature:
        raise RuntimeError('Deployment choice differs from frozen INCART selection')
    return choice


def deploy(study):
    identity=deployment_identity(study);choice=identity['choice']
    blocks,saved=load_bundle(study);folder=study.run/'deployment'
    if verified(folder,identity):
        print('Detector deployment lengkap dan terverifikasi; dilewati.',flush=True);return
    if (study.run/'external_source.json').exists():
        raise RuntimeError('Evaluasi eksternal sudah dibuka; deployment ulang pada run ini dibekukan')
    disk_check(study,16*1024**2)
    x,y,w,samples=training_rows(blocks,study.protocol,choice['arm'])
    model=fit_pipeline(x,y,w,choice['C'],study.protocol)
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,folder/'pipeline.joblib.tmp');(folder/'pipeline.joblib.tmp').replace(folder/'pipeline.joblib')
    samples.to_csv(folder/'training_samples.csv',index=False)
    write_json(folder/'choice.json',{**choice,'training_groups':saved['groups'],'training_lead_rows':len(y),
        'experiment_hash':study.signature,'anatomical_origin_model':False})
    complete(folder,identity,['pipeline.joblib','training_samples.csv','choice.json'])
    print('Satu model identitas beat siap untuk evaluasi MIT-BIH. Ini bukan model final asal anatomi.',flush=True)


def external_score(study):
    choice=require_deployment(study)
    blocks,saved=load_bundle(study,'mitdb');folder=study.run/'external_evaluation'
    identity={'experiment_hash':study.signature,'deployment_manifest_sha256':sha256_file(study.run/'deployment/completed.json'),
        'external_feature_manifest_sha256':sha256_file(study.run/'features/mitdb/completed.json')}
    if verified(folder,identity):
        print('Evaluasi eksternal lengkap; tidak memilih threshold ulang.',flush=True);return
    model=joblib.load(study.run/'deployment/pipeline.joblib')
    prob=probabilities(model,blocks,study.protocol,choice['arm'])
    m,records=evaluate(blocks,prob,choice['threshold'],study.protocol['external_gate']['normal_N_threshold'])
    gate=study.protocol['external_gate']
    checks={'PVC_precision':m['precision']>=gate['minimum_PVC_precision'],
        'PVC_recall':m['recall']>=gate['minimum_PVC_recall'],'PVC_F1':m['f1']>=gate['minimum_PVC_F1'],
        'normal_N_precision':m['normal_N_high_score_precision']>=gate['minimum_normal_N_precision'],
        'normal_N_reference_recall':m['normal_N_high_score_reference_recall']>=gate['minimum_normal_N_reference_recall'],
        'normal_N_selection_nonempty':m['normal_N_high_score_selected']>0}
    write_json(folder/'summary.json',{'metrics':m,'gate_checks':checks,'engineering_gate_pass':all(checks.values()),
        'external_groups':len(saved['groups']),'external_records':len(blocks),'threshold_tuned_externally':False,
        'clinical_local_validation':False,'anatomical_origin_validation':False,'choice':choice})
    pd.DataFrame(records).to_csv(folder/'per_record.csv',index=False)
    prediction_frame(blocks,prob).to_csv(folder/'predictions.csv',index=False)
    complete(folder,identity,['summary.json','per_record.csv','predictions.csv'])
    print(f'MIT-BIH: P/R/F1={m["precision"]:.4f}/{m["recall"]:.4f}/{m["f1"]:.4f}; gate {"PASS" if all(checks.values()) else "FAIL"}. Threshold tetap.',flush=True)
