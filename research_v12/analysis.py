"""Recompute scores from saved predictions; no model loading or waveform reads."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from src.artifacts import read_json,write_json,sha256_file
from .common import verified,complete
from .data import load_bundle
from .experiment import completed_choices,deployment_choice,require_deployment
from .metrics import evaluate,group_bootstrap


def saved_probabilities(folder,blocks):
    frame=pd.read_csv(folder/'predictions.csv',dtype={'record':str,'group':str})
    if len(frame)!=sum(len(b['centers']) for b in blocks):
        raise RuntimeError('Saved candidate count mismatch')
    probs=[];offset=0
    for b in blocks:
        row=frame.iloc[offset:offset+len(b['centers'])];offset+=len(row)
        if not row.record.eq(b['record']).all() or not row.group.eq(b['group']).all():
            raise RuntimeError('Record/group identity mismatch')
        for key,values in [('center_sample',b['centers']),('reference_class',b['labels']),('matched_reference_sample',b['matched_reference'])]:
            if not np.array_equal(row[key].to_numpy(),values):
                raise RuntimeError('Saved reference/center identity mismatch')
        prob=row[['prob_normal_N','prob_PVC','prob_other']].to_numpy()
        if not np.isfinite(prob).all() or (prob<0).any() or (prob>1).any():
            raise RuntimeError('Invalid saved probability')
        probs.append(prob)
    return probs


def compare_metric(actual,expected):
    if isinstance(actual,dict):
        if actual.keys()!=expected.keys():raise RuntimeError('Metric schema differs')
        for key,value in actual.items():compare_metric(value,expected[key])
    elif isinstance(actual,list):
        if len(actual)!=len(expected):raise RuntimeError('Metric length differs')
        for a,b in zip(actual,expected):compare_metric(a,b)
    elif isinstance(actual,(int,float)):
        if expected is None or not np.isclose(actual,expected,atol=1e-10,rtol=1e-9):
            raise RuntimeError('Saved metric differs from predictions/reference')
    elif actual!=expected:
        raise RuntimeError('Saved metric differs')


def analyze(study):
    rows=completed_choices(study);blocks,saved=load_bundle(study)
    report=study.run/'analysis';report.mkdir(parents=True,exist_ok=True)
    audit_cases=0;pooled=[];all_record_counts={};oof_frames={}
    for arm in study.protocol['arms']:
        records=[];frames=[];seen=set()
        for split in saved['folds']:
            base=study.run/'cases'/arm/f'fold_{split["fold"]}'
            for name in ['inner_C1','inner_C2','inner_C3','outer']:
                folder=base/name;verified(folder)
                result=read_json(folder/'metrics.json');identity=result['identity']
                expected_groups=split['outer_validation_groups'] if name=='outer' else split['inner_validation_groups']
                train_groups=split['outer_training_groups'] if name=='outer' else split['inner_training_groups']
                if identity['experiment_hash']!=study.signature or identity['evaluation_groups']!=expected_groups or identity['training_groups']!=train_groups:
                    raise RuntimeError('Case experiment/split identity differs')
                selected=[b for b in blocks if b['group'] in set(expected_groups)]
                prob=saved_probabilities(folder,selected)
                for score in result['scores']:
                    actual,counts=evaluate(selected,prob,score['threshold'],study.protocol['external_gate']['normal_N_threshold'])
                    compare_metric(actual,score['metrics']);compare_metric(counts,score['per_record'])
                audit_cases+=1
                if name=='outer':
                    if seen&set(expected_groups):raise RuntimeError('Repeated OOF group')
                    seen|=set(expected_groups);records.extend(result['scores'][0]['per_record'])
                    frame=pd.read_csv(folder/'predictions.csv',dtype={'record':str,'group':str})
                    frame['fold']=split['fold'];frame['threshold']=result['scores'][0]['threshold'];frames.append(frame)
        if seen!=set(saved['groups']):raise RuntimeError('Incomplete OOF groups')
        frame=pd.concat(frames,ignore_index=True);frame.to_csv(report/f'oof_{arm}.csv',index=False)
        pd.DataFrame(records).to_csv(report/f'record_counts_{arm}.csv',index=False)
        counts={k:sum(r[k] for r in records) for k in ['tp','fp','fn']}
        from .labels import count_metrics
        interval=group_bootstrap(records,seed=study.protocol['seed'])
        pooled.append({'arm':arm,**counts,**count_metrics(counts),'conditional_group_bootstrap_95CI':interval})
        all_record_counts[arm]=records;oof_frames[arm]=frame
    table=pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in rows])
    table.to_csv(report/'outer_scores.csv',index=False)
    summary=table.groupby('arm')[['precision','recall','f1','matched_known_macro_f1','missed_reference_PVC']].agg(['mean','std'])
    summary.to_csv(report/'summary.csv')
    choice=deployment_choice(rows,study.protocol['arms'])
    write_json(report/'audit.json',{'experiment_hash':study.signature,'fit_cases_verified':audit_cases,
        'expected_CV_cases':40,'source_groups':32,'all_saved_evaluation_metrics_reproduced':True,
        'one_outer_prediction_per_candidate_per_arm':True,'pooled_PVC':pooled,'deployment_choice':choice,
        'model_loaded':False,'waveforms_read':False,'origin_score':None,
        'CI_scope':'source-group bootstrap conditional on fixed OOF predictions; not retraining or model-selection uncertainty',
        'INCART_role':'development after historical detector review; not a fresh external cohort'})
    plot(study,table,oof_frames,report)
    names=['audit.json','summary.csv','outer_scores.csv','beat_identity_overview.png','PVC_precision_recall.png']
    names += [f'{prefix}_{arm}.csv' for arm in study.protocol['arms'] for prefix in ['oof','record_counts']]
    identity={'experiment_hash':study.signature,'case_manifests':{str(q.relative_to(study.run)):sha256_file(q) for q in sorted((study.run/'cases').glob('*/*/*/completed.json'))}}
    complete(report,identity,names)
    print(summary.to_string(),flush=True)
    print(json.dumps({'pooled_PVC':pooled,'deployment_choice':choice,'cases_verified':audit_cases},indent=2),flush=True)


def plot(study,table,frames,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for arm in study.protocol['arms']:
        rows=table[table.arm.eq(arm)]
        axes[0].plot(rows.fold,rows.f1,marker='o',label=arm)
        axes[1].plot(rows.fold,rows.precision,marker='o',label=arm)
        axes[2].plot(rows.fold,rows.recall,marker='o',label=arm)
    for ax,title in zip(axes,['End-to-end PVC F1','End-to-end PVC precision','End-to-end PVC recall']):
        ax.set(xlabel='Outer source-group fold',ylabel='Score',title=title,ylim=(0,1.02),xticks=range(1,6));ax.legend()
    fig.suptitle('Beat identity: detector misses included; no anatomical origin score');fig.tight_layout()
    fig.savefig(folder/'beat_identity_overview.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    for arm,frame in frames.items():
        known=frame[frame.reference_class.ge(0)]
        precision,recall,_=precision_recall_curve(known.reference_class.eq(1),known.prob_PVC)
        ax.plot(recall,precision,label=arm)
    ax.set(xlabel='Recall conditional on matched candidates',ylabel='Precision',title='Diagnostic PR curve (detector misses excluded here)',xlim=(0,1),ylim=(0,1.02));ax.legend()
    fig.tight_layout();fig.savefig(folder/'PVC_precision_recall.png',dpi=160);plt.close(fig)


def external_analysis(study):
    choice=require_deployment(study)
    folder=study.run/'external_evaluation'
    identity={'experiment_hash':study.signature,'deployment_manifest_sha256':sha256_file(study.run/'deployment/completed.json'),
        'external_feature_manifest_sha256':sha256_file(study.run/'features/mitdb/completed.json')}
    if not verified(folder,identity):raise RuntimeError('External score dahulu')
    blocks,_=load_bundle(study,'mitdb');saved=read_json(folder/'summary.json')
    if saved['choice']!=choice:raise RuntimeError('External score/deployment choice differs')
    prob=saved_probabilities(folder,blocks)
    m,records=evaluate(blocks,prob,choice['threshold'],study.protocol['external_gate']['normal_N_threshold'])
    compare_metric(m,saved['metrics'])
    write_json(folder/'analysis.json',{'all_saved_metrics_reproduced':True,
        'source_group_bootstrap_95CI':group_bootstrap(records,seed=study.protocol['seed']),
        'groups':47,'records':48,'model_loaded':False,'waveforms_read':False,
        'gate_pass':saved['engineering_gate_pass'],'clinical_local_validation':False,'origin_validation':False})
    print(json.dumps({**saved,'analysis':read_json(folder/'analysis.json')},indent=2),flush=True)


def feature_qa(study):
    blocks,saved=load_bundle(study)
    rows=[]
    for b in blocks:
        refs=b['reference'];labels=b['labels'];positive=sum(r[1]==1 for r in refs)
        matched=int(np.sum(labels==1))
        rows.append({'record':b['record'],'group':b['group'],'reference_beats':len(refs),
            'candidates':len(labels),'matched_known':int(np.sum(labels>=0)),
            'unknown_Q':int(np.sum(labels==-1)),'unmatched':int(np.sum(labels==-2)),
            'reference_PVC':positive,'detected_PVC_reference':matched,
            'detection_recall_PVC':matched/positive if positive else None})
    folder=study.run/'features/incart/qa';folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(folder/'per_record.csv',index=False)
    positive=sum(r['reference_PVC'] for r in rows);matched=sum(r['detected_PVC_reference'] for r in rows)
    result={'records':len(rows),'source_groups':len(saved['groups']),'reference_PVC':positive,
        'detected_PVC_reference':matched,'detection_recall_PVC':matched/positive if positive else None,
        'PVC_classification_score_available':False,'model_loaded':False,'waveforms_read':False,
        'note':'This measures candidate detection only; train/evaluate beat identity next.'}
    write_json(folder/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)
