"""Read-only saved predictions/metrics and visualization, no new inference."""
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from src.artifacts import read_json,write_json,sha256_file
from src.evaluation import compute_metrics
from .common import verified


def compare_metrics(actual,saved):
    for name,value in actual.items():
        expected=saved[name]
        if (value is None)!=(expected is None) or (value is not None and abs(value-expected)>1e-10):
            raise RuntimeError(f'Saved metric mismatch: {name}')


def audit_case(study,folder):
    identity=read_json(folder/'identity.json');verified(folder,identity)
    if identity['experiment_hash']!=study.signature:raise RuntimeError('Case belongs to different experiment')
    split=study.folds[identity['fold']-1];inner=identity['phase']=='inner'
    train=sorted(split['gan_fit_ids'] if inner else split['gan_fit_ids']+split['gan_reference_ids'])
    evaluation=sorted(split['gan_reference_ids'] if inner else split['outer_validation_ids'])
    if identity['training_ids']!=train or identity['evaluation_ids']!=evaluation or set(train)&set(evaluation):
        raise RuntimeError('Saved fit split changed')
    if identity['preprocessing_fit_ids']!=train or identity['class_weight_fit_ids']!=train:
        raise RuntimeError('Training-only preprocessing identity changed')
    labels={r['hospital_id']:r['label'] for r in study.records}
    result=read_json(folder/'result.json');frames={};scores={}
    for mode,filename,ids,key in [('evaluation','predictions.csv',evaluation,'metrics'),('training','training_predictions.csv',train,'training_metrics')]:
        f=pd.read_csv(folder/filename,dtype={'hospital_id':str})
        if f.hospital_id.tolist()!=ids or f.true_label.tolist()!=[labels[hid] for hid in ids]:raise RuntimeError('Saved prediction identities changed')
        probability=f[[f'probability_{name}' for name in study.classes]].to_numpy()
        if not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any() or not np.allclose(probability.sum(1),1):
            raise RuntimeError('Invalid saved probabilities')
        metrics=compute_metrics(f.true_label.to_numpy(),probability);compare_metrics(metrics,result[key])
        frames[mode]=f;scores[mode]=metrics
    for key,expected in [('evaluation_f1',scores['evaluation']['f1']),('training_f1',scores['training']['f1']),
                         ('f1_gap',scores['training']['f1']-scores['evaluation']['f1'])]:
        if abs(result[key]-expected)>1e-10:raise RuntimeError('Saved F1/gap mismatch')
    if not result['converged'] or identity['C']!=result['C']:raise RuntimeError('Invalid fit/C identity')
    return identity,result,frames['evaluation']


def audit_choices(study,outer_identity):
    from .experiment import select
    arm=outer_identity['arm'];fold=outer_identity['fold'];path=study.run/'choices'/arm/f'fold_{fold}.json'
    if sha256_file(path)!=outer_identity['choice_manifest']:raise RuntimeError('Outer selection hash changed')
    choice=read_json(path);rows=[]
    if choice['experiment_hash']!=study.signature or choice['arm']!=arm or choice['fold']!=fold:raise RuntimeError('Choice identity changed')
    for i,C in enumerate(study.protocol['inverse_regularization_C'],1):
        folder=study.run/'cases'/arm/f'fold_{fold}'/f'inner_S{i:02}'
        if sha256_file(folder/'completed.json')!=choice['inner_completed_sha256'][f'S{i:02}']:raise RuntimeError('Inner selection source changed')
        row=read_json(folder/'result.json')
        if row['C']!=C:raise RuntimeError('Scenario ordering changed')
        rows.append(row)
    best=select(rows)
    if choice['C']!=best['C'] or choice['inner_f1']!=best['evaluation_f1'] or outer_identity['C']!=best['C']:
        raise RuntimeError('Inner C choice cannot be reproduced')


def macro_f1(labels,predicted):
    matrix=np.bincount(3*np.asarray(labels,dtype=int)+np.asarray(predicted,dtype=int),minlength=9).reshape(3,3)
    denominator=matrix.sum(0)+matrix.sum(1)
    return float(np.divide(2*np.diag(matrix),denominator,out=np.zeros(3),where=denominator>0).mean())


def paired_bootstrap(control,fusion,classes,repeats=2000,seed=118):
    a=control.sort_values('hospital_id').reset_index(drop=True);b=fusion.sort_values('hospital_id').reset_index(drop=True)
    if a.hospital_id.tolist()!=b.hospital_id.tolist() or a.true_label.tolist()!=b.true_label.tolist():raise RuntimeError('Paired OOF identities differ')
    y=a.true_label.to_numpy();pa=a[[f'probability_{name}' for name in classes]].to_numpy().argmax(1)
    pb=b[[f'probability_{name}' for name in classes]].to_numpy().argmax(1)
    groups=[np.flatnonzero(y==label) for label in range(3)]
    if any(not len(group) for group in groups):raise RuntimeError('All classes required in bootstrap')
    rng=np.random.default_rng(seed);delta=[]
    for _ in range(repeats):
        sample=np.concatenate([rng.choice(group,len(group),replace=True) for group in groups])
        delta.append(macro_f1(y[sample],pb[sample])-macro_f1(y[sample],pa[sample]))
    return {'pooled_oof_delta_f1':macro_f1(y,pb)-macro_f1(y,pa),'conditional_patient_bootstrap_ci95':np.quantile(delta,[.025,.975]).tolist(),
        'repeats':repeats,'seed':seed,'class_stratified':True,'scope':'conditional_on_fixed_saved_OOF_predictions_not_model_selection_uncertainty'}


def analyze(study):
    rows=[];pooled=[];predictions=[]
    all_cases=[]
    for folder in sorted((study.run/'cases').glob('*/*/*')):
        if (folder/'completed.json').exists():
            identity,_,_=audit_case(study,folder);all_cases.append(identity)
    for arm in study.protocol['arms']:
        frames=[]
        for fold in study.protocol['folds']:
            folder=study.run/'cases'/arm/f'fold_{fold}'/'outer'
            if not (folder/'completed.json').exists():continue
            identity,result,f=audit_case(study,folder);audit_choices(study,identity)
            probability=f[[f'probability_{name}' for name in study.classes]].to_numpy();metrics=compute_metrics(f.true_label.to_numpy(),probability)
            if any(metrics[k]!=None and abs(metrics[k]-result['metrics'][k])>1e-10 for k in metrics):raise RuntimeError('Saved evaluation metric mismatch')
            if f.hospital_id.tolist()!=sorted(study.folds[fold-1]['outer_validation_ids']):raise RuntimeError('Outer patient IDs changed')
            recalls=recall_score(f.true_label,probability.argmax(1),labels=[0,1,2],average=None,zero_division=0)
            rows.append({'arm':arm,'fold':fold,'C':result['C'],'f1':metrics['f1'],'training_f1':result['training_f1'],'gap':result['f1_gap'],
                **{f'recall_{i}':float(value) for i,value in enumerate(recalls)}})
            frames.append(f.assign(arm=arm,fold=fold))
        if len(frames)==5:
            full=pd.concat(frames).sort_values('hospital_id')
            if len(full)!=230 or not full.hospital_id.is_unique:raise RuntimeError('OOF must cover 230 unique patients')
            probability=full[[f'probability_{name}' for name in study.classes]].to_numpy()
            pooled.append({'arm':arm,'patients':230,**compute_metrics(full.true_label.to_numpy(),probability)})
            predictions.append(full)
    out=study.run/'analysis';out.mkdir(parents=True,exist_ok=True);cases=pd.DataFrame(rows)
    if cases.empty:print('No completed outer fit; no scores available',flush=True);return
    cases.to_csv(out/'outer_scores.csv',index=False)
    summary=cases.groupby('arm',sort=False).agg(completed_folds=('fold','count'),mean_fold_f1=('f1','mean'),sd_fold_f1=('f1','std'),
        mean_training_f1=('training_f1','mean'),mean_gap=('gap','mean'),recall_LVOT=('recall_0','mean'),recall_cusp=('recall_1','mean'),recall_other=('recall_2','mean')).reset_index()
    summary['full_cv_complete']=summary.completed_folds.eq(5);summary.to_csv(out/'summary.csv',index=False)
    if pooled:pd.DataFrame(pooled).to_csv(out/'pooled_oof.csv',index=False)
    if predictions:pd.concat(predictions).to_csv(out/'oof_predictions.csv',index=False)
    primary=[]
    for label in ['random','pretrained']:
        control=cases[cases.arm.eq(label+'_l2')];fusion=cases[cases.arm.eq('fusion_'+label)]
        pairs=control.merge(fusion,on='fold',suffixes=('_control','_fusion'))
        for row in pairs.to_dict('records'):
            primary.append({'comparison':'fusion_'+label+' minus '+label+'_l2','fold':row['fold'],
                'delta_f1':row['f1_fusion']-row['f1_control'],'delta_cusp_recall':row['recall_1_fusion']-row['recall_1_control'],
                'delta_other_recall':row['recall_2_fusion']-row['recall_2_control']})
    if primary:pd.DataFrame(primary).to_csv(out/'paired_fusion_delta.csv',index=False)
    bootstrap={}
    for label in ['random','pretrained']:
        control=next((f for f in predictions if f.arm.iloc[0]==label+'_l2'),None)
        fusion=next((f for f in predictions if f.arm.iloc[0]=='fusion_'+label),None)
        if control is not None and fusion is not None:bootstrap['fusion_'+label+' minus '+label+'_l2']=paired_bootstrap(control,fusion,study.classes)
    if bootstrap:write_json(out/'paired_bootstrap.json',bootstrap)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,4));positions=np.arange(len(summary));ax.bar(positions,summary.mean_fold_f1,color='#397ba6')
    ax.set_xticks(positions,summary.arm,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Mean fold patient macro-F1');ax.set_title('V11 development; incomplete arms remain pilot results')
    fig.tight_layout();fig.savefig(out/'mean_fold_f1.png',dpi=150);plt.close(fig)
    counts=len(all_cases)
    write_json(out/'audit.json',{'experiment_hash':study.signature,'completed_training_cases':counts,'expected_cases':350,'outer_cases':len(cases),
        'full_oof_sets':len(pooled),'all_reconstructed_evaluation_metrics_match':True,'new_training_performed':False,'new_waveform_inference_performed':False,
        'all_reconstructed_training_metrics_match':True,'inner_choices_reproduced':len(cases),
        'test_signals_read':False,'pvc_verified':False,'fully_nested_confirmatory_estimate':False})
    (out/'REPORT.md').write_text('V11 development morphology and L2 heads\n\n'
        'Primary endpoint is mean fold patient macro-F1. Pooled OOF is a separate aggregation. '
        'Single deterministic convex head per fold; repeated identical seeds are not replications. '
        'All 230 patients retained. Features summarize candidate QRS, not verified PVC beats. '
        'LUDB PASS concerns measurement engineering only. Final and historical test remain disabled.\n\n'+summary.to_string(index=False)+'\n')
    print(summary.to_string(index=False),flush=True)
