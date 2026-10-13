"""Audit saved features/predictions; no fitting, model loading or raw ECG reads."""
import argparse
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from src.artifacts import ROOT,read_json,write_json,sha256_file
from research_v12.common import context,verified,DEFAULT_RUN
from research_v12.data import load_bundle
from research_v12.analysis import saved_probabilities,compare_metric
from research_v12.experiment import completed_choices,require_deployment
from research_v12.metrics import evaluate,group_bootstrap,pvc_decisions
from research_v12.labels import match_labels,count_metrics


def check_training(folder,blocks,identity,p):
    sample=pd.read_csv(folder/'training_samples.csv',dtype={'record':str,'group':str})
    if set(sample.group)!=set(identity['training_groups']):
        raise RuntimeError('Training sample groups differ')
    if set(sample.group)&set(identity['evaluation_groups']):
        raise RuntimeError('Training/evaluation leakage')
    if sample.duplicated(['record','center_sample','lead_index']).any():
        raise RuntimeError('Training lead row repeated')
    lookup={b['record']:b for b in blocks}
    for record,part in sample.groupby('record'):
        b=lookup[record];centers=b['centers'];pos=np.searchsorted(centers,part.center_sample.to_numpy())
        if (pos>=len(centers)).any() or not np.array_equal(centers[pos],part.center_sample):
            raise RuntimeError('Training center is not a saved candidate')
        if not np.array_equal(b['labels'][pos],part['class']) or (part['class']<0).any():
            raise RuntimeError('Unknown/unmatched or wrong training label')
        if not part.group.eq(b['group']).all() or not part.lead_index.between(0,b['features'].shape[1]-1).all():
            raise RuntimeError('Wrong training source/lead')
    beats=sample.drop_duplicates(['record','center_sample'])
    caps=beats.groupby(['group','class']).size()
    if caps.max()>p['training_cap_per_group_class']:
        raise RuntimeError('Training cap exceeded')
    group_weight=sample.groupby('group').weight_raw.sum().to_numpy()
    if not np.allclose(group_weight,1/len(group_weight),atol=1e-12):
        raise RuntimeError('Unequal source-group weight')
    for _,part in sample.groupby('group'):
        totals=part.groupby('class').weight_raw.sum().to_numpy()
        if not np.allclose(totals,totals[0],atol=1e-12):
            raise RuntimeError('Unequal present-class weight within group')
    return {'lead_rows':len(sample),'unique_beats':len(beats),'maximum_group_class_beats':int(caps.max())}


def paired_diagnostics(blocks,prob,threshold):
    rows=[];fp_class={};fp_symbol={};offsets=[]
    for b,pr in zip(blocks,prob):
        pvc=pvc_decisions(pr,threshold);labels=b['labels'];refs={int(r[0]):r for r in b['reference']}
        reference=np.array(sorted(refs),dtype=np.int64)
        false=pvc&(labels!=1)&(labels!=-1)
        for label in np.unique(labels[false]):
            key={-2:'unmatched_candidate',0:'normal_N',2:'other_known_beat'}[int(label)]
            fp_class[key]=fp_class.get(key,0)+int(np.sum(false&(labels==label)))
        for i in np.flatnonzero(false&(labels>=0)):
            symbol=refs[int(b['matched_reference'][i])][2]
            fp_symbol[symbol]=fp_symbol.get(symbol,0)+1
        events=b['centers'][false&(labels==-2)]
        if len(events) and len(reference):
            j=np.searchsorted(reference,events)
            distance=np.minimum(abs(events-reference[np.clip(j,0,len(reference)-1)]),
                abs(events-reference[np.clip(j-1,0,len(reference)-1)]))*1000/b['metadata']['fs']
            offsets.extend(distance.tolist())
            quantiles=np.quantile(distance,[.1,.5,.9]).tolist()
        else:quantiles=None
        rows.append({'record':b['record'],'group':b['group'],'selected_leads':'/'.join(b['metadata']['selected_leads']),
            'candidates':len(labels),'references':len(reference),'unmatched_candidates':int(np.sum(labels==-2)),
            'unmatched_PVC_predictions':len(events),'unmatched_FP_nearest_reference_ms_p10_p50_p90':quantiles})
    return {'false_positives_by_reference_class':fp_class,'false_positives_known_reference_symbols':fp_symbol,
        'unmatched_FP_nearest_reference_ms_p10_p50_p90':np.quantile(offsets,[.1,.5,.9]).tolist() if offsets else None},rows


def audit(study,out):
    out.mkdir(parents=True,exist_ok=True)
    choice=require_deployment(study);choices=completed_choices(study)
    blocks,saved=load_bundle(study);cases=[];outer={arm:[] for arm in study.protocol['arms']}
    seen=set();all_groups={b['group'] for b in blocks}
    for split in saved['folds']:
        a,b,c=map(set,[split['inner_training_groups'],split['inner_validation_groups'],split['outer_validation_groups']])
        if a&b or (a|b)&c or a|b|c!=all_groups or a|b!=set(split['outer_training_groups']) or seen&c:
            raise RuntimeError('Source groups overlap or coverage differs')
        seen|=c
    if seen!=all_groups or len(all_groups)!=32:
        raise RuntimeError('Expected complete outer coverage of 32 INCART groups')
    for b in blocks:
        y,matched=match_labels(b['reference'],b['centers'],b['metadata']['fs'],study.protocol['matching_tolerance_seconds'])
        if not np.array_equal(y,b['labels']) or not np.array_equal(matched,b['matched_reference']):
            raise RuntimeError('Stored reference matching differs')
    for arm in study.protocol['arms']:
        for split in saved['folds']:
            folder=study.run/'cases'/arm/f'fold_{split["fold"]}'
            for name in ['inner_C1','inner_C2','inner_C3','outer']:
                case=folder/name;manifest=verified(case);result=read_json(case/'metrics.json');identity=result['identity']
                selected=[b for b in blocks if b['group'] in identity['evaluation_groups']]
                prob=saved_probabilities(case,selected)
                for score in result['scores']:
                    metric,records=evaluate(selected,prob,score['threshold'],study.protocol['external_gate']['normal_N_threshold'])
                    compare_metric(metric,score['metrics']);compare_metric(records,score['per_record'])
                    if name=='outer':outer[arm].extend(records)
                sample=check_training(case,blocks,identity,study.protocol)
                cases.append({'arm':arm,'fold':split['fold'],'case':name,'evaluation_scores_reproduced':len(result['scores']),**sample})
            # Deterministic samples must match across C and input arms.
            paired=study.run/'cases'/study.protocol['arms'][0]/f'fold_{split["fold"]}'
            for name in ['inner_C1','inner_C2','inner_C3','outer']:
                reference_name='inner_C1' if name.startswith('inner') else 'outer'
                if sha256_file(folder/name/'training_samples.csv')!=sha256_file(paired/reference_name/'training_samples.csv'):
                    raise RuntimeError('Unpaired training samples across C/arms')
    saved_analysis=read_json(study.run/'analysis/audit.json');verified(study.run/'analysis')
    pooled=[]
    for arm in study.protocol['arms']:
        counts={k:sum(r[k] for r in outer[arm]) for k in ['tp','fp','fn']}
        summary={'arm':arm,**counts,**count_metrics(counts),
            'conditional_group_bootstrap_95CI':group_bootstrap(outer[arm],seed=study.protocol['seed'])}
        compare_metric(summary,next(r for r in saved_analysis['pooled_PVC'] if r['arm']==arm));pooled.append(summary)
        oof=pd.read_csv(study.run/'analysis'/f'oof_{arm}.csv',dtype={'record':str,'group':str})
        if oof.duplicated(['record','center_sample']).any() or len(oof)!=sum(len(b['labels']) for b in blocks):
            raise RuntimeError('OOF candidate coverage/uniqueness differs')
        frames=[]
        for split in saved['folds']:
            base=study.run/'cases'/arm/f'fold_{split["fold"]}'
            f=pd.read_csv(base/'outer/predictions.csv',dtype={'record':str,'group':str})
            f['fold']=split['fold'];f['threshold']=read_json(base/'selection.json')['threshold'];frames.append(f)
        pd.testing.assert_frame_equal(oof,pd.concat(frames,ignore_index=True),check_exact=False,atol=1e-12,rtol=1e-9)
    external,external_saved=load_bundle(study,'mitdb')
    if len(external)!=48 or len({b['group'] for b in external})!=47:
        raise RuntimeError('Expected all 48 records / 47 external subject groups')
    subjects={b['record']:b['group'] for b in external}
    if subjects['201']!=subjects['202']:
        raise RuntimeError('MIT-BIH 201/202 must stay the same source group')
    ef=study.run/'external_evaluation'
    verified(ef,{'experiment_hash':study.signature,'deployment_manifest_sha256':sha256_file(study.run/'deployment/completed.json'),
        'external_feature_manifest_sha256':sha256_file(study.run/'features/mitdb/completed.json')})
    external_summary=read_json(ef/'summary.json');prob=saved_probabilities(ef,external)
    m,records=evaluate(external,prob,choice['threshold'],study.protocol['external_gate']['normal_N_threshold'])
    compare_metric(m,external_summary['metrics'])
    gate=study.protocol['external_gate']
    checks={'PVC_precision':m['precision']>=gate['minimum_PVC_precision'],
        'PVC_recall':m['recall']>=gate['minimum_PVC_recall'],'PVC_F1':m['f1']>=gate['minimum_PVC_F1'],
        'normal_N_precision':m['normal_N_high_score_precision']>=gate['minimum_normal_N_precision'],
        'normal_N_reference_recall':m['normal_N_high_score_reference_recall']>=gate['minimum_normal_N_reference_recall'],
        'normal_N_selection_nonempty':m['normal_N_high_score_selected']>0}
    if checks!=external_summary['gate_checks'] or all(checks.values())!=external_summary['engineering_gate_pass']:
        raise RuntimeError('External gate differs from frozen rules')
    compare_metric(group_bootstrap(records,seed=study.protocol['seed']),read_json(ef/'analysis.json')['source_group_bootstrap_95CI'])
    frame=pd.read_csv(ef/'per_record.csv',dtype={'record':str,'group':str})
    pd.testing.assert_frame_equal(frame,pd.DataFrame(records),check_dtype=False)
    diagnostic,details=paired_diagnostics(external,prob,choice['threshold'])
    reference=m['tp']+m['fn'];detected=reference-m['missed_reference_PVC']
    m['detected_reference_PVC']=detected;m['detector_recall_PVC']=detected/reference
    m['classifier_FN_among_detected_PVC']=m['fn']-m['missed_reference_PVC']
    m['maximum_recall_with_current_candidates']=detected/reference
    frame['detection_recall_PVC']=1-frame.missed_reference_PVC/frame.positive_reference.replace(0,np.nan)
    frame['f1']=2*frame.tp/(2*frame.tp+frame.fp+frame.fn).replace(0,np.nan)
    frame['false_negatives_after_detection']=frame.fn-frame.missed_reference_PVC
    frame.merge(pd.DataFrame(details).drop(columns=['group','candidates','references']),on='record',validate='one_to_one').to_csv(out/'external_failure_by_record.csv',index=False)
    pd.DataFrame(cases).to_csv(out/'case_audit.csv',index=False)
    result={'experiment_hash':study.signature,'fit_cases_verified':len(cases),'evaluation_scores_reproduced':sum(c['evaluation_scores_reproduced'] for c in cases),
        'source_groups_disjoint':True,'training_samples_known_matched_only':True,'training_caps_and_group_weights_verified':True,
        'training_samples_paired_between_C_and_arms':True,'OOF_coverage_and_predictions_verified':True,
        'pooled_development_PVC':pooled,'mean_fold_PVC_F1':choice['development_mean_PVC_F1'],
        'external_records':len(external),'external_groups':len(external_saved['groups']),'external_metrics':m,
        'external_failure_diagnostics':diagnostic,'deployment_choice':choice,'external_gate_pass':external_summary['engineering_gate_pass'],
        'local_export_completed':(study.run/'local_candidates/completed.json').exists(),
        'origin_training':False,'new_model_fits':0,'model_loaded':False,'waveforms_read':False,'model_inference':False,
        'MIT_BIH_status_after_review':'Observed evaluation source; cannot claim pristine unseen evaluation when used for future method design.'}
    write_json(out/'audit.json',result);plot(result,out)
    print(json.dumps({k:v for k,v in result.items() if k not in ['deployment_choice','pooled_development_PVC']},indent=2),flush=True)


def plot(result,out):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.4))
    scores=[*result['mean_fold_PVC_F1'].values(),result['external_metrics']['f1']]
    axes[0].bar(['INCART\nMorphology','INCART\nMorphology\n+ RR','MIT-BIH\nDeployment'],scores,color=['#6c8ebf','#44a583','#c95353'])
    axes[0].set(title='PVC F1: mean CV / external pooled',ylim=(0,1),ylabel='F1')
    for i,v in enumerate(scores):axes[0].text(i,v+.02,f'{v:.3f}',ha='center')
    counts=result['external_failure_diagnostics']['false_positives_by_reference_class']
    bars=axes[1].bar(['Unmatched\ncandidate','Normal N','Other beat'],[counts.get(k,0) for k in ['unmatched_candidate','normal_N','other_known_beat']],color=['#c95353','#d89a40','#6c8ebf'])
    axes[1].bar_label(bars,padding=4,fontsize=9)
    axes[1].set(title=f'MIT-BIH: {sum(counts.values()):,} false positives',ylabel='Count',ylim=(0,max(counts.values())*1.16))
    m=result['external_metrics'];values=[m['missed_reference_PVC'],m['classifier_FN_among_detected_PVC']]
    bars=axes[2].bar(['Missed by\ndetector','Missed after\ndetection'],values,color=['#c95353','#d89a40'])
    axes[2].bar_label(bars,padding=4,fontsize=9)
    axes[2].set(title=f'MIT-BIH: {sum(values):,} false negatives',ylabel='Count',ylim=(0,max(values)*1.16))
    fig.suptitle('V12 beat identity audit — these are not anatomical origin scores')
    fig.tight_layout();fig.savefig(out/'v12_failure_overview.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',default=DEFAULT_RUN)
    args=parser.parse_args();study=context(args.run_id)
    audit(study,study.run/'analysis_20261007_complete')


if __name__=='__main__':main()
