"""Audit saved V10 artifacts only: no genuine waveform/model inference or fits."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from src.artifacts import ROOT, canonical_hash, read_json, sha256_file, write_json, lock
from src.evaluation import compute_metrics
from research_v6.common import verify_files
from research_v8.data import load_development
from research_v10.analysis import verify_case
from research_v10.assets import checkpoint_path
from research_v10.common import (DEFAULT_RUN, context, case_directory, grid, select_best)
from research_v10.data import fit_feature_scale
from research_v10.features import open_features, require_features
from research_v10.experiment import make_datasets, settings_for

DESTINATION = 'analysis_20261006_complete'


def snapshot_unchanged(path):
    before = read_json(path)
    bad = [name for name,digest in before.items() if not (ROOT/name).is_file() or sha256_file(ROOT/name)!=digest]
    if bad: raise RuntimeError(f'Frozen source/results changed: {bad}')
    return len(before)


def macro_f1(truth, prediction):
    matrix = np.bincount(np.asarray(truth)*3+np.asarray(prediction),minlength=9).reshape(3,3)
    denominator = matrix.sum(0)+matrix.sum(1)
    values = np.divide(2*np.diag(matrix),denominator,out=np.zeros(3),where=denominator>0)
    return float(values.mean())


def paired_patient_bootstrap(truth, random_predictions, pretrained_predictions, repetitions=2000, seed=20261006):
    """Paired stratified patient bootstrap, all head seeds kept together."""
    truth = np.asarray(truth); random_predictions=np.asarray(random_predictions);pretrained_predictions=np.asarray(pretrained_predictions)
    if random_predictions.shape != pretrained_predictions.shape or random_predictions.shape != (3,len(truth)):
        raise ValueError('Exactly three paired head-seed prediction arrays required')
    rng=np.random.default_rng(seed);groups=[np.flatnonzero(truth==c) for c in range(3)];deltas=[]
    for _ in range(repetitions):
        indices=np.concatenate([rng.choice(group,size=len(group),replace=True) for group in groups])
        deltas.append(np.mean([macro_f1(truth[indices],p[indices])-macro_f1(truth[indices],r[indices])
            for r,p in zip(random_predictions,pretrained_predictions)]))
    observed=float(np.mean([macro_f1(truth,p)-macro_f1(truth,r) for r,p in zip(random_predictions,pretrained_predictions)]))
    low,high=np.quantile(deltas,[.025,.975])
    return {'statistic':'mean_pooled_oof_macro_f1_pretrained_minus_random_across_three_fixed_head_seeds',
        'observed_delta':observed,'ci95_lower':float(low),'ci95_upper':float(high),'repetitions':repetitions,
        'seed':seed,'unit':'patient','paired':True,'class_stratified':True,'refitting_performed':False,
        'scope':'conditional_exploratory_uncertainty_of_saved_predictions_not_training_or_method_selection_variance'}


def checkpoint_shapes(study, state, mean, scale, tail_source):
    expected={'feature_mean':(768,),'feature_scale':(768,),'head.weight':(3,768),'head.bias':(3,)}
    if tail_source is not None: expected.update({name:tuple(t.shape) for name,t in tail_source.items()})
    if set(state)!=set(expected): raise RuntimeError('Checkpoint parameter coverage changed')
    for name,tensor in state.items():
        if tuple(tensor.shape)!=expected[name] or not torch.isfinite(tensor).all():
            raise RuntimeError(f'Checkpoint shape/finite mismatch: {name}')
    if not torch.equal(state['feature_mean'],torch.from_numpy(mean)) or not torch.equal(state['feature_scale'],torch.from_numpy(scale)):
        raise RuntimeError('Checkpoint scaler differs from its training patients')
    if tail_source is None: return None
    deltas=[(state[name]-tensor).double() for name,tensor in tail_source.items() if name.startswith('tail.')]
    changed=sum(int(torch.count_nonzero(x)) for x in deltas)
    if not changed: raise RuntimeError('Fine-tuned tail did not change any parameter')
    for name,tensor in tail_source.items():
        if name.startswith('finish_norm.') and not torch.equal(state[name],tensor): raise RuntimeError('Frozen final norm changed')
    return {'tail_values_changed':changed,'tail_max_absolute_change':max(float(x.abs().max()) for x in deltas),
        'tail_l2_change':float(np.sqrt(sum(float((x*x).sum()) for x in deltas)))}


def audit():
    study=context(DEFAULT_RUN);destination=study.run/DESTINATION;destination.mkdir(parents=True,exist_ok=True)
    with lock(study.run/'.v10_stage.lock'):
        snapshot=ROOT/'tmp/v10_completed_before_audit_20261006.json'
        original=study.run/'analysis';original_audit=read_json(original/'audit.json')
        if (original_audit['experiment_hash']!=study.signature or original_audit['cases_verified']!=130
            or original_audit['inner_cases']!=99 or original_audit['outer_cases']!=31
            or original_audit['inner_choices_verified']!=11 or not original_audit['full_probe_benchmark_complete']
            or original_audit['test_signals_read']):raise RuntimeError('Original complete analysis missing or changed')
        smoke=read_json(study.run/'synthetic_smoke_mps.json')
        if smoke['experiment_hash']!=study.signature or smoke['device']!='mps' or not smoke['tail_backward_finite']:
            raise RuntimeError('Require successful manual MPS smoke')
        patients,windows,arrays,feature_hash=open_features(study);feature_folder,feature_manifest=require_features(study)
        if feature_manifest['device']!='mps' or feature_manifest['patients']!=230 or feature_manifest['native_windows']!=2274:
            raise RuntimeError('Require complete manual development feature extraction')
        if sha256_file(feature_folder/'completed.json')!=read_json(study.run/'feature_cache.json')['manifest_sha256']:
            raise RuntimeError('Feature cache linkage changed')
        X,native_table=load_development(study)  # Metadata only; waveform mmap never indexed.
        native=pd.DataFrame({'segment_index':np.arange(len(native_table)),
            'hospital_id':native_table.hospital_id.astype(str),'label':native_table.label})
        pd.testing.assert_frame_equal(windows.reset_index(drop=True),native.reset_index(drop=True),check_dtype=False)
        counts=native.groupby('hospital_id').size()
        if patients.hospital_id.tolist()!=sorted(counts.index) or not np.array_equal(patients.windows.to_numpy(),counts.loc[patients.hospital_id].to_numpy()):
            raise RuntimeError('Patient window pooling metadata changed')
        summaries=[]
        for arm in study.protocol['probe_arms']:
            x=np.asarray(arrays[arm],dtype=np.float64)
            if x.shape!=(230,768) or not np.isfinite(x).all() or len(np.unique(x,axis=0))!=230:
                raise RuntimeError('Feature vectors missing/nonfinite/duplicated')
            centered=x-x.mean(0);singular=np.linalg.svd(centered,compute_uv=False);ratios=singular**2/np.sum(singular**2)
            positive=ratios[ratios>0]
            summaries.append({'arm':arm,'patients':230,'dimensions':768,'unique_rows':230,
                'centered_numerical_rank':int(np.linalg.matrix_rank(centered)),
                'effective_rank':float(np.exp(-np.sum(positive*np.log(positive)))),
                'pc1_variance_fraction':float(ratios[0]),'pc10_variance_fraction':float(ratios[:10].sum()),
                'minimum_dimension_std':float(x.std(0).min()),'median_dimension_std':float(np.median(x.std(0))),
                'mean_raw_feature_norm':float(np.linalg.norm(x,axis=1).mean())})
        tokens=arrays['tokens']
        if tokens.shape!=(2274,78,768) or tokens.dtype!=np.float32: raise RuntimeError('Cached tokens shape/dtype changed')
        for start in range(0,len(tokens),64):
            if not np.isfinite(tokens[start:start+64]).all():raise RuntimeError('Nonfinite cached upstream tokens')
        upstream=torch.load(checkpoint_path(),map_location='cpu',weights_only=True,mmap=True)['model']
        tail_source={f'tail.{name.removeprefix("encoder.layers.11.")}':value for name,value in upstream.items() if name.startswith('encoder.layers.11.')}
        tail_source.update({f'finish_norm.{name.removeprefix("encoder.layer_norm.")}':value for name,value in upstream.items() if name.startswith('encoder.layer_norm.')})
        expected={(arm,fold,42,f'inner_{s["scenario"]}') for arm in study.protocol['probe_arms'] for fold in study.protocol['folds'] for s in grid(study.protocol,arm)}
        expected|={(arm,fold,seed,f'outer_seed_{seed}') for arm in study.protocol['probe_arms'] for fold in study.protocol['folds'] for seed in study.protocol['outer_seeds']}
        expected|={('pretrained_tail',1,42,f'inner_{s["scenario"]}') for s in grid(study.protocol,'pretrained_tail')}
        expected.add(('pretrained_tail',1,42,'outer_seed_42'))
        cases=[];traces=[];choices=[];sources={};frames={};observed=set();tail_changes=[]
        cache_sources={str(feature_folder/name):digest for name,digest in feature_manifest['files'].items()}
        for marker in sorted((study.run/'cases').glob('*/*/*/completed.json')):
            folder=marker.parent;case,result,table=verify_case(study,folder)
            arm,fold,seed=case['arm'],case['fold'],case['seed'];observed.add((arm,fold,seed,folder.name))
            if case['device']!='mps' or case['feature_cache_manifest']!=feature_hash or case['test_signals_read']:
                raise RuntimeError('Case device/cache/test provenance changed')
            choice=None if case['inner_selection'] else read_json(study.run/'choices'/arm/f'fold_{fold}.json')
            head_lr=read_json(study.run/'choices/pretrained_probe/fold_1.json')['learning_rate'] if arm=='pretrained_tail' else None
            settings=settings_for(study,arm,case['scenario'],seed,fold,None if choice is None else choice['selected_epoch'],head_lr)
            if case['settings']!=settings:raise RuntimeError('Settings do not follow frozen protocol')
            train,evaluation,mean,scale=make_datasets(study,arm,case['train_ids'],case['evaluation_ids'],patients,windows,arrays)
            scaler_hash=canonical_hash({'mean':mean.tolist(),'scale':scale.tolist()})
            if scaler_hash!=case['scaler_hash']:raise RuntimeError('Scaler hash changed')
            state=torch.load(folder/'selected.pth',map_location='cpu',weights_only=True)
            if sha256_file(folder/'selected.pth')!=result['selected_checkpoint_sha256']:raise RuntimeError('Selected checkpoint SHA mismatch')
            changes=checkpoint_shapes(study,state,mean,scale,tail_source if arm=='pretrained_tail' else None)
            if changes:tail_changes.append({'case':str(folder.relative_to(study.run)),**changes})
            del state,train,evaluation
            history=pd.read_csv(folder/'history.csv');numeric=history.select_dtypes('number')
            if not np.isfinite(numeric).all().all():raise RuntimeError('Nonfinite training history')
            per_epoch=int(np.ceil(len(case['train_ids'])/settings['batch_size']))
            if not history.optimizer_steps.eq(per_epoch).all() or result['optimizer_steps']!=per_epoch*len(history):
                raise RuntimeError('Optimizer step budget changed')
            lr=np.array([settings['learning_rate']*(1+np.cos(np.pi*epoch/settings['scheduler_t_max']))/2 for epoch in range(len(history))])
            head=np.array([settings['head_learning_rate']*(1+np.cos(np.pi*epoch/settings['scheduler_t_max']))/2 for epoch in range(len(history))])
            if not np.allclose(history.learning_rate,lr,rtol=1e-9,atol=1e-15) or not np.allclose(history.head_learning_rate,head,rtol=1e-9,atol=1e-15):
                raise RuntimeError('Learning-rate schedule changed')
            if case['inner_selection'] and not np.isclose(result['f1'],result['best_inner_f1'],rtol=0,atol=1e-12):
                raise RuntimeError('Inner selected metrics differ from selected history')
            selected=history.iloc[result['selected_epoch']-1]
            traces.append({'arm':arm,'fold':fold,'seed':seed,'case':folder.name,'inner_selection':case['inner_selection'],
                'selected_epoch':result['selected_epoch'],'epochs_run':len(history),
                'selected_online_train_accuracy':float(selected.train_accuracy),'selected_online_train_loss':float(selected.train_loss),
                'last_online_train_accuracy':float(history.iloc[-1].train_accuracy),'selected_evaluation_f1':result['f1'],
                'elapsed_seconds':result['elapsed_seconds']})
            row={k:result[k] for k in ['arm','fold','seed','scenario','learning_rate','batch_size','selected_epoch','epochs_run',
                'optimizer_steps','inner_selection','accuracy','f1','sensitivity','auc']}
            for name in study.classes:row[f'recall_{name}']=result['classification_report'][name]['recall']
            cases.append(row)
            if not case['inner_selection']:frames[(arm,fold,seed)]=table
            for name,digest in read_json(marker)['files'].items():sources[str((folder/name).relative_to(ROOT))]=digest
            sources[str(marker.relative_to(ROOT))]=sha256_file(marker)
        if observed!=expected:raise RuntimeError('Expected exactly 99 inner + 31 outer cases')
        cases=pd.DataFrame(cases);traces=pd.DataFrame(traces)
        for path in sorted((study.run/'choices').glob('*/*.json')):
            choice=read_json(path);arm,fold=choice['arm'],choice['fold']
            group=cases[cases.arm.eq(arm)&cases.fold.eq(fold)&cases.inner_selection]
            best=select_best(group.to_dict('records'))
            for key in ['scenario','learning_rate','batch_size','selected_epoch']:
                if best[key]!=choice[key]:raise RuntimeError('Grid selection changed')
            if not np.isclose(best['f1'],choice['inner_f1'],rtol=0,atol=1e-12):raise RuntimeError('Grid selection F1 changed')
            choices.append({k:choice[k] for k in ['arm','fold','seed','scenario','learning_rate','head_learning_rate','batch_size','selected_epoch','inner_f1']})
            sources[str(path.relative_to(ROOT))]=sha256_file(path)
        outer=cases[~cases.inner_selection];inner=cases[cases.inner_selection]
        summary=outer.groupby('arm').agg(f1_mean=('f1','mean'),f1_std=('f1','std'),cases=('f1','size'),
            recall_lvot=('recall_LVOT','mean'),recall_cusp=('recall_RVOT_cusp','mean'),
            recall_other=('recall_RVOT_septal_freewall_other','mean')).reset_index()
        original_summary=pd.read_csv(original/'summary.csv')
        for row in summary.itertuples():
            old=original_summary[original_summary.arm.eq(row.arm)].iloc[0]
            if not np.isclose(row.f1_mean,old.mean_fold_macro_f1,rtol=0,atol=1e-12):raise RuntimeError('Saved summary mismatch')
        seed_summary=outer.groupby(['arm','seed']).agg(mean_fold_macro_f1=('f1','mean'),folds=('f1','size')).reset_index()
        pooled=[];pooled_predictions={};confusions=[]
        for arm in study.protocol['probe_arms']:
            preds=[]
            for seed in study.protocol['outer_seeds']:
                combined=pd.concat([frames[(arm,fold,seed)] for fold in study.protocol['folds']],ignore_index=True).sort_values('hospital_id').reset_index(drop=True)
                if combined.hospital_id.tolist()!=patients.hospital_id.tolist():raise RuntimeError('OOF missing/duplicate patients')
                probabilities=combined[[f'probability_{c}' for c in study.classes]].to_numpy();truth=combined.true_label.to_numpy()
                metrics=compute_metrics(truth,probabilities);pooled.append({'arm':arm,'seed':seed,'patients':len(combined),**metrics})
                old=pd.read_csv(original/f'oof_{arm}_seed{seed}.csv',dtype={'hospital_id':str}).sort_values('hospital_id').reset_index(drop=True)
                pd.testing.assert_frame_equal(combined,old,check_dtype=False,check_exact=False,rtol=1e-9,atol=1e-10)
                pred=probabilities.argmax(1);preds.append(pred)
                cm=np.bincount(truth*3+pred,minlength=9).reshape(3,3)
                for y in range(3):
                    for predicted in range(3):confusions.append({'arm':arm,'seed':seed,'true_class':study.classes[y],
                        'predicted_class':study.classes[predicted],'patients':int(cm[y,predicted])})
            pooled_predictions[arm]=np.stack(preds)
        pooled=pd.DataFrame(pooled);old=pd.read_csv(original/'pooled_oof.csv')
        pd.testing.assert_frame_equal(pooled,old,check_dtype=False,check_exact=False,rtol=1e-9,atol=1e-10)
        paired=outer[outer.arm.eq('pretrained_probe')].merge(outer[outer.arm.eq('random_probe')],on=['fold','seed'],suffixes=('_pretrained','_random'))
        paired['f1_delta_pretrained_minus_random']=paired.f1_pretrained-paired.f1_random
        bootstrap=paired_patient_bootstrap(patients.label.to_numpy(),pooled_predictions['random_probe'],pooled_predictions['pretrained_probe'])
        tables={'cases_verified.csv':cases,'summary_verified.csv':summary,'seed_summary.csv':seed_summary,
            'choices_verified.csv':pd.DataFrame(choices),'training_trace_summary.csv':traces,'feature_summary.csv':pd.DataFrame(summaries),
            'paired_probe_scores.csv':paired,'pooled_oof_verified.csv':pooled,'confusion_by_seed.csv':pd.DataFrame(confusions),
            'tail_parameter_changes.csv':pd.DataFrame(tail_changes)}
        for name,frame in tables.items():frame.to_csv(destination/name,index=False)
        write_json(destination/'paired_patient_bootstrap.json',bootstrap)
        notebooks=[]
        paths=[path for path in sorted((ROOT/'notebooks').glob('*V10*.ipynb')) if path.name[:2] in {'51','52','53','54','55'}]
        if len(paths)!=5:raise RuntimeError('Require the five original V10 notebooks')
        for path in paths:
            notebook=read_json(path);code=[c for c in notebook['cells'] if c['cell_type']=='code']
            errors=[o for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error']
            if errors or any(c['execution_count'] is None for c in code):raise RuntimeError('Manual notebook incomplete/error')
            notebooks.append({'file':str(path.relative_to(ROOT)),'code_cells_complete':len(code),'saved_errors':0,'sha256':sha256_file(path)})
        checked=snapshot_unchanged(snapshot)
        audit={'experiment_hash':study.signature,'cases_verified':130,'inner_cases_verified':99,'outer_cases_verified':31,
            'finite_selected_checkpoints_verified':130,'checkpoint_training_scalers_verified':130,'tail_parameter_updates_verified':10,
            'inner_choices_verified':len(choices),'epochs':int(cases.epochs_run.sum()),'optimizer_steps':int(cases.optimizer_steps.sum()),
            'development_patients':230,'development_windows':2274,'feature_cache_sha256':feature_hash,
            'feature_cache_finite_and_unique':True,'six_oof_sets_cover_exactly_230_patients':True,'all_saved_metrics_reproduced':True,
            'pretrained_probe_mean_fold_f1_delta':float(paired.f1_delta_pretrained_minus_random.mean()),
            'pretrained_probe_winning_fold_seed_pairs':int((paired.f1_delta_pretrained_minus_random>0).sum()),
            'paired_comparisons':len(paired),'tail_is_only_fold1_seed42_pilot':True,
            'protected_source_results_unchanged':checked,'notebooks':notebooks,'source_sha256':sources,'feature_files_sha256':cache_sources,
            'audit_code_sha256':sha256_file(Path(__file__)),'new_training_performed':False,'new_genuine_model_inference_performed':False,
            'new_genuine_waveform_inference_performed':False,'test_signals_read':False,'pretraining_patient_independence_verified':False}
        write_json(destination/'audit.json',audit)
        print(json.dumps({k:v for k,v in audit.items() if k not in ['source_sha256','feature_files_sha256','notebooks']},indent=2),flush=True)
        print(summary.to_string(index=False),flush=True)
        print('Paired patient bootstrap',bootstrap,flush=True)
        return study,destination,audit


if __name__=='__main__':audit()
