"""Reconstruct patient metrics; partial tail pilot never counted as full CV."""
import numpy as np
import pandas as pd
from src.artifacts import canonical_hash, read_json, sha256_file, write_json
from src.evaluation import report
from research_v6.common import verify_files
from .common import grid, select_best, split_for


def verify_case(study,folder):
    done = read_json(folder/'completed.json'); verify_files(folder,done)
    fit_done = read_json(folder/'fit_finished.json'); verify_files(folder,fit_done)
    case = read_json(folder/'case.json'); result = read_json(folder/'metrics.json')
    if case['experiment_hash'] != study.signature or done['case_hash'] != canonical_hash(case): raise RuntimeError('Case binding changed')
    split = split_for(study,case['fold']); inner = case['inner_selection']
    train = split['gan_fit_ids'] if inner else split['gan_fit_ids']+split['gan_reference_ids']
    evaluation = split['gan_reference_ids'] if inner else split['outer_validation_ids']
    if case['train_ids'] != sorted(train) or case['evaluation_ids'] != sorted(evaluation) or case['scaler_fit_ids'] != sorted(train):
        raise RuntimeError('Case patient partition/scaler scope changed')
    table = pd.read_csv(folder/'predictions.csv',dtype={'hospital_id':str})
    if table.hospital_id.tolist() != sorted(evaluation) or table.hospital_id.duplicated().any(): raise RuntimeError('Prediction patient set changed')
    labels = {r['hospital_id']:r['label'] for r in study.records}
    truth = table.true_label.to_numpy(); probabilities = table[[f'probability_{c}' for c in study.classes]].to_numpy()
    if any(labels[hid] != y for hid,y in zip(table.hospital_id,truth)): raise RuntimeError('Prediction truth changed')
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.allclose(probabilities.sum(1),1,atol=1e-5):
        raise RuntimeError('Invalid patient probabilities')
    reconstructed = report(truth,probabilities,study.classes)
    for metric,value in reconstructed['metrics'].items():
        if value is not None and not np.isclose(value,result[metric],rtol=0,atol=1e-10): raise RuntimeError(f'Metric {metric} changed')
    if reconstructed['confusion_matrix'] != result['confusion_matrix']: raise RuntimeError('Confusion changed')
    history = pd.read_csv(folder/'history.csv'); fit_result = read_json(folder/'fit_result.json')
    if len(history) != fit_result['epochs_run'] or history.epoch.tolist() != list(range(1,len(history)+1)):
        raise RuntimeError('History epochs incomplete')
    if inner:
        if result['selected_epoch'] != int(history.inner_f1.idxmax())+1: raise RuntimeError('Selected epoch is not first inner maximum')
    elif 'inner_f1' in history or result['selected_epoch'] != case['settings']['epochs'] or len(history) != case['settings']['epochs']:
        raise RuntimeError('Outer refit must be fixed-last without evaluation callbacks')
    if not inner:
        choice = read_json(study.run/'choices'/case['arm']/f'fold_{case["fold"]}.json')
        if (case['selection_hash'] != canonical_hash(choice) or case['settings']['epochs'] != choice['selected_epoch']
            or case['settings']['learning_rate'] != choice['learning_rate'] or case['settings']['batch_size'] != choice['batch_size']
            or case['settings']['head_learning_rate'] != choice['head_learning_rate']):
            raise RuntimeError('Outer refit is not bound to the frozen inner choice')
    counts = {c:sum(labels[hid]==c for hid in train) for c in range(3)}
    for c,n in counts.items():
        if not history[f'class_{c}_patients'].eq(n).all(): raise RuntimeError('Patient class exposure changed')
    for encoded in history.patient_exposure_json:
        import json
        exposure = json.loads(encoded)
        if set(exposure) != set(train) or set(exposure.values()) != {1}: raise RuntimeError('Patient exposure changed')
    return case,result,table


def analyze(study):
    destination = study.run/'analysis'; destination.mkdir(parents=True,exist_ok=True)
    rows=[]; sources={}; frames={}; inner_groups={}
    for marker in sorted((study.run/'cases').glob('*/*/*/completed.json')):
        folder=marker.parent; case,result,table=verify_case(study,folder)
        arm,fold,seed=case['arm'],case['fold'],case['seed']
        rows.append({k:result[k] for k in ['arm','fold','seed','scenario','learning_rate','batch_size','selected_epoch',
            'epochs_run','optimizer_steps','inner_selection','accuracy','f1','sensitivity','auc']})
        sources[str(marker.relative_to(study.run))]=sha256_file(marker)
        if case['inner_selection']: inner_groups.setdefault((arm,fold),[]).append(result)
        else: frames[(arm,fold,seed)]=table
    verified_choices=0
    for (arm,fold),group in inner_groups.items():
        if len(group)!=9: continue
        path=study.run/'choices'/arm/f'fold_{fold}.json'
        if not path.exists(): continue
        choice=read_json(path); best=select_best(group)
        if choice['experiment_hash']!=study.signature or any(choice[k]!=best[k] for k in ['scenario','learning_rate','batch_size','selected_epoch']):
            raise RuntimeError('Choice does not match complete inner grid')
        for name,digest in choice['source_completed_sha256'].items():
            if sha256_file(study.run/name)!=digest: raise RuntimeError('Selection source changed')
        verified_choices+=1
    columns=['arm','fold','seed','scenario','learning_rate','batch_size','selected_epoch','epochs_run','optimizer_steps',
        'inner_selection','accuracy','f1','sensitivity','auc']
    cases=pd.DataFrame(rows,columns=columns); cases.to_csv(destination/'cases_verified.csv',index=False)
    outer=cases.loc[~cases.inner_selection.astype(bool)]
    summaries=[]; oof=[]; recalls=[]
    for arm in [*study.protocol['probe_arms'],'pretrained_tail']:
        subset=outer.loc[outer.arm.eq(arm)]
        expected=1 if arm=='pretrained_tail' else 15
        summaries.append({'arm':arm,'completed_outer_cases':len(subset),'expected_outer_cases':expected,
            'scope':'one_fold_one_seed_pilot' if arm=='pretrained_tail' else 'five_fold_three_head_seed_benchmark',
            'complete':len(subset)==expected,'mean_fold_macro_f1':float(subset.f1.mean()) if len(subset) else None,
            'sd_fold_seed_macro_f1':float(subset.f1.std()) if len(subset)>1 else None})
        for seed in study.protocol['outer_seeds']:
            tables=[frames[(arm,f,seed)] for f in study.protocol['folds'] if (arm,f,seed) in frames]
            if len(tables)!=5: continue
            combined=pd.concat(tables,ignore_index=True).sort_values('hospital_id')
            if set(combined.hospital_id)!={r['hospital_id'] for r in study.records} or combined.hospital_id.duplicated().any():
                raise RuntimeError('OOF requires exactly one prediction per development patient')
            p=combined[[f'probability_{c}' for c in study.classes]].to_numpy(); truth=combined.true_label.to_numpy()
            reconstructed=report(truth,p,study.classes)
            oof.append({'arm':arm,'seed':seed,'patients':len(combined),**reconstructed['metrics']})
            for c,name in enumerate(study.classes):
                recalls.append({'arm':arm,'seed':seed,'class':name,'recall':reconstructed['classification_report'][name]['recall']})
            combined.to_csv(destination/f'oof_{arm}_seed{seed}.csv',index=False)
    summary=pd.DataFrame(summaries); summary.to_csv(destination/'summary.csv',index=False)
    pd.DataFrame(oof).to_csv(destination/'pooled_oof.csv',index=False)
    pd.DataFrame(recalls).to_csv(destination/'class_recall.csv',index=False)
    if len(cases):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        for i,arm in enumerate(study.protocol['probe_arms']):
            subset=outer.loc[outer.arm.eq(arm)]
            if len(subset): ax.scatter(np.full(len(subset),i),subset.f1,alpha=.7,label=arm)
        ax.set_xticks(range(2),study.protocol['probe_arms']);ax.set_ylim(0,1);ax.set_ylabel('Outer patient macro-F1')
        ax.set_title('Frozen probes: each point is a fold × head seed')
        fig.savefig(destination/'outer_macro_f1.png',dpi=180);plt.close(fig)
        for (arm,fold),group in inner_groups.items():
            matrix=np.full((3,3),np.nan); rates=sorted({r['learning_rate'] for r in grid(study.protocol,arm)})
            for r in group: matrix[rates.index(r['learning_rate']),study.protocol['patient_batch_sizes'].index(r['batch_size'])]=r['f1']
            fig,ax=plt.subplots(figsize=(5,4),layout='constrained');im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis')
            ax.set_xticks(range(3),study.protocol['patient_batch_sizes']);ax.set_yticks(range(3),[f'{x:g}' for x in rates])
            ax.set_xlabel('Patient batch size');ax.set_ylabel('Learning rate');ax.set_title(f'Inner macro-F1: {arm}, fold {fold}')
            for y in range(3):
                for x in range(3):
                    if np.isfinite(matrix[y,x]): ax.text(x,y,f'{matrix[y,x]:.3f}',ha='center',va='center',color='white')
            fig.colorbar(im,ax=ax);fig.savefig(destination/f'inner_grid_{arm}_fold{fold}.png',dpi=180);plt.close(fig)
    audit={'experiment_hash':study.signature,'cases_verified':len(cases),'inner_cases':int(cases.inner_selection.sum()) if len(cases) else 0,
        'outer_cases':len(outer),'inner_choices_verified':verified_choices,'full_probe_benchmark_complete':all(s['complete'] for s in summaries[:2]),
        'tail_scope':'fold1_seed42_pilot_only','source_completed_sha256':sources,'new_training_performed':False,
        'new_waveform_inference_performed':False,'test_signals_read':False,'historical_test_previously_accessed':True,
        'confirmatory_unseen_test_claim_allowed':False}
    audit['pretraining_patient_independence_verified']=False
    audit['pretraining_provenance_limitation']='shared_ningbo_hospital_source_patient_overlap_unverified'
    write_json(destination/'audit.json',audit)
    lines=['# V10 development transfer analysis', '', '```text\n'+summary.to_string(index=False)+'\n```' if len(summary) else 'Belum ada hasil.', '',
        'Macro-F1 pada tabel adalah mean per fold/seed. Pooled OOF dilaporkan terpisah karena F1 nonlinear.',
        'Tiga seed head menggunakan split dan encoder yang sama; 15 skor bukan 15 sampel independen.',
        'Random probe adalah encoder acak beku, bukan model from-scratch yang dilatih penuh.',
        'Fine-tuning layer terakhir hanya pilot fold1 seed42. Skornya tidak boleh dianggap hasil lima fold.',
        'Seluruh analisis development bersifat eksploratif. Test historis pernah diakses pada fase lama; tidak menjadi confirmatory unseen test baru.',
        'Pretraining ECG-FM mencakup Ningbo; dataset lokalisasi juga dari Ningbo First Hospital. Belum ada crosswalk pasien yang membuktikan ada/tidaknya overlap. Independensi pretraining belum terverifikasi.',
        'Tidak ada jaminan macro-F1 0.9 atau publikasi Q1. Evaluasi klinis/eksternal masih perlu untuk klaim generalisasi.']
    (destination/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
    print(summary.to_string(index=False),flush=True)
    return audit
