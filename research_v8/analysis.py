"""Check real-patient predictions and paired augmentation effects; no test access."""
import os
import json
import numpy as np
import pandas as pd
from src.artifacts import ROOT,read_json,sha256_file,write_json
from src.evaluation import compute_metrics
from .common import checked_complete,directory,progress,require_prepared
from .data import load_development


def checked_predictions(study,model,fold,seed,arm,table):
    result=checked_complete(study,model,fold,seed,arm)
    folder=directory(study,model,fold,seed,arm);case=read_json(folder/'case.json')
    split=study.folds[fold-1];ids=split['outer_validation_ids'] if arm!='inner_selection' else split['gan_reference_ids']
    expected={r['hospital_id']:r['label'] for r in study.records if r['hospital_id'] in ids}
    frame=pd.read_csv(folder/'predictions.csv',dtype={'hospital_id':str});classes=study.inherited['class_names']
    if not frame.hospital_id.is_unique or set(frame.hospital_id)!=set(expected):raise RuntimeError('Wrong patient coverage')
    truth=np.array([expected[h] for h in frame.hospital_id]);p=frame[[f'probability_{c}' for c in classes]].to_numpy()
    if (not np.array_equal(frame.true_label,truth) or not np.isfinite(p).all() or (p<0).any()
            or not np.allclose(p.sum(1),1,atol=2e-6)):
        raise RuntimeError('Invalid probabilities or labels')
    pred=p.argmax(1)
    if (frame.predicted_class.tolist()!=[classes[c] for c in pred]
            or not np.array_equal(frame.correct,pred==truth)):
        raise RuntimeError('Prediction metadata changed')
    computed=compute_metrics(truth,p)
    for key,v in computed.items():
        if not np.isclose(v,result['metrics'][key],rtol=1e-8,atol=1e-9):raise RuntimeError(f'Metric differs: {key}')
    segment_weights=pd.read_csv(folder/'segment_weights.csv',dtype={'hospital_id':str})
    indices=np.flatnonzero(table.hospital_id.isin(ids).to_numpy())
    if (not segment_weights.segment_index.is_unique or set(segment_weights.segment_index)!=set(indices)
            or not np.array_equal(table.iloc[segment_weights.segment_index].hospital_id.to_numpy(),segment_weights.hospital_id)
            or not np.isfinite(segment_weights.weight).all() or (segment_weights.weight<0).any()
            or not np.allclose(segment_weights.groupby('hospital_id').weight.sum(),1,atol=2e-6)):
        raise RuntimeError('Real global patient pooling coverage differs')
    counts=segment_weights.groupby('hospital_id').size()
    if not np.array_equal(segment_weights.hospital_id.map(counts),segment_weights.n_patient_segments):
        raise RuntimeError('Pooling segment counts differ')
    if study.choices[model]['pooling']=='mean' and not np.allclose(segment_weights.weight,1/segment_weights.hospital_id.map(counts),atol=2e-6):
        raise RuntimeError('Mean pooling weights differ')
    history=pd.read_csv(folder/'history.csv')
    if history.epoch.tolist()!=list(range(1,result['epochs_run']+1)):raise RuntimeError('Epoch history incomplete')
    if arm=='inner_selection':
        if result['selected_epoch']!=int(history.loc[history.inner_f1.idxmax(),'epoch']):raise RuntimeError('Inner epoch selection differs')
        if not np.isclose(history.inner_f1.max(),computed['f1']):raise RuntimeError('Inner selected prediction differs')
    else:
        selector=checked_complete(study,model,fold,seed)
        if (result['epochs_run']!=selector['selected_epoch'] or result['selected_epoch']!=selector['selected_epoch']
                or case['epoch_selection_manifest_sha256']!=sha256_file(directory(study,model,fold,seed)/'completed.json')
                or any(c.startswith('inner_') or c.startswith('validation_') for c in history.columns)):
            raise RuntimeError('Outer budget/selection was changed')
    expected_train=set(case['train_ids']);expected_steps=int(np.ceil(len(expected_train)/study.protocol['patient_batch_size']))
    if not history.optimizer_steps.eq(expected_steps).all() or not history.anchor_patients.eq(len(expected_train)).all():raise RuntimeError('Paired update budget differs')
    patient_labels={r['hospital_id']:r['label'] for r in study.records}
    fake_labels=pd.read_csv(study.gan.folder(fold)/'synthetic_labels.csv').label.to_numpy()
    for row in history.itertuples():
        anchors=json.loads(row.anchor_exposure_json)
        if set(anchors)!=expected_train or set(anchors.values())!={1}:raise RuntimeError('Anchor exposure differs')
        if row.replacements!=(0 if arm in ('inner_selection','real_only') else row.eligible_bags):raise RuntimeError('Dose differs')
        if (row.synthetic_windows!=(row.replacements if arm=='gan_aug' else 0)
                or row.real_donor_windows!=(row.replacements if arm=='real_aug' else 0)):
            raise RuntimeError('Source accounting differs')
        donors=json.loads(row.real_donor_exposure_json)
        if not set(donors)<=set(split['gan_fit_ids']):raise RuntimeError('Real donor leaked outside GAN-fit')
        fakes=json.loads(row.synthetic_exposure_json)
        if any(not 0<=int(i)<len(fake_labels) or n<=0 for i,n in fakes.items()):raise RuntimeError('Synthetic source accounting differs')
        for c in range(3):
            source_count=sum(n for hid,n in donors.items() if patient_labels[hid]==c) if arm=='real_aug' else sum(n for i,n in fakes.items() if fake_labels[int(i)]==c)
            if source_count!=getattr(row,f'class_{c}_replacements'):raise RuntimeError('Augmentation condition accounting differs')
    recall={f'recall_class_{c}':float((pred[truth==c]==c).mean()) for c in range(3)}
    if any(not np.isclose(recall[f'recall_class_{c}'],result[f'validation_recall_class_{c}']) for c in range(3)):
        raise RuntimeError('Class recall differs')
    return frame,{**computed,**recall,'epochs':result['epochs_run'],'selected_epoch':result['selected_epoch'],
        'optimizer_steps':result['optimizer_steps'],'elapsed_seconds':result['elapsed_seconds'],
        'missing_predicted_classes':int(sum(not (pred==c).any() for c in range(3)))}


def analyze(study,pilot=False):
    require_prepared(study)
    if not pilot and not progress(study)['ready_for_comparison']:raise RuntimeError('Lengkapi notebook 43 dan 44 dahulu')
    _,table=load_development(study);rows=[];prediction_frames=[]
    for m in study.protocol['models']:
        for f in ([1] if pilot else study.protocol['folds']):
            for seed in ([42] if pilot else study.protocol['seeds']):
                checked_predictions(study,m,f,seed,'inner_selection',table)
                for arm in study.protocol['arms']:
                    frame,result=checked_predictions(study,m,f,seed,arm,table)
                    rows.append({'model':m,'fold':f,'seed':seed,'arm':arm,**result})
                    prediction_frames.append(frame.assign(model=m,fold=f,seed=seed,arm=arm))
    cases=pd.DataFrame(rows)
    output=study.run/('analysis_pilot' if pilot else 'analysis_complete');output.mkdir(exist_ok=True)
    deltas=[]
    indexed=cases.set_index(['model','fold','seed','arm'])
    for row in cases.itertuples():
        if row.arm=='real_only':continue
        control=indexed.loc[(row.model,row.fold,row.seed,'real_only')]
        donors=indexed.loc[(row.model,row.fold,row.seed,'real_aug')]
        deltas.append({'model':row.model,'fold':row.fold,'seed':row.seed,'arm':row.arm,
            'delta_f1_vs_real_only':row.f1-control.f1,'delta_f1_vs_real_aug':row.f1-donors.f1,
            'delta_rvot_cusp_recall':row.recall_class_1-control.recall_class_1,
            'delta_rvot_other_recall':row.recall_class_2-control.recall_class_2})
    delta=pd.DataFrame(deltas)
    summary=cases.groupby(['model','arm']).agg(f1_mean=('f1','mean'),f1_std=('f1','std'),
        recall_rvot_cusp_mean=('recall_class_1','mean'),recall_rvot_other_mean=('recall_class_2','mean'),
        collapse_cases=('missing_predicted_classes',lambda x:int(x.gt(0).sum())),seconds_total=('elapsed_seconds','sum')).reset_index()
    predictions=pd.concat(prediction_frames,ignore_index=True)
    seed_summary=cases.groupby(['model','seed','arm']).f1.mean().rename('fold_mean_f1').reset_index()
    pooled=[]
    if not pilot:
        for (model,seed,arm),group in predictions.groupby(['model','seed','arm']):
            if len(group)!=len(study.records) or not group.hospital_id.is_unique:raise RuntimeError('OOF must cover every patient once per arm/seed')
            p=group[[f'probability_{c}' for c in study.inherited['class_names']]].to_numpy()
            pooled.append({'model':model,'seed':seed,'arm':arm,**compute_metrics(group.true_label.to_numpy(),p)})
    for name,data in (('cases.csv',cases),('paired_deltas.csv',delta),('summary.csv',summary),('seed_summary.csv',seed_summary),
                      ('patient_predictions.csv',predictions),('pooled_oof.csv',pd.DataFrame(pooled))):data.to_csv(output/name,index=False)
    plot(output,cases)
    summary_lines=['| Model | Arm | Mean macro-F1 | SD deskriptif | Recall cusp | Recall other | Collapse cases |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in summary.itertuples():
        deviation='—' if not np.isfinite(row.f1_std) else f'{row.f1_std:.4f}'
        summary_lines.append(f'| {row.model} | {row.arm} | {row.f1_mean:.4f} | {deviation} | {row.recall_rvot_cusp_mean:.4f} | {row.recall_rvot_other_mean:.4f} | {row.collapse_cases} |')
    lines=['# Perbandingan classifier V8'+(' — pilot fold 1 seed 42' if pilot else ''),'',
        'Seluruh metrik direkonstruksi dari prediksi pasien real. Epoch dipilih dari inner real-only dan sama antarketiga arm.','',
        *summary_lines,'',
        '![Macro-F1 pembanding](comparison_f1.png)','',
        'Pilot mengecek proses; satu fold/seed belum cukup memilih arm atau menyatakan manfaat. Selesaikan seluruh kasus meskipun sebagian hasil negatif.','',
        'Bandingkan GAN dengan real-only DAN real augmentation; periksa kedua recall RVOT, arah efek per seed, OOF, collapse, exposure dan biaya. Tidak memilih seed terbaik.','',
        'Variasi fold/seed deskriptif, bukan 15 sampel independen. LR/pooling sudah dipilih pada development V4; ini bukan estimasi nested yang bebas seluruh pilihan sebelumnya.','',
        'Sintetis hanya menggantikan satu window pada bag K4 penuh. Mixed bags bukan pasien nyata. Label sintetis adalah kondisi diagnosis, belum anotasi PVC tervalidasi.','',
        'Tidak ada final/test atau jaminan 0,9/Q1. Belum memperluas rasio atau membuka test untuk memilih arm.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(output/'audit.json',{'experiment_hash':study.signature,'cases_verified':len(cases),
        'inner_cases_verified':len(cases)//3,'predictions_recomputed':True,'pilot':pilot,
        'test_signals_read':False,'analysis_code_sha256':sha256_file(__file__)})
    print(f'ANALYSIS complete: {len(cases)} comparison cases; {output / "REPORT.md"}',flush=True)


def plot(output,data):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=data.model.unique();fig,axes=plt.subplots(1,len(models),figsize=(6*len(models),4),squeeze=False)
    for ax,model in zip(axes[0],models):
        for x,arm in enumerate(('real_only','real_aug','gan_aug')):
            values=data[(data.model==model)&(data.arm==arm)].f1.to_numpy()
            ax.scatter(np.full(len(values),x),values,s=22,alpha=.6)
            ax.plot([x-.18,x+.18],[values.mean()]*2,color='black',linewidth=2)
        ax.set(title=model,xticks=range(3),xticklabels=('Real only','Real augmentation','GAN augmentation'),ylim=(0,1),ylabel='Real-patient macro-F1')
    fig.tight_layout();fig.savefig(output/'comparison_f1.png',dpi=150);plt.close(fig)
