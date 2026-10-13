"""Verify 120 cases and explain paired effects without rewriting original V8.

Reads saved predictions/checkpoints on CPU. No waveform inference or training.
"""
from pathlib import Path
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from src.artifacts import ROOT,canonical_hash,read_json,sha256_file,write_json
from src.evaluation import compute_metrics
from research_v8.common import DEFAULT_RUN,context,progress,directory,require_prepared,verify_files
from research_v8.analysis import checked_predictions
from research_v8.data import load_development


def unchanged(path):
    before=read_json(path)
    bad=[p for p,d in before.items() if not (ROOT/p).is_file() or sha256_file(ROOT/p)!=d]
    if bad:raise RuntimeError(f'Protected source/results changed: {bad}')
    return len(before)


def close_frames(actual,expected):
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=1e-9,atol=1e-10)


def audit(run_id=DEFAULT_RUN):
    s=context(run_id);require_prepared(s)
    if not progress(s)['ready_for_comparison']:raise RuntimeError('Complete notebook 41–45 first')
    output=s.run/'analysis_20261006_complete';output.mkdir(exist_ok=True)
    _,table=load_development(s)  # waveform mmap is not indexed; only segment metadata is used
    controls=read_json(ROOT/'results/research_v5/20261003_window_bag_ablation_v5/control_manifest.json')
    cases=[];inner_rows=[];predictions=[];sources={};epoch_comparison=[];dose_rows=[]
    checkpoint_count=0;case_epochs=case_updates=0
    for model in s.protocol['models']:
        smoke=read_json(s.run/f'smoke/{model}.json');verify_files(s.run,smoke)
        if smoke['experiment_hash']!=s.signature or smoke['device']!='mps' or not smoke['passed']:
            raise RuntimeError('Wrong MPS smoke identity')
        if [r['arm'] for r in smoke['arms']]!=s.protocol['arms'] or any(r['batch_shape']!=[8,4,12,5000] for r in smoke['arms']):
            raise RuntimeError('Wrong smoke coverage')
        sources[str((s.run/f'smoke/{model}.json').relative_to(ROOT))]=sha256_file(s.run/f'smoke/{model}.json')
        print(f'AUDIT {model}: 15 selectors and 45 paired refits',flush=True)
        for fold in s.protocol['folds']:
            split=s.folds[fold-1]
            for seed in s.protocol['seeds']:
                histories={};selected=None
                for arm in ('inner_selection',*s.protocol['arms']):
                    frame,result=checked_predictions(s,model,fold,seed,arm,table)
                    folder=directory(s,model,fold,seed,arm);case=read_json(folder/'case.json')
                    inner=arm=='inner_selection'
                    expected_train=sorted(split['gan_fit_ids'] if inner else split['gan_fit_ids']+split['gan_reference_ids'])
                    expected_eval=sorted(split['gan_reference_ids'] if inner else split['outer_validation_ids'])
                    if (case['train_ids']!=expected_train or case['evaluation_ids']!=expected_eval or case['device']!='mps'
                            or case['effective_seed']!=seed+100*fold+(500000 if inner else 0)
                            or case['pooling']!=s.choices[model]['pooling']
                            or case['source_gan_generated_sha256']!=sha256_file(s.gan.folder(fold)/'generated.json')
                            or case['scaler_sha256']!=sha256_file(s.run/f'scalers/{"inner" if inner else "outer"}_{fold}/scaler.joblib')):
                        raise RuntimeError('Wrong case patients/device/GAN/scaler')
                    settings={**s.training,'learning_rate':s.choices[model]['learning_rate']}
                    if not inner:settings['epochs']=selected
                    if case['settings']!=settings:raise RuntimeError('Wrong paired training recipe')
                    verify_files(folder,read_json(folder/'fit_finished.json'))
                    h=pd.read_csv(folder/'history.csv');histories[arm]=h
                    if not np.isfinite(h.select_dtypes('number')).all().all():raise RuntimeError('Nonfinite training history')
                    state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
                    if state['case_hash']!=case['case_hash'] or state['device']!='mps' or state['epoch']!=len(h):
                        raise RuntimeError('Last checkpoint does not match history')
                    close_frames(pd.DataFrame(state['history']),h)
                    name='best.pth' if inner else 'model.pth'
                    weights=torch.load(folder/name,map_location='cpu',weights_only=True)
                    if any(not torch.isfinite(v).all() for v in weights.values()):raise RuntimeError('Nonfinite selected weights')
                    if not inner and (set(weights)!=set(state['model']) or any(not torch.equal(v,state['model'][k]) for k,v in weights.items())):
                        raise RuntimeError('Outer selected weights are not fixed last epoch')
                    if read_json(folder/'metrics.json')['selected_checkpoint_sha256']!=sha256_file(folder/name):
                        raise RuntimeError('Selected checkpoint hash differs')
                    del state,weights
                    checkpoint_count+=1;case_epochs+=len(h);case_updates+=int(h.optimizer_steps.sum())
                    if inner:
                        selected=result['selected_epoch']
                        inner_rows.append({'model':model,'fold':fold,'seed':seed,**result})
                    else:
                        cases.append({'model':model,'fold':fold,'seed':seed,'arm':arm,**result})
                        predictions.append(frame.assign(model=model,fold=fold,seed=seed,arm=arm))
                    marker=read_json(folder/'completed.json')
                    sources[str((folder/'completed.json').relative_to(ROOT))]=sha256_file(folder/'completed.json')
                    sources.update({str((folder/n).relative_to(ROOT)):d for n,d in marker['files'].items()})
                base=histories['real_only']
                for arm in ('real_aug','gan_aug'):
                    h=histories[arm]
                    for key in ('epoch','optimizer_steps','learning_rate','anchor_exposure_json','eligible_bags','bag_windows'):
                        if base[key].tolist()!=h[key].tolist():raise RuntimeError('Anchors or compute budget differ between arms')
                counts=table[table.hospital_id.isin(expected_train)].groupby('hospital_id').size()
                expected_windows=int(counts.clip(upper=4).sum());eligible=int(counts.ge(4).sum())
                for arm,h in histories.items():
                    if arm!='inner_selection' and (not h.bag_windows.eq(expected_windows).all() or not h.eligible_bags.eq(eligible).all()):
                        raise RuntimeError('Short-bag or dose eligibility differs')
                fake_counts={}
                for value in histories['gan_aug'].synthetic_exposure_json:
                    for index,n in json.loads(value).items():fake_counts[index]=fake_counts.get(index,0)+n
                dose_rows.append({'model':model,'fold':fold,'seed':seed,'epochs':selected,
                    'replacement_window_fraction':float(histories['gan_aug'].replacements.sum()/histories['gan_aug'].bag_windows.sum()),
                    'distinct_synthetic_indices_used':len(fake_counts),'total_synthetic_draws':sum(fake_counts.values())})
                old=next(c for c in controls if c['model']==model and c['fold']==fold and c['seed']==seed)
                for name,digest in old['sha256'].items():
                    if sha256_file(ROOT/name)!=digest:raise RuntimeError('Historical control changed')
                old_folder=ROOT/old['origin'];old_history=pd.read_csv(old_folder/'history.csv');old_metrics=read_json(old_folder/'metrics.json')
                common=min(len(base),len(old_history))
                loss_diff=float(np.max(np.abs(base.train_loss.iloc[:common].to_numpy()-old_history.train_loss_minibatch_weighted_mean.iloc[:common].to_numpy())))
                accuracy_diff=float(np.max(np.abs(base.train_patient_bag_accuracy.iloc[:common].to_numpy()-old_history.train_accuracy_patient_bag.iloc[:common].to_numpy())))
                f1_at_epoch=float(old_history.iloc[selected-1].validation_f1) if selected<=len(old_history) else None
                new_f1=next(c['f1'] for c in cases if c['model']==model and c['fold']==fold and c['seed']==seed and c['arm']=='real_only')
                if loss_diff!=0 or accuracy_diff!=0:raise RuntimeError('Unexpected change from matched V4 real-only trajectory')
                if f1_at_epoch is not None and not np.isclose(f1_at_epoch,new_f1,atol=1e-10,rtol=1e-10):
                    raise RuntimeError('Predictions differ at the identical V4 epoch')
                epoch_comparison.append({'model':model,'fold':fold,'seed':seed,'inner_selected_epoch':selected,
                    'v4_outer_selected_epoch':old_metrics['best_epoch'],'v4_outer_selected_f1':old_metrics['metrics']['f1'],
                    'v8_real_only_f1':new_f1,'v4_f1_at_same_epoch':f1_at_epoch,
                    'training_loss_max_difference':loss_diff,'training_accuracy_max_difference':accuracy_diff,
                    'common_epochs_compared':common})
    cases=pd.DataFrame(cases);inner=pd.DataFrame(inner_rows);prediction=pd.concat(predictions,ignore_index=True)
    original=s.run/'analysis_complete'
    # Preserve ordering from the original analysis before checking it.
    close_frames(cases,pd.read_csv(original/'cases.csv'))
    old_predictions=pd.read_csv(original/'patient_predictions.csv',dtype={'hospital_id':str})
    close_frames(prediction,old_predictions)
    summary=cases.groupby(['model','arm']).agg(f1_mean=('f1','mean'),f1_std=('f1','std'),
        recall_rvot_cusp_mean=('recall_class_1','mean'),recall_rvot_other_mean=('recall_class_2','mean'),
        collapse_cases=('missing_predicted_classes',lambda x:int(x.gt(0).sum())),seconds_total=('elapsed_seconds','sum')).reset_index()
    close_frames(summary,pd.read_csv(original/'summary.csv'))
    seeds=cases.groupby(['model','seed','arm']).f1.mean().rename('fold_mean_f1').reset_index()
    close_frames(seeds,pd.read_csv(original/'seed_summary.csv'))
    pooled=[]
    for (model,seed,arm),g in prediction.groupby(['model','seed','arm']):
        if not g.hospital_id.is_unique or set(g.hospital_id)!={r['hospital_id'] for r in s.records}:raise RuntimeError('OOF coverage differs')
        p=g[[f'probability_{c}' for c in s.inherited['class_names']]].to_numpy()
        pooled.append({'model':model,'seed':seed,'arm':arm,**compute_metrics(g.true_label.to_numpy(),p)})
    pooled=pd.DataFrame(pooled);close_frames(pooled,pd.read_csv(original/'pooled_oof.csv'))
    effects=[]
    for model in s.protocol['models']:
        for seed in s.protocol['seeds']:
            g=cases[(cases.model==model)&(cases.seed==seed)].groupby('arm').mean(numeric_only=True)
            for control in ('real_only','real_aug'):
                effects.append({'model':model,'seed':seed,'control':control,'fold_mean_f1_delta':float(g.loc['gan_aug','f1']-g.loc[control,'f1']),
                    'rvot_cusp_recall_delta':float(g.loc['gan_aug','recall_class_1']-g.loc[control,'recall_class_1']),
                    'rvot_other_recall_delta':float(g.loc['gan_aug','recall_class_2']-g.loc[control,'recall_class_2'])})
    effects=pd.DataFrame(effects);epoch_comparison=pd.DataFrame(epoch_comparison);doses=pd.DataFrame(dose_rows)
    for name,frame in (('cases_verified.csv',cases),('inner_selection_verified.csv',inner),('summary_verified.csv',summary),
        ('paired_seed_effects.csv',effects),('pooled_oof_verified.csv',pooled),('epoch_selection_v4_v8.csv',epoch_comparison),
        ('augmentation_dose.csv',doses)):
        frame.to_csv(output/name,index=False)
    plot(output,seeds,summary)
    notebooks=[]
    for path in sorted((ROOT/'notebooks').glob('*V8*.ipynb')):
        nb=read_json(path);cells=[c for c in nb['cells'] if c['cell_type']=='code']
        errors=[o.get('evalue') for c in cells for o in c.get('outputs',[]) if o['output_type']=='error']
        executed=sum(c.get('execution_count') is not None for c in cells)
        if errors or executed!=len(cells):raise RuntimeError('Incomplete/error notebook')
        notebooks.append({'file':path.name,'executed_code_cells':executed,'total_code_cells':len(cells),'errors':errors})
    protected={p.name:unchanged(p) for p in (ROOT/'tmp/v8_frozen_before.json',ROOT/'tmp/v8_completed_before_audit_20261006.json')}
    result={'experiment_hash':s.signature,'inner_cases_verified':len(inner),'outer_cases_verified':len(cases),
        'finite_selected_checkpoints_verified':checkpoint_count,'total_completed_epochs':case_epochs,
        'total_optimizer_steps':case_updates,'outer_oof_prediction_rows':len(prediction),'unique_development_patients':len(s.records),
        'paired_epoch_update_anchor_budget_verified':True,'original_tables_reproduced':True,
        'v4_common_training_trajectories_identical':len(epoch_comparison),
        'v4_same_epoch_f1_available_and_identical':int(epoch_comparison.v4_f1_at_same_epoch.notna().sum()),
        'protected_unchanged':protected,'notebooks':notebooks,'source_sha256':sources,
        'audit_code_sha256':sha256_file(Path(__file__)),'new_training_performed':False,
        'new_waveform_inference_performed':False,'test_signals_read':False,
        'gan_benefit_consistent_across_controls_seeds_and_both_rvot':False}
    write_json(output/'audit.json',result)
    report(output,result,summary,effects,pooled,epoch_comparison)
    print(f'VERIFIED: {len(inner)} selectors + {len(cases)} paired refits, all predictions and checkpoints valid. Report: {output / "REPORT.md"}',flush=True)
    return result


def plot(output,seeds,summary):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=['real_only','real_aug','gan_aug'];models=['hierarchical','encoder']
    labels=['Real\nonly','Real\naugmentation','GAN\naugmentation']
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for model,ax in zip(models,axes):
        for seed in (42,43,44):
            g=seeds[(seeds.model==model)&(seeds.seed==seed)].set_index('arm')
            ax.plot(range(3),g.loc[arms].fold_mean_f1,marker='o',label=f'Seed {seed}')
        ax.set(title=model,xlim=(-.2,2.2),ylim=(0,1),xticks=range(3),xticklabels=labels,ylabel='Mean of five fold patient macro-F1')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(output/'paired_seed_macro_f1.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for model,ax in zip(models,axes):
        g=summary[summary.model.eq(model)].set_index('arm')
        for i,(field,name) in enumerate([('recall_rvot_cusp_mean','RVOT cusp'),('recall_rvot_other_mean','RVOT septal/freewall/other')]):
            ax.bar(np.arange(3)+(i-.5)*.3,g.loc[arms,field],width=.3,label=name)
        ax.set(title=model,ylim=(0,1),xticks=range(3),xticklabels=labels,ylabel='Mean real-patient recall across folds/seeds')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(output/'rvot_recall_comparison.png',dpi=180);plt.close(fig)


def report(output,audit,summary,effects,pooled,epoch):
    lines=['# Audit classifier V8 — 6 Oktober 2026','',
        '**120 fit selesai dan valid; manfaat GAN belum konsisten terhadap kedua kontrol dan kedua recall RVOT.**','',
        f'Fingerprint: `{audit["experiment_hash"]}`. Notebook 41–45 lengkap tanpa error tersimpan.','',
        '| Model | Arm | Mean fold/seed macro-F1 ± SD | Recall cusp | Recall other |',
        '|---|---|---:|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.model} | {r.arm} | {r.f1_mean:.4f} ± {r.f1_std:.4f} | {r.recall_rvot_cusp_mean:.1%} | {r.recall_rvot_other_mean:.1%} |')
    lines+=['','Encoder: GAN +0,0146 dibanding real-only, tetapi hanya +0,0024 dibanding augmentasi real. Mean fold-F1 meningkat terhadap real-only hanya pada seed 43; seed 42/44 turun. Recall other turun 7,62 poin persentase.','',
        'Hierarchical: GAN −0,0285 dibanding real-only dan −0,0543 dibanding augmentasi real. Mean fold-F1 turun pada seluruh tiga seed; recall other turun 20,48 poin persentase.','',
        '![Efek per seed](paired_seed_macro_f1.png)','',
        '![Recall RVOT](rvot_recall_comparison.png)','',
        '## Mengapa angka sekitar 0,5 berbeda dari V4','',
        'Training real-only cocok persis dengan V4 pada semua epoch yang sama-sama tersedia: loss/accuracy pada 30 trajectory identik. Pada 24 kasus dengan epoch pilihan V8 masih tersedia di history V4, F1 di epoch yang sama juga identik.','',
        'V4 memilih checkpoint dengan F1 tertinggi dari outer validation yang juga dipakai melaporkan CV. V8 memilih epoch dari 37 pasien inner, lalu melakukan refit dan menilai outer pada epoch tetap. Karena aturan pemilihan checkpoint berbeda, 0,65 V4 versus sekitar 0,51 V8 bukan pengukuran efek GAN. Baseline V8 tanpa GAN juga turun. Tidak ada bukti dari audit ini bahwa perubahan kode training/scaler/crop merusak real-only.','',
        'Pilihan epoch V8 bervariasi: Encoder 2–65 (median 14), Hierarchical 3–49 (median 16). Jumlah epoch lebih banyak tidak otomatis meningkatkan outer F1. Ukuran inner dan transfer epoch antara train 146–148 dan refit 183–185 adalah kandidat penelitian stabilitas, belum penyebab yang terbukti.','',
        '## Mean fold dan pooled OOF','',
        'Pooled OOF menghitung satu macro-F1 dari seluruh 230 pasien per arm/seed. Ini berbeda dari rata-rata lima macro-F1 fold karena F1 nonlinear; keduanya dihitung ulang dan benar. Pooled OOF Encoder menunjukkan sinyal lebih positif, tetapi penurunan recall other dan keuntungan kecil terhadap kontrol real tetap harus dilaporkan.','',
        '| Model | Arm | Mean pooled OOF ± SD tiga seed |','|---|---|---:|']
    for (m,a),g in pooled.groupby(['model','arm']):lines.append(f'| {m} | {a} | {g.f1.mean():.4f} ± {g.f1.std():.4f} |')
    lines+=['','## Keputusan','',
        'Tidak mengulang notebook 41–45. Bekukan hasil V8 sebagai ablation lengkap, termasuk hasil negatif. GAN recipe ini tidak dipilih sebagai perbaikan final Hierarchical, dan belum menjadi pilihan stabil Encoder. Tidak memperbesar jumlah sintetis/rasio atau memilih seed terbaik berdasarkan hasil ini.','',
        'Prioritas berikut: stabilitas seleksi epoch dan generalisasi real-only, lalu representasi/pretraining yang sesuai sumber ECG pada run baru. Pertahankan tiga kelas dan semua pasien. Jika GAN diuji lagi, pisahkan efek mixed bag/dosis dari kualitas generator; manfaat QC tidak sama dengan manfaat classifier.','',
        'Tidak ada final/test baru atau penentuan batas maksimal performa. Evaluasi tetap eksploratif karena LR/pooling dipilih pada development sebelumnya; 59 pasien test historis tidak dipakai memilih resep berikut.','',
        'Audit memverifikasi bobot pada CPU dan metrik dari prediksi tersimpan; tidak melatih atau menjalankan waveform inference baru. Hasil/kode/notebook asli dipertahankan.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default=DEFAULT_RUN)
    args=parser.parse_args();audit(args.run_id)
