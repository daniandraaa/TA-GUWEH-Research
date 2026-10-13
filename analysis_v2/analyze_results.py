"""Read-only model audit; creates separate reports without changing frozen training code."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score
from research_v2.common import context, verify_content
from src.evaluation import compute_metrics

LABELS = ['LVOT', 'RVOT cusp', 'RVOT septal/free-wall/other']
MODELS = ['encoder', 'patchtst', 'hierarchical']


def read_predictions(path, names, expected):
    t = pd.read_csv(path, dtype={'hospital_id': str})
    probs = t[[f'probability_{c}' for c in names]].to_numpy()
    if t.hospital_id.duplicated().any() or set(t.hospital_id) != expected:
        raise ValueError(f'Patient coverage mismatch: {path}')
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-5):
        raise ValueError(f'Invalid probabilities: {path}')
    return t, probs


def table_md(frame):
    # No optional tabulate dependency.
    def value(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.4f}'
        return str(v).replace('|', '/')
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(value(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None))


def analyze(run_id='20260930_anatomy_fs2000_v2'):
    run, source, cfg, config, scenarios = context(run_id)
    verify_content(run)
    out = run / 'analysis_20261001'; out.mkdir(exist_ok=True)
    selection = json.loads((run / 'selection.json').read_text())
    names = config['class_names']
    scenarios_by_id = {s['id']: s for s in scenarios}
    records = pd.DataFrame(json.loads((run / 'data/patients.json').read_text()))
    dev = set(records.loc[records.pool == 'development', 'hospital_id'])
    test = set(records.loc[records.pool == 'test', 'hospital_id'])
    if dev & test or (len(dev), len(test)) != (230, 59):
        raise ValueError('Cohort differs from frozen protocol')
    truth_map = records.set_index('hospital_id').label.to_dict()
    folds = json.loads((run / 'folds/manifest.json').read_text())
    if sorted(h for f in folds for h in f['validation_ids']) != sorted(dev):
        raise ValueError('Patient-level validation coverage differs')
    for f in folds:
        if set(f['train_ids']) & set(f['validation_ids']) or set(f['train_ids']) | set(f['validation_ids']) != dev:
            raise ValueError('Fold leakage')
        provenance = json.loads((run / 'folds' / f'fold_{f["fold"]}' / 'scaler_provenance.json').read_text())
        if set(provenance['fit_patient_ids']) != set(f['train_ids']):
            raise ValueError('Scaler was not fit on training fold only')
    provenance = json.loads((run / 'folds/final/scaler_provenance.json').read_text())
    if set(provenance['fit_patient_ids']) != dev:
        raise ValueError('Final scaler includes patients outside development')
    all_fold_rows = []
    oof = {}
    for arm in cfg['arms']:
        for model in MODELS:
            for sc in scenarios:
                for f in folds:
                    d = run / 'cv' / arm / model / sc['id'] / f'fold_{f["fold"]}'
                    t, probs = read_predictions(d / 'validation_predictions.csv', names, set(f['validation_ids']))
                    if any(truth_map[hid] != label for hid, label in zip(t.hospital_id, t.true_label)):
                        raise ValueError('True labels changed')
                    metrics = json.loads((d / 'metrics.json').read_text())
                    recomputed = compute_metrics(t.true_label.to_numpy(), probs)
                    if not all(np.isclose(metrics[k], v) for k, v in recomputed.items()):
                        raise ValueError(f'Metrics differ: {d}')
                    case = json.loads((d / 'config.json').read_text())
                    if set(case['train_patient_ids']) != set(f['train_ids']):
                        raise ValueError('Case train patient IDs differ')
                    expected_weight = 'inverse_frequency_training_fold' if arm == 'weighted_real' else 'none'
                    if case['settings']['class_weighting'] != expected_weight:
                        raise ValueError('Wrong class-weight arm')
                    if case['seed'] != config['seed'] + 100*f['fold'] + 17*MODELS.index(model):
                        raise ValueError('Paired seeds changed')
                    all_fold_rows.append({'arm': arm, 'model': model, 'scenario': sc['id'], 'fold': f['fold'], **metrics})
    fold_data = pd.DataFrame(all_fold_rows)
    expected = len(cfg['arms']) * len(MODELS) * len(scenarios) * len(folds)
    if len(fold_data) != expected:
        raise ValueError('CV incomplete')
    summaries = fold_data.groupby(['model', 'arm', 'scenario']).agg(f1_mean=('f1', 'mean'), f1_std=('f1', 'std'), epochs=('best_epoch', 'median')).reset_index()
    saved = pd.read_csv(run / 'reports/cv_summary.csv').sort_values(['model','arm','scenario']).reset_index(drop=True)
    check = summaries.sort_values(['model','arm','scenario']).reset_index(drop=True)
    if not np.allclose(saved[['f1_mean','f1_std','epochs']], check[['f1_mean','f1_std','epochs']]):
        raise ValueError('Saved CV summary differs')
    best = summaries.sort_values(['f1_mean','f1_std','arm','scenario'], ascending=[False,True,True,True]).groupby(['model','arm'], sort=False).head(1).sort_values(['model','arm'])
    best.to_csv(out / 'best_by_arm.csv', index=False)
    matched = fold_data.pivot(index=['model','scenario','fold'], columns='arm', values='f1').reset_index()
    matched['delta_weighted_minus_real'] = matched.weighted_real - matched.real_only
    matched.to_csv(out / 'paired_fold_comparison.csv', index=False)
    paired = matched.groupby('model').agg(mean_delta=('delta_weighted_minus_real','mean'), median_delta=('delta_weighted_minus_real','median'), wins=('delta_weighted_minus_real',lambda s:int((s>0).sum())), losses=('delta_weighted_minus_real',lambda s:int((s<0).sum()))).reset_index()
    paired.to_csv(out / 'weighting_effect_descriptive.csv', index=False)
    primary_rows, per_class_rows, seed_rows, binary_rows, train_rows, old_rows = [], [], [], [], [], []
    review = []
    stability_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, model in enumerate(MODELS):
        choices = [c for c in selection['choices'] if c['model'] == model]
        choice = choices[0]
        expected_choice = summaries[summaries.model == model].sort_values(['f1_mean','f1_std','arm','scenario'], ascending=[False,True,True,True]).iloc[0]
        if (choice['arm'], choice['scenario']) != (expected_choice.arm, expected_choice.scenario):
            raise ValueError('Selection differs from CV')
        parts = []
        for f in folds:
            d = run / 'cv' / choice['arm'] / model / choice['scenario'] / f'fold_{f["fold"]}'
            t = pd.read_csv(d / 'validation_predictions.csv', dtype={'hospital_id':str})
            t['validation_fold'] = f['fold']; parts.append(t)
        development = pd.concat(parts, ignore_index=True)
        if len(development) != len(dev) or development.hospital_id.duplicated().any():
            raise ValueError('OOF duplicated patients')
        development = development.merge(records[['hospital_id','sublocation']], on='hospital_id', validate='one_to_one')
        dp = development[[f'probability_{c}' for c in names]].to_numpy()
        development['predicted_label'] = dp.argmax(1)
        development['confidence'] = dp.max(1)
        development['is_error'] = development.true_label != development.predicted_label
        development.to_csv(out / f'development_oof_{model}.csv', index=False)
        oof[model] = development
        review.append(development[['hospital_id','sublocation','validation_fold','true_label','predicted_label','confidence','is_error']].assign(model=model))
        seed_predictions = []
        for seed in cfg['final_seeds']:
            d = run / 'test' / model / f'seed_{seed}'
            t, probs = read_predictions(d / 'patient_predictions.csv', names, test)
            if any(truth_map[hid] != label for hid, label in zip(t.hospital_id, t.true_label)):
                raise ValueError('Test labels changed')
            m = compute_metrics(t.true_label.to_numpy(), probs)
            saved_m = json.loads((d / 'metrics.json').read_text())['metrics']
            if not all(np.isclose(saved_m[k], v) for k,v in m.items()):
                raise ValueError('Test metrics differ')
            seed_rows.append({'model':model,'seed':seed,**m})
            btruth = (t.true_label.to_numpy() != 0).astype(int)
            # Fixed 0.5 threshold after summing both RVOT probabilities; descriptive secondary audit.
            bpred = (1-probs[:,0] >= .5).astype(int)
            binary_rows.append({'model':model,'seed':seed,'secondary_exploratory':True,'macro_f1_binary':f1_score(btruth,bpred,average='macro'),'accuracy_binary':accuracy_score(btruth,bpred),'auc_binary':roc_auc_score(btruth,1-probs[:,0])})
            t['predicted_label'] = probs.argmax(1)
            seed_predictions.append(t[['hospital_id','predicted_label']].set_index('hospital_id').rename(columns={'predicted_label':str(seed)}))
            final_case = json.loads((run / 'final' / model / f'seed_{seed}' / 'config.json').read_text())
            if (final_case['seed'], final_case['arm'], final_case['scenario']['id']) != (seed, choice['arm'], choice['scenario']):
                raise ValueError('Final case identity differs from selection')
            if set(final_case['train_patient_ids']) != dev:
                raise ValueError('Final model training membership differs')
            h = pd.read_csv(run / 'final' / model / f'seed_{seed}' / 'history.csv')
            if len(h) != choice['epochs']:
                raise ValueError('Final epochs differ from CV rule')
            train_rows.append({'model':model,'seed':seed,'epochs':len(h),'train_accuracy_segment_last':h.train_accuracy_segment.iloc[-1],'train_loss_last':h.train_loss.iloc[-1]})
            if seed == selection['primary_seed']:
                primary_rows.append({'model':model,'arm':choice['arm'],'scenario':choice['scenario'],**scenarios_by_id[choice['scenario']],'epochs':choice['epochs'],'cv_f1':choice['f1_mean'],'cv_sd':choice['f1_std'],'test_f1_seed42':m['f1'],'test_accuracy_seed42':m['accuracy'],'test_auc_seed42':m['auc']})
                for tag, truth, probabilities in [('development_OOF_selected',development.true_label.to_numpy(),dp),('historical_test_seed42',t.true_label.to_numpy(),probs)]:
                    cm = confusion_matrix(truth, probabilities.argmax(1), labels=[0,1,2])
                    ax = axes[0 if tag.startswith('development') else 1,j]
                    ax.imshow(cm, cmap='Blues', vmin=0)
                    for a in range(3):
                        for b in range(3): ax.text(b,a,str(cm[a,b]),ha='center',va='center')
                    ax.set_xticks(range(3),['LVOT','RVOT cusp','RVOT other'],rotation=15)
                    ax.set_yticks(range(3),['LVOT','RVOT cusp','RVOT other'])
                    ax.set_title(f'{model}: {"OOF selected" if tag.startswith("development") else "test seed 42"}')
                    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
                    report = classification_report(truth, probabilities.argmax(1), labels=[0,1,2], target_names=names, output_dict=True,zero_division=0)
                    for name in names:
                        per_class_rows.append({'model':model,'pool':tag,'class_name':name,**report[name]})
                old = json.loads((source / model / 'test/seed_42/metrics.json').read_text())['metrics']
                old_rows.append({'model':model,'historical_v1_f1':old['f1'],'v2_f1_seed42':m['f1'],'difference':m['f1']-old['f1'],'causal_GAN_effect':False})
        joined = pd.concat(seed_predictions,axis=1).sort_index()
        joined['unique_predictions_across_5_seeds'] = joined.astype(int).nunique(axis=1)
        joined['correct_in_5_seeds'] = [(row.iloc[:5].to_numpy().astype(int) == truth_map[hid]).sum() for hid,row in joined.iterrows()]
        joined.to_csv(out / f'test_seed_stability_{model}.csv')
        stability_rows.append({'model':model,'n_patients':len(joined),'prediction_changes_across_seeds':int((joined.unique_predictions_across_5_seeds>1).sum()),'wrong_all_5_seeds':int((joined.correct_in_5_seeds==0).sum())})
    fig.suptitle('Patient confusion matrices — selected OOF is selection optimistic; test is historical',fontsize=12)
    fig.tight_layout();fig.savefig(out/'confusion_oof_and_test.png',dpi=160);plt.close(fig)
    primary = pd.DataFrame(primary_rows)
    seeds = pd.DataFrame(seed_rows)
    variation = seeds.groupby('model').agg(seed_mean_f1=('f1','mean'),seed_sd_f1=('f1','std'),seed_min_f1=('f1','min'),seed_max_f1=('f1','max')).reset_index()
    summary = primary.merge(variation,on='model',validate='one_to_one')
    summary.to_csv(out/'model_summary.csv',index=False)
    pd.DataFrame(per_class_rows).to_csv(out/'per_class_diagnostics.csv',index=False)
    pd.DataFrame(binary_rows).to_csv(out/'secondary_binary_audit.csv',index=False)
    pd.DataFrame(train_rows).to_csv(out/'final_training_diagnostics.csv',index=False)
    pd.DataFrame(old_rows).to_csv(out/'historical_v1_vs_v2.csv',index=False)
    pd.DataFrame(stability_rows).to_csv(out/'seed_patient_stability_summary.csv',index=False)
    review_data = pd.concat(review,ignore_index=True)
    review_data.to_csv(out/'development_review_all_predictions.csv',index=False)
    hard = review_data.groupby('hospital_id').agg(sublocation=('sublocation','first'),true_label=('true_label','first'),models_wrong=('is_error','sum'),max_any_prediction_confidence=('confidence',lambda v:float(v.max()))).reset_index()
    wrong_confidence = review_data[review_data.is_error].groupby('hospital_id').confidence.max().rename('max_wrong_confidence')
    hard = hard.merge(wrong_confidence,on='hospital_id',how='left',validate='one_to_one')
    hard = hard.sort_values(['models_wrong','max_wrong_confidence'],ascending=False)
    hard.to_csv(out/'development_patient_review_priority.csv',index=False)
    subtype = review_data.groupby(['model','sublocation']).agg(n_patients=('is_error','size'),errors=('is_error','sum')).reset_index()
    subtype['error_rate'] = subtype.errors / subtype.n_patients
    subtype.to_csv(out/'development_errors_by_sublocation.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for i,m in enumerate(MODELS):
        d=matched[matched.model==m].groupby('scenario').delta_weighted_minus_real.mean()
        axes[i].bar(d.index,d.values,color=['#2d8b77' if x>=0 else '#be5c4f' for x in d.values]);axes[i].axhline(0,color='black',lw=.7)
        axes[i].set_title(m);axes[i].set_ylim(-.13,.06);axes[i].set_ylabel('Mean paired CV F1 difference');axes[i].tick_params(axis='x',rotation=45)
    fig.suptitle('Class weighting minus real only: same scenario, fold and seed; descriptive comparison')
    fig.tight_layout();fig.savefig(out/'class_weighting_matched.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for i,m in enumerate(MODELS):
        values=seeds[seeds.model==m].f1.to_numpy()
        ax.scatter(np.full(5,i)+np.linspace(-.10,.10,5),values,color='#527fb0')
        for x,y,seed in zip(np.full(5,i)+np.linspace(-.10,.10,5),values,cfg['final_seeds']):ax.annotate(str(seed),(x,y),xytext=(3,3),textcoords='offset points',fontsize=8)
        row=summary[summary.model==m].iloc[0];ax.errorbar(i,row.seed_mean_f1,yerr=row.seed_sd_f1,color='#222',fmt='s',capsize=5)
    ax.set_xticks(range(3),MODELS);ax.set_ylim(.4,.8);ax.set_ylabel('Historical test macro-F1');ax.set_title('Five fixed seeds; squares show mean ± training-seed SD')
    fig.tight_layout();fig.savefig(out/'seed_variation.png',dpi=160);plt.close(fig)
    audit={'date':'2026-10-01','cv_verified':len(all_fold_rows),'final_verified':len(train_rows),'test_verified':len(seed_rows),'development_patients':len(dev),'historical_test_patients':len(test),'frozen_input_sha256_verified':True,'metrics_recomputed':True,'test_used_for_selection':False,'development_wrong_all_3_models':int((hard.models_wrong==3).sum()),'model_selected_from_cv':selection['primary']['model']}
    (out/'analysis_audit.json').write_text(json.dumps(audit,indent=2))
    text = f'''# Analisis hasil V2 — 1 Oktober 2026

Audit run `{run_id}`: **{len(all_fold_rows)}/270 CV, {len(train_rows)}/15 final, {len(seed_rows)}/15 test lengkap**. Input frozen lolos checksum; metrik dihitung ulang dari prediksi per pasien. Tidak ada training/inferensi tambahan pada analisis ini.

## Hasil utama

{table_md(summary.drop(columns=['id']))}

Encoder tetap model utama dari development CV. Test memakai seed utama 42; mean/SD lima seed dilaporkan terpisah. Seed terbaik pada test tidak menggantikan seed utama. CI 95% seed utama: Encoder 0,514–0,751; PatchTST 0,467–0,671; Hierarchical 0,424–0,662 (bootstrap pasien yang telah dibuat notebook 09). Interval lebar dan beririsan; hasil ini tidak membuktikan perbedaan signifikan antar-model.

## Efek class weighting

{table_md(best)}

{table_md(paired)}

Best weighted-vs-best real: Encoder +0,0082, PatchTST +0,0081, Hierarchical −0,0083 macro-F1. Perbandingan ini juga memilih konfigurasi LR/batch berbeda. Analisis pasangan seluruh 45 kasus per model menunjukkan class weighting tidak konsisten membantu; pada PatchTST 31/45 pasangan turun. Kelima fold memiliki training set beririsan, sehingga 45 pasangan bukan 45 replikasi independen. Tidak dihitung p-value naif. Satu seed CV per kasus belum mengukur variasi seed pada proses seleksi.

## Sumber kesalahan

{table_md(pd.DataFrame(per_class_rows)[pd.DataFrame(per_class_rows).pool=='historical_test_seed42'][['model','class_name','recall','f1-score','support']])}

PatchTST hanya mendeteksi 4/21 pasien RVOT cusp pada test seed 42; 14/21 berpindah ke kelompok RVOT septal/free-wall/other. Pola ini sudah terlihat pada development OOF: recall RVOT cusp 50/103 = 48,5%. Hierarchical hanya benar 7/24 kelompok RVOT septal/free-wall/other pada test. Encoder lebih seimbang, tetapi OOF recall kelompok itu masih 37/70 = 52,9%.

Development OOF adalah prediksi pasien yang ditahan dari training fold, tetapi konfigurasi dipilih memakai fold yang sama: angka OOF bersifat diagnostik setelah seleksi. Pooled OOF F1 berbeda dari rata-rata F1 fold; gunakan rata-rata fold untuk aturan seleksi yang sudah dibekukan.

Ada **{audit['development_wrong_all_3_models']} pasien development salah pada ketiga model**. Daftar `development_patient_review_priority.csv` menyediakan antrean review; bukan perintah mengganti label. Confidence softmax belum dikalibrasi dan tidak membuktikan diagnosis/label salah. File `development_errors_by_sublocation.csv` perlu dibaca bersama jumlah pasien per sublokasi yang kecil.

## Training dan variasi seed

Encoder mencapai akurasi segmen training sekitar 90,3–92,4%; Hierarchical 88,6–90,3%, sementara akurasi pasien test lebih rendah. Unit evaluasinya berbeda sehingga selisih bukan ukuran overfitting formal, tetapi konsisten dengan kebutuhan memeriksa generalisasi. PatchTST training 62,1–65,8% dan loss akhir 0,676–0,730: kemungkinan resep/representasi/anggaran training belum cocok; belum terbukti bahwa menambah epoch akan memperbaiki test.

Walaupun SD macro-F1 PatchTST paling kecil, prediksi individu berubah pada **46/59 pasien** di antara lima seed (Encoder 31/59, Hierarchical 37/59). Variasi skor rendah tidak sama dengan keputusan pasien yang stabil. Ini hasil diagnostik test historis, bukan alasan memilih seed/ensemble dari test.

Best epoch CV sangat bervariasi: Encoder [10,1,16,2,12]; PatchTST [25,15,12,1,15]; Hierarchical [8,6,20,1,14]. Aturan median digunakan sesuai protokol, tetapi variasi ini perlu diuji melalui CV dengan seed tambahan atau pengaturan jadwal training pada run baru. Jangan mengganti jumlah epoch menurut skor test.

## Audit biner sekunder, eksploratif

Kedua kelompok RVOT dijumlahkan probabilitasnya lalu diprediksi menggunakan threshold tetap 0,5. Untuk PatchTST seed 42, LVOT-vs-RVOT macro-F1 **0,928**, accuracy **56/59 = 94,9%**, AUC **0,986**. Ini mendukung dugaan bahwa kesulitan besar berada pada pemisahan subkelompok RVOT. Analisis dilakukan setelah melihat hasil tiga kelas, bukan endpoint primer/preregistered. Target tiga kelas tetap; angka biner tidak menggantikan macro-F1 tiga kelas dan bukan bukti kelayakan jurnal.

## Perbandingan hasil historis

{table_md(pd.DataFrame(old_rows))}

Encoder nyaris sama dengan V1, PatchTST turun, Hierarchical naik sedikit. V1 memakai preprocessing lama dan GAN, V2 mengubah preprocessing dan augmentasi sekaligus; tidak boleh menyimpulkan perubahan disebabkan GAN saja atau fs saja. Nama anatomi dikoreksi tanpa perubahan label numerik. Test pernah dievaluasi sebelumnya, sehingga tetap hold-out historis; bukan validasi eksternal.

## Prioritas berikutnya dari bukti ini

1. **Review morfologi PVC dan kurasi segmen pada development.** Periksa pasien yang salah pada tiga model, kualitas sadapan, apakah segmen mengandung PVC, dan konsistensi target anatomi bersama ahli. Jangan otomatis mengubah diagnosis. [Paper sumber](https://pmc.ncbi.nlm.nih.gov/articles/PMC8962834/) memakai jendela sekitar R peak PVC; project memakai potongan 2,5 detik yang belum teranotasi PVC. Detektor R peak saja belum membedakan beat sinus dan PVC.
2. **Uji masukan berpusat pada PVC dan kontribusi pasien yang seimbang.** Tetapkan panjang jendela/aturan agregasi/penanganan segmen pendek sebelum eksperimen, pilih melalui CV development. Sampling saat ini memberi pasien rekaman panjang lebih banyak kontribusi segmen; class weighting belum menyelesaikan pembobotan antar-pasien.
3. **Perbaiki khusus batas dua kelas RVOT.** Bandingkan baseline morfologi/beat dan classifier dua tahap LVOT-vs-RVOT lalu subkelompok RVOT pada run baru; validasi seluruh rangkaian, bukan dua tahap terpisah. Tiga kelas primer dipertahankan. Ambang/kalibrasi harus dipelajari pada validation development, tidak pada test.
4. **Uji resep PatchTST dan stabilitas CV.** Ukur ulang seed pada CV, bobot kelas alternatif/real-only, scheduler, pooling dan kapasitas sesuai anggaran yang ditetapkan. Jangan hanya memilih LR/batch/seed baru yang kebetulan memperbaiki 59 test ini.
5. **GAN sebagai eksperimen tambahan yang lolos QA.** Generator kanal independen, QA morfologi/spektrum/relasi sadapan/keragaman pada training saja. Bandingkan dengan real dan penanganan imbalance pada preprocessing sama. Hasil V2 tidak mendukung klaim GAN wajib dipakai atau GAN lama penyebab tunggal penurunan.
6. **Validasi pasien baru dan kontribusi ilmiah.** F1 tiga kelas masih terbatas, interval lebar, dan test historis. Untuk target publikasi kuat, dokumentasikan kontribusi, dataset/split, ketidakpastian, keterbatasan, serta evaluasi eksternal/prospektif bila tersedia. Status saat ini belum cukup untuk menyatakan model siap dipakai klinis atau dijamin terbit Q1.

Grafik: `confusion_oof_and_test.png`, `class_weighting_matched.png`, `seed_variation.png`. Sumber tabel adalah artefak run yang telah Anda jalankan; kode analisis tersedia pada `analysis_v2/analyze_results.py`.
'''
    (out/'ANALISIS_V2.md').write_text(text)
    print(summary.to_string(index=False))
    print('Audit:', audit)
    print('Report:',out/'ANALISIS_V2.md')
    return out


if __name__ == '__main__':
    analyze()
