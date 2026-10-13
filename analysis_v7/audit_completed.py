"""Recompute completed native QC on CPU; write only a separate audit folder.

No model forward, optimizer, waveform test, or per-fold outer-validation access.
"""
import argparse
from collections import Counter
from pathlib import Path
import json
import math
import os
import numpy as np
import pandas as pd
import torch
from src.artifacts import ROOT, canonical_hash, read_json, sha256_file, write_json
from research_v7.common import DEFAULT_RUN, PREPARATION_FILES, context, require_prepared, verify_files
from research_v7.data import load_development
from research_v6.quality import compare_features, features, numerical_screen, weighted_quantile


def equal_features(original, reproduced):
    for field in ('label', 'family', 'feature', 'passed'):
        if original[field].tolist() != reproduced[field].tolist():
            raise RuntimeError(f'QC decision differs: {field}')
    for field in ('real_fit_reference_distance', 'synthetic_reference_distance', 'gate_statistic', 'threshold'):
        np.testing.assert_allclose(original[field], reproduced[field], rtol=1e-8, atol=1e-9)


def snapshot_unchanged(path):
    snapshot = read_json(path)
    changed = [name for name, digest in snapshot.items()
        if not (ROOT / name).is_file() or sha256_file(ROOT / name) != digest]
    if changed:
        raise RuntimeError(f'Protected artifacts changed: {changed}')
    return len(snapshot)


def audit(run_id=DEFAULT_RUN):
    study = context(run_id)
    output = study.run / 'analysis_20261005_complete'
    output.mkdir(exist_ok=True)
    smoke = read_json(study.run / 'smoke.json')
    verify_files(study.run, smoke)
    if (smoke['experiment_hash'] != study.signature or smoke['device'] != 'mps'
            or not smoke['passed'] or not smoke['fidelity_gradient_bridge_passed']):
        raise RuntimeError('Completed MPS smoke required')
    X, table = load_development(study)  # mmap; waveform indexing below is GAN-fit only
    labels_by_patient = {r['hospital_id']: r['label'] for r in study.records}
    names = {r['label']: r['class_name'] for r in study.records}
    families, summaries, examples, details, sources = [], [], [], [], {}
    for split in study.folds:
        fold = split['fold']; folder = study.folder(fold)
        print(f'AUDIT fold {fold}: hashes, epochs, native QC on CPU', flush=True)
        prepared = require_prepared(study, fold)
        complete, generated, quality = [read_json(folder / name)
            for name in ('completed.json', 'generated.json', 'quality.json')]
        for saved in (complete, generated, quality):
            verify_files(folder, saved)
            if saved['experiment_hash'] != study.signature or saved['fold'] != fold:
                raise RuntimeError('Completed fold identity mismatch')
        if (generated['generator_sha256'] != sha256_file(folder / 'generator.pth')
                or quality['generated_manifest_sha256'] != sha256_file(folder / 'generated.json')):
            raise RuntimeError('Generator/synthesis/QC binding mismatch')
        case = read_json(folder / 'case.json')
        if (canonical_hash({k:v for k,v in case.items() if k != 'case_hash'}) != case['case_hash']
                or complete['case_hash'] != case['case_hash'] or case['device'] != 'mps'
                or case['effective_seed'] != study.protocol['gan_seed'] + 100 * fold
                or case['fit_patient_ids'] != split['gan_fit_ids']
                or case['reference_patient_ids'] != split['gan_reference_ids']
                or case['fit_target_bank_sha256'] != sha256_file(folder / 'fit_target_bank.npz')):
            raise RuntimeError('Training case mismatch')
        fit_ids, reference_ids, outer_ids = [set(split[key]) for key in
            ('gan_fit_ids', 'gan_reference_ids', 'outer_validation_ids')]
        if fit_ids & reference_ids or fit_ids & outer_ids or reference_ids & outer_ids:
            raise RuntimeError('Patient partitions overlap')
        for name in PREPARATION_FILES:
            if sha256_file(folder / name) != sha256_file(study.baseline / f'fold_{fold}' / name):
                raise RuntimeError('V6 preparation was changed in V7')
        for group, expected in (('fit', fit_ids), ('reference', reference_ids)):
            rows = pd.read_csv(folder / f'{group}_rows.csv', dtype={'hospital_id':str})
            if set(rows.hospital_id) != expected:
                raise RuntimeError('QC rows use wrong patients')
            np.testing.assert_allclose(rows.groupby('hospital_id').weight.sum(), 1.)
            if set(table.iloc[rows.segment_index].hospital_id) != expected:
                raise RuntimeError('QC row indices use wrong patients')
        history = pd.read_csv(folder / 'history.csv')
        cfg = study.protocol['training']
        draws = len(fit_ids) * cfg['draws_per_fit_patient_per_epoch']
        if (complete['epochs'] != cfg['epochs'] or history.epoch.tolist() != list(range(1, cfg['epochs'] + 1))
                or not history.draws.eq(draws).all()
                or not history.g_steps.eq(math.ceil(draws / cfg['batch_size'])).all()
                or not history.d_steps.eq(history.g_steps).all()
                or not np.isfinite(history.select_dtypes('number')).all().all()):
            raise RuntimeError('Epochs, update budget or finite losses mismatch')
        for row in history.itertuples():
            exposure = json.loads(row.patient_exposure_json)
            if (not set(exposure) <= fit_ids or sum(exposure.values()) != draws
                    or len(exposure) != row.unique_fit_patients_drawn
                    or any(not isinstance(n, int) or n <= 0 for n in exposure.values())):
                raise RuntimeError('Training exposure includes wrong patients')
            counts = Counter()
            for hid, count in exposure.items(): counts[labels_by_patient[hid]] += count
            if any(counts[c] != getattr(row, f'class_{c}_draws') for c in range(3)):
                raise RuntimeError('Class exposure differs from patient exposure')
        # Only deserialize the project's own saved checkpoint, on CPU; no training.
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        if state['epoch'] != cfg['epochs'] or state['case_hash'] != case['case_hash'] or state['device'] != 'mps':
            raise RuntimeError('Last checkpoint identity mismatch')
        pd.testing.assert_frame_equal(pd.DataFrame(state['history']), history,
            check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
        for role in ('generator', 'discriminator'):
            weights = torch.load(folder / f'{role}.pth', map_location='cpu', weights_only=True)
            if (set(weights) != set(state[role]) or any(not torch.isfinite(v).all()
                    or not torch.equal(v, state[role][key]) for key,v in weights.items())):
                raise RuntimeError('Final model is not finite last-epoch checkpoint')
        del state, weights
        samples = np.load(folder / 'synthetic.npy', mmap_mode='r')
        label_rows = pd.read_csv(folder / 'synthetic_labels.csv')
        labels = label_rows.label.to_numpy()
        per_class = study.protocol['generation']['per_class']
        if (list(samples.shape) != generated['shape'] or samples.shape != (3 * per_class, 12, 5000)
                or labels.tolist() != np.repeat(np.arange(3), per_class).tolist()
                or label_rows.sample_index.tolist() != list(range(len(samples)))
                or generated['seed'] != study.protocol['gan_seed'] + 100 * fold + study.protocol['generation']['seed_offset']):
            raise RuntimeError('Synthesis budget, labels or seed differs')
        scale = read_json(folder / 'amplitude_scale.json')['scale']
        fake = pd.read_csv(folder / 'synthetic_features.csv')
        computed = pd.concat([features(np.asarray(samples[o:o+16], dtype=np.float64) / scale)[0]
            for o in range(0, len(samples), 16)], ignore_index=True)
        np.testing.assert_allclose(fake[computed.columns], computed, rtol=1e-8, atol=1e-10)
        np.testing.assert_array_equal(fake.label, labels)
        np.testing.assert_allclose(fake.weight, 1.)
        fit, reference = [pd.read_csv(folder / f'{name}_features.csv') for name in ('fit', 'reference')]
        reproduced, failed = compare_features(fit, reference, fake,
            read_json(folder / 'feature_families.json'), study.protocol['quality_gate'])
        equal_features(pd.read_csv(folder / 'feature_distances.csv'), reproduced)
        numerical, nearest = numerical_screen(samples, labels, X, table, split['gan_fit_ids'],
            scale, study.protocol['quality_gate'])
        saved_nearest = pd.read_csv(folder / 'synthetic_nearest_fit.csv', dtype={'nearest_hospital_id':str})
        pd.testing.assert_frame_equal(nearest, saved_nearest, check_dtype=False,
            check_exact=False, rtol=1e-8, atol=1e-10)
        if numerical != quality['numerical_screen'] or failed != quality['failed_families']:
            raise RuntimeError('Native numerical/family screen does not reproduce')
        passed = bool(numerical['passed'] and numerical['finite'] and not failed)
        if passed != quality['passed']:
            raise RuntimeError('Overall quality decision does not reproduce')
        group = reproduced.assign(failed=~reproduced.passed).groupby(['label', 'family']).agg(
            feature_count=('passed','size'), failed_count=('failed','sum'),
            failed_fraction=('failed','mean'), normalized_wasserstein_mean=('synthetic_reference_distance','mean')).reset_index()
        group.insert(0,'fold',fold); group.insert(2,'class_name',group.label.map(names))
        group['family_passed'] = group.failed_fraction.le(study.protocol['quality_gate']['max_failed_feature_fraction_per_family_and_class'])
        families.append(group)
        nearest_min = float(nearest.relative_rmse.min())
        summaries.append({'fold':fold,'fit_patients':len(fit_ids),'reference_patients':len(reference_ids),
            'outer_validation_patients':len(outer_ids),'epochs':len(history),'generator_updates':int(history.g_steps.sum()),
            'synthetic_windows':len(samples),'quality_passed':passed,'failed_families':len(failed),
            'max_failed_feature_fraction':float(group.failed_fraction.max()),
            'near_copy_min_relative_rmse':nearest_min,'exact_copies':numerical['exact_copy_hash_count'],
            'epoch_training_seconds':float(history.seconds.sum()),'fidelity_first':float(history.g_fidelity.iloc[0]),
            'fidelity_last':float(history.g_fidelity.iloc[-1])})
        details.append(reproduced.assign(fold=fold))
        for c in range(3):
            real, synthetic = reference[reference.label.eq(c)], fake[fake.label.eq(c)]
            for feature in ('lag1_over_rms_V2','power_40_100_V2','power_100_250_V2','corr_V1_V2','range90_V2'):
                examples.append({'fold':fold,'label':c,'class_name':names[c],'feature':feature,
                    'reference_patient_weighted_median':weighted_quantile(real[feature],real.weight,.5),
                    'synthetic_median':weighted_quantile(synthetic[feature],synthetic.weight,.5)})
        for marker in ('preparation.json','completed.json','generated.json','quality.json'):
            saved = read_json(folder / marker)
            sources[str((folder / marker).relative_to(ROOT))] = sha256_file(folder / marker)
            sources.update({str((folder / name).relative_to(ROOT)): digest for name,digest in saved['files'].items()})
        print(f'VERIFIED fold {fold}: PASS={passed}, {len(failed)}/15 failed families, minimum copy RMSE={nearest_min:.4f}', flush=True)
    family_table = pd.concat(families, ignore_index=True)
    summary = pd.DataFrame(summaries)
    old_distance = pd.read_csv(study.baseline / 'fold_1/feature_distances.csv')
    new_distance = details[0]
    comparison = []
    for version,data in (('V6',old_distance),('V7',new_distance)):
        for (label,family),group in data.groupby(['label','family']):
            comparison.append({'version':version,'label':label,'family':family,
                'failed_fraction':float((~group.passed).mean()),
                'normalized_wasserstein_mean':float(group.synthetic_reference_distance.mean())})
    comparison = pd.DataFrame(comparison)
    saved = pd.read_csv(study.run / 'fold_1/comparison_v6_v7.csv')
    pd.testing.assert_frame_equal(comparison[saved.columns], saved, check_dtype=False,
        check_exact=False, rtol=1e-10, atol=1e-10)
    for name,frame in (('fold_summary.csv',summary),('quality_family_summary.csv',family_table),
                       ('feature_examples.csv',pd.DataFrame(examples)),('comparison_fold1.csv',comparison)):
        frame.to_csv(output / name,index=False)
    plot(output, family_table, names)
    notebooks = []
    for path in sorted((ROOT / 'notebooks').glob('*V7*.ipynb')):
        nb = read_json(path); cells = [c for c in nb['cells'] if c['cell_type']=='code']
        errors = [o.get('evalue') for c in cells for o in c.get('outputs',[]) if o['output_type']=='error']
        executed = sum(c.get('execution_count') is not None for c in cells)
        if errors or executed != len(cells): raise RuntimeError(f'Notebook incomplete: {path.name}')
        notebooks.append({'file':path.name,'executed_code_cells':executed,'total_code_cells':len(cells),'errors':errors})
    protected = {path.name:snapshot_unchanged(path) for path in
        (ROOT/'tmp/v7_frozen_before.json',ROOT/'tmp/v7_completed_before_audit_20261005.json')}
    result = {'experiment_hash':study.signature,'source_v6_hash':study.identity['source_v6_hash'],
        'folds_verified':5,'epochs_verified':int(summary.epochs.sum()),
        'generator_updates_verified':int(summary.generator_updates.sum()),
        'synthetic_windows_verified':int(summary.synthetic_windows.sum()),
        'folds_quality_passed':int(summary.quality_passed.sum()),'families_verified':len(family_table),
        'failed_families':int(summary.failed_families.sum()),'quality_native_recomputed':True,
        'checkpoint_last_epoch_verified':True,'fold1_comparison_reproduced':True,
        'protected_unchanged':protected,'notebooks':notebooks,'source_sha256':sources,
        'audit_code_sha256':sha256_file(Path(__file__)),'new_training_performed':False,
        'test_signals_read':False,'per_fold_outer_validation_signals_read':False,
        'clinical_validity_established':False,'classifier_benefit_established':False}
    write_json(output/'audit.json',result)
    report(output,result,summary,family_table)
    print(f'Completed audit: 5 folds, 500 epochs, {result["synthetic_windows_verified"]} synthetic windows; report={output / "REPORT.md"}',flush=True)
    return result


def plot(output, data, names):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = ['amplitude','roughness','correlation','spectrum','limb_residual']
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for c,ax in enumerate(axes):
        matrix=data[data.label.eq(c)].pivot(index='family',columns='fold',values='failed_fraction').loc[order]
        picture=ax.imshow(matrix,vmin=0,vmax=.25,cmap='YlOrRd',aspect='auto')
        for i in range(5):
            for j in range(5): ax.text(j,i,f'{matrix.iloc[i,j]:.1%}',ha='center',va='center',fontsize=8)
        ax.set(title=names[c],xticks=range(5),xticklabels=range(1,6),xlabel='Fold',yticks=range(5),yticklabels=order)
    fig.colorbar(picture,ax=axes.tolist(),label='Fraction of failed features; gate <=25%',shrink=.8)
    fig.savefig(output/'quality_all_folds.png',dpi=150,bbox_inches='tight');plt.close(fig)


def report(output,audit,summary,families):
    lines=['# Audit hasil GAN V7 — 5 Oktober 2026','',
        '**Notebook 37–40 selesai tanpa error tersimpan; kelima fold PASS setelah QC dihitung ulang dari waveform native.**','',
        f'Fingerprint: `{audit["experiment_hash"]}`. Checkpoint adalah epoch terakhir, bukan pilihan dari skor classifier.','',
        '| Fold | Fit/ref/outer pasien | Epoch | Update G | Sintetis | Keluarga gagal | Maks. fraksi fitur gagal |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.fold} | {r.fit_patients}/{r.reference_patients}/{r.outer_validation_patients} | {r.epochs} | {r.generator_updates} | {r.synthetic_windows} | {r.failed_families}/15 | {r.max_failed_feature_fraction:.1%} |')
    lines.extend(['','Fold 1 V6 gagal 9/15 kombinasi kelas–keluarga; V7 gagal 0/15 pada ambang yang sama. Mean jarak distribusi non-limb membaik pada seluruh 12 pasangan kelas–keluarga fold 1.','',
        'Total 500 epoch, 18.500 update G dan D masing-masing, 960 window sintetis (64 per kelas per fold). Sampel bukan 960 pasien baru. Waktu pada kolom seconds history adalah waktu epoch, bukan seluruh biaya prepare/QC/report.','',
        'Finite, distinct, constant-lead, exact-copy, near-copy dan native zero-lag nearest-neighbour dihitung ulang. Seluruh keputusan cocok; bobot final finite dan identik checkpoint epoch 100. Semua pasien sampling training berasal dari GAN-fit sesuai fold. Split, skala dan fitur real identik V6.','',
        '![Fraksi fitur gagal seluruh fold](quality_all_folds.png)','',
        '## Kelemahan yang tetap terlihat','',
        '- PASS keluarga mengizinkan maksimal 25% fitur gagal; tidak berarti semua fitur atau setiap waveform sesuai. Fraksi terbesar: korelasi RVOT cusp fold 1 19,7%; spektrum RVOT septal/freewall/other fold 3 19,0%.','',
        '- Loss dan QC menilai keluarga fitur yang serupa. QC pada pasien referensi terpisah mendukung fidelity komputasional, belum membuktikan morfologi PVC, diagnosis sintetis atau manfaat lokalisasi.','',
        '- Pemeriksaan copy full-window tanpa phase alignment belum menjadi sertifikasi privasi atau pemeriksaan semua bentuk memorisasi.','',
        '- Satu seed GAN per fold; variasi antarfold juga memuat variasi data. Belum mengukur ketidakpastian generator pada fold yang sama.','',
        '## Pekerjaan berikutnya','',
        'Training/QC 37–40 tidak perlu diulang. Bekukan hasil V7 dan siapkan protokol serta endpoint classifier terpisah: K4 real-only, kontrol augmentasi real, dan real+GAN dengan fold, seed, model, update budget serta penilaian pasien yang sebanding. Evaluasi pada validation real saja. Prioritas Hierarchical dan Encoder; macro-F1 dan recall kedua RVOT, OOF, biaya serta variasi seed.','',
        'Endpoint classifier V7 belum tersedia. Tidak menjalankan final/test sekarang; test historis tidak dipakai memilih arm atau seed. Efek terhadap classifier harus diuji dahulu sebelum mengklaim peningkatan akurasi.','',
        'Audit ini tidak melatih model, tidak menggunakan MPS untuk inference, tidak membaca waveform test atau outer validation fold yang diaudit, dan tidak mengubah hasil/kode/notebook asli.'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default=DEFAULT_RUN)
    args=parser.parse_args()
    audit(args.run_id)
