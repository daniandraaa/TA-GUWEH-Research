"""Verify completed pilot provenance and reproduce its statistical quality decision."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.artifacts import ROOT, read_json, sha256_file, write_json
from research_v6.common import DEFAULT_RUN, context, require_prepared, verify_files
from research_v6.quality import compare_features, weighted_quantile


def diagnose(run_id=DEFAULT_RUN, fold=1):
    study = context(run_id)
    directory = study.folder(fold)
    require_prepared(study, fold)
    completed, generated, quality = [read_json(directory / name)
        for name in ('completed.json', 'generated.json', 'quality.json')]
    for saved in (completed, generated, quality):
        verify_files(directory, saved)
        if saved['experiment_hash'] != study.signature:
            raise RuntimeError('Pilot provenance mismatch')
    if (generated['generator_sha256'] != sha256_file(directory / 'generator.pth')
            or quality['generated_manifest_sha256'] != sha256_file(directory / 'generated.json')):
        raise RuntimeError('Generator/sample/QC source changed')
    fit, reference, fake = [pd.read_csv(directory / name)
        for name in ('fit_features.csv', 'reference_features.csv', 'synthetic_features.csv')]
    original = pd.read_csv(directory / 'feature_distances.csv')
    reproduced, failed = compare_features(fit, reference, fake,
        read_json(directory / 'feature_families.json'), study.protocol['quality_gate'])
    for field in ('label', 'family', 'feature', 'passed'):
        if original[field].tolist() != reproduced[field].tolist():
            raise RuntimeError(f'Recomputed quality differs: {field}')
    for field in ('real_fit_reference_distance', 'synthetic_reference_distance', 'gate_statistic', 'threshold'):
        np.testing.assert_allclose(original[field], reproduced[field], rtol=1e-8, atol=1e-9)
    if failed != quality['failed_families']:
        raise RuntimeError('Failed-family decision does not reproduce')
    numerical = quality['numerical_screen']
    passed = bool(numerical['passed'] and numerical['finite'] and not failed)
    if passed != quality['passed']:
        raise RuntimeError('Overall gate decision does not reproduce')
    history = pd.read_csv(directory / 'history.csv')
    if history.epoch.tolist() != list(range(1, completed['epochs'] + 1)):
        raise RuntimeError('Incomplete/nonconsecutive GAN epochs')
    if generated['shape'] != [len(fake), 12, 5000] or fake.groupby('label').size().tolist() != [64, 64, 64]:
        raise RuntimeError('Unexpected synthesis budget')
    source_names = ['preparation.json', 'completed.json', 'generated.json', 'quality.json',
        'case.json', 'history.csv', 'feature_distances.csv', 'reference_features.csv', 'synthetic_features.csv',
        'synthetic_nearest_fit.csv', 'reference_nearest_fit.csv']
    sources = {str((directory / name).relative_to(ROOT)): sha256_file(directory / name) for name in source_names}
    summary = reproduced.assign(failed=lambda x: ~x.passed).groupby(['label', 'family']).agg(
        feature_count=('passed', 'size'), failed_count=('failed', 'sum'), failed_fraction=('failed', 'mean')).reset_index()
    summary['family_passed'] = summary.failed_fraction.le(study.protocol['quality_gate']['max_failed_feature_fraction_per_family_and_class'])
    names = {r['label']: r['class_name'] for r in study.records}
    summary.insert(1, 'class_name', summary.label.map(names))
    examples = []
    for c in range(3):
        a, b = reference[reference.label.eq(c)], fake[fake.label.eq(c)]
        for feature in ('lag1_over_rms_V2', 'power_40_100_V2', 'power_100_250_V2', 'corr_V1_V2', 'range90_V2'):
            real = weighted_quantile(a[feature], a.weight, .5)
            synthetic = weighted_quantile(b[feature], b.weight, .5)
            examples.append({'label': c, 'class_name': names[c], 'feature': feature,
                'reference_patient_weighted_median': real, 'synthetic_median': synthetic})
    output = study.run / f'analysis_20261004_pilot_fold{fold}'
    output.mkdir(exist_ok=True)
    summary.to_csv(output / 'quality_family_summary.csv', index=False)
    pd.DataFrame(examples).to_csv(output / 'feature_examples.csv', index=False)
    audit = {'experiment_hash': study.signature, 'fold': fold, 'epochs_verified': len(history),
        'generator_updates': int(history.g_steps.sum()), 'synthetic_windows': len(fake),
        'quality_decision_reproduced': True, 'passed': passed, 'failed_families': len(failed),
        'families_evaluated': len(summary), 'source_sha256': sources,
        'test_signals_read': False, 'outer_validation_signals_read': False,
        'new_training_performed': False,
        'diagnosis_code_sha256': sha256_file(Path(__file__))}
    write_json(output / 'audit.json', audit)
    lines = [f'# Diagnosis pilot V6 fold {fold}', '',
        f'Training selesai {len(history)} epoch; {int(history.g_steps.sum())} update generator; {len(fake)} sintetis.', '',
        f'Engineering gate: **{"PASS" if passed else "FAIL"}**; {len(failed)}/{len(summary)} kombinasi kelas–keluarga fitur gagal.', '',
        'Keputusan QC dihitung ulang dari fitur tersimpan dan cocok. Hash preparation/checkpoint/sintesis/QC valid.', '',
        'Notebook36 dihentikan oleh require_quality karena passed=false. Ini kegagalan screening pilot, bukan crash training/device.', '',
        '| Kelas | Keluarga | Fitur gagal | Fraksi gagal | Lolos keluarga |', '|---|---|---:|---:|---|']
    for row in summary.itertuples():
        lines.append(f'| {row.class_name} | {row.family} | {row.failed_count}/{row.feature_count} | {row.failed_fraction:.1%} | {row.family_passed} |')
    lines.extend(['', 'Ambang: maksimal25% fitur gagal perkeluarga/perkelas. Relasi limb lead, finite, duplicate dan near-copy screening lolos; bukan bukti validitas klinis/privasi.', '',
        'Langkah berikut: perbaiki fidelity spektrum/roughness, korelasi antar-lead dan distribusi amplitudo pada run baru, lalu ulang pilot fold1. Jangan mengubah gate atau melanjutkan fold2–5 dengan generator ini.', '',
        'Seluruh kode, notebook eksekusi, checkpoint dan tabel V6 asli dipertahankan. Tidak ada training baru atau akses waveform test/outer-validation.'])
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(f'Pilot diagnosis verified: epochs={len(history)}, synthetic={len(fake)}, gate={passed}, failed_families={len(failed)}/{len(summary)}', flush=True)
    print(f'Report: {output / "REPORT.md"}', flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    diagnose(args.run_id, args.fold)
