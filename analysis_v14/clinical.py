"""Pembanding klinis: kriteria EKG standar LVOT vs RVOT, dihitung otomatis tanpa training.

Ambang diambil dari literatur dan dikunci di PRAREGISTRASI_PEMBANDING_KLINIS_V14 sebelum dihitung:
- Zona transisi prekordial (TZ) beat PVC: <= V2 -> LVOT, >= V4 -> RVOT, V3 tidak tegas.
- Indeks V2S/V3R (Yoshida dkk., 2014): <= 1,5 -> LVOT.
- Rasio transisi V2 (Betensky dkk., 2011): [R/(R+S)] V2 PVC / [R/(R+S)] V2 sinus >= 0,6 -> LVOT.
Beat PVC = kandidat QRS dengan rentang QRS >= 120 ms; sinus = < 120 ms. Pengukuran memakai
research_v11.morphology (belum divalidasi ahli). Kriteria tidak punya parameter yang dilatih.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from src.artifacts import ROOT, read_json, sha256_file, write_json
from research_v11.common import LEADS
from research_v11.morphology import measure_record
from research_v13.common import context as v13_context, cv_directory, test_directory

PACKAGE = Path(__file__).resolve().parent
PREREGISTRATION = ROOT / 'docs/revisi/PRAREGISTRASI_PEMBANDING_KLINIS_V14_20261008.md'
OUT = ROOT / 'results/research_v14/clinical_criteria_v14'
PVC_MIN_QRS_MS = 120.0
TZ_LVOT_MAX, TZ_RVOT_MIN = 2, 4
V2S_V3R_LVOT_MAX = 1.5
V2_TRANSITION_RATIO_LVOT_MIN = 0.6
P13 = ['probability_LVOT', 'probability_RVOT_cusp', 'probability_RVOT_septal_freewall_other']


def code_hashes():
    return {p.name: sha256_file(p) for p in sorted(PACKAGE.glob('*.py'))}


def require_preregistered():
    text = PREREGISTRATION.read_text()
    for name, digest in code_hashes().items():
        if f'{name}: {digest}' not in text:
            raise RuntimeError(f'Kode {name} berubah setelah pra-registrasi')


def patient_measures(raw, fs, params):
    _, quality, beats, _ = measure_record(raw, fs, params)
    beats = pd.DataFrame(beats)
    empty = {'n_beats': len(beats), 'n_pvc_beats': 0, 'n_sinus_beats': 0, 'tz': np.nan, 'v2s_v3r': np.nan,
             'v2_r_fraction_pvc': np.nan, 'v2_r_fraction_sinus': np.nan, 'v2_transition_ratio': np.nan}
    if beats.empty or 'robust_qrs_span_ms' not in beats:
        return empty
    width = beats.robust_qrs_span_ms
    pvc, sinus = beats[width >= PVC_MIN_QRS_MS], beats[width < PVC_MIN_QRS_MS]
    row = {**empty, 'n_pvc_beats': len(pvc), 'n_sinus_beats': len(sinus)}
    if len(pvc):
        row['tz'] = float(np.nanmedian(pvc.first_r_dominant_v_index)) if pvc.first_r_dominant_v_index.notna().any() else np.nan
        row['v2s_v3r'] = float(np.nanmedian(pvc.v2_s_over_v3_r)) if pvc.v2_s_over_v3_r.notna().any() else np.nan
        row['v2_r_fraction_pvc'] = float(np.nanmedian(pvc.V2_r_fraction)) if pvc.V2_r_fraction.notna().any() else np.nan
    if len(sinus) and sinus.V2_r_fraction.notna().any():
        row['v2_r_fraction_sinus'] = float(np.nanmedian(sinus.V2_r_fraction))
    if np.isfinite(row['v2_r_fraction_pvc']) and np.isfinite(row['v2_r_fraction_sinus']) and row['v2_r_fraction_sinus'] > 0:
        row['v2_transition_ratio'] = row['v2_r_fraction_pvc'] / row['v2_r_fraction_sinus']
    return row


def rule_calls(m):
    """Keputusan per kriteria: 0=LVOT, 1=RVOT, NaN=tidak dapat diputuskan; skor = makin besar makin RVOT."""
    tz = m.tz
    calls = pd.DataFrame(index=m.index)
    calls['tz_call'] = np.where(tz <= TZ_LVOT_MAX, 0., np.where(tz >= TZ_RVOT_MIN, 1., np.nan))
    calls.loc[tz.isna(), 'tz_call'] = np.nan
    calls['tz_score'] = tz
    calls['v2s_v3r_call'] = np.where(m.v2s_v3r.isna(), np.nan, (m.v2s_v3r > V2S_V3R_LVOT_MAX).astype(float))
    calls['v2s_v3r_score'] = m.v2s_v3r
    calls['v2tr_call'] = np.where(m.v2_transition_ratio.isna(), np.nan,
                                  (m.v2_transition_ratio < V2_TRANSITION_RATIO_LVOT_MIN).astype(float))
    calls['v2tr_score'] = -m.v2_transition_ratio
    return calls


def scores(truth, call, score):
    """Dua cara: (a) cakupan = hanya pasien yang dapat diputuskan; (b) paksa = tak dapat diputuskan dianggap RVOT."""
    truth, call, score = np.asarray(truth), np.asarray(call, float), np.asarray(score, float)
    covered = np.isfinite(call)
    forced = np.where(covered, call, 1.).astype(int)
    auc_mask = np.isfinite(score)
    out = {'coverage': float(covered.mean()),
           'forced_macro_f1': float(f1_score(truth, forced, average='macro', labels=[0, 1], zero_division=0)),
           'forced_balanced_accuracy': float(balanced_accuracy_score(truth, forced)),
           'forced_sensitivity_LVOT': float(recall_score(truth, forced, pos_label=0, zero_division=0)),
           'covered_macro_f1': None, 'covered_sensitivity_LVOT': None,
           'auc_on_measurable': float(roc_auc_score(truth[auc_mask], score[auc_mask]))
           if auc_mask.any() and len(set(truth[auc_mask])) == 2 else None, 'n_measurable': int(auc_mask.sum())}
    if covered.any() and len(set(truth[covered])) == 2:
        out['covered_macro_f1'] = float(f1_score(truth[covered], call[covered].astype(int), average='macro', labels=[0, 1]))
        out['covered_sensitivity_LVOT'] = float(recall_score(truth[covered], call[covered].astype(int), pos_label=0))
    return out


def model_scores(truth, p_rvot):
    pred = (np.asarray(p_rvot) > .5).astype(int)
    return {'coverage': 1., 'forced_macro_f1': float(f1_score(truth, pred, average='macro', labels=[0, 1])),
            'forced_balanced_accuracy': float(balanced_accuracy_score(truth, pred)),
            'forced_sensitivity_LVOT': float(recall_score(truth, pred, pos_label=0)),
            'auc_on_measurable': float(roc_auc_score(truth, p_rvot)), 'n_measurable': len(truth)}


def paired_auc_bootstrap(truth, a, b, resamples=5000, seed=20261008):
    """Selisih AUC (b - a) pada pasien yang terukur kedua skornya, bootstrap terstratifikasi."""
    truth, a, b = map(np.asarray, (truth, a, b)); mask = np.isfinite(a) & np.isfinite(b)
    truth, a, b = truth[mask], a[mask], b[mask]; rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(truth == c) for c in (0, 1)]
    observed = roc_auc_score(truth, b) - roc_auc_score(truth, a); draws = []
    for _ in range(resamples):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        draws.append(roc_auc_score(truth[idx], b[idx]) - roc_auc_score(truth[idx], a[idx]))
    draws = np.asarray(draws)
    return {'n': int(mask.sum()), 'delta_auc': float(observed), 'ci95_low': float(np.quantile(draws, .025)),
            'ci95_high': float(np.quantile(draws, .975)), 'p_two_sided': float(min(1., 2 * min((draws <= 0).mean(), (draws >= 0).mean())))}


def run(v13_run='20261008_proposal_baseline_vs_finetune_v13'):
    require_preregistered()
    study = v13_context(v13_run)
    params = yaml.safe_load((ROOT / 'research_v11/protocol.yaml').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    patients = [r for r in study.records if r['pool'] in {'development', 'test'}]
    for i, r in enumerate(patients, 1):
        raw = pd.read_csv(ROOT / params['raw_directory'] / f"{r['hospital_id']}.csv", usecols=LEADS)[LEADS].to_numpy(float)
        rows.append({'hospital_id': r['hospital_id'], 'pool': r['pool'], 'label3': r['label'], 'sublocation': r['sublocation'],
                     'truth': int(r['label'] > 0), **patient_measures(raw, params['sampling_rate'], params)})
        if i % 25 == 0: print(f'Kriteria klinis {i}/{len(patients)}', flush=True)
    m = pd.DataFrame(rows).set_index('hospital_id')
    m = m.join(rule_calls(m)); m.to_csv(OUT / 'patient_measures_and_calls.csv')
    results, comparisons = [], []
    fine = read_json(study.run / 'selection.json')['choices']
    for pool in ('development', 'test'):
        g = m[m.pool == pool]
        for rule in ('tz', 'v2s_v3r', 'v2tr'):
            results.append({'pool': pool, 'method': f'kriteria_{rule}', **scores(g.truth, g[f'{rule}_call'], g[f'{rule}_score'])})
        for model in study.protocol['models']:
            if pool == 'development':
                per_seed = []
                for seed in study.protocol['cv_seeds']:
                    frames = [pd.read_csv(cv_directory(study, model, fine[model]['finetune']['scenario'], f, seed) /
                                          'validation_predictions.csv', dtype={'hospital_id': str}) for f in study.protocol['folds']]
                    per_seed.append(pd.concat(frames).set_index('hospital_id'))
                p = pd.concat([(s[P13[1]] + s[P13[2]]).rename(i) for i, s in enumerate(per_seed)], axis=1).mean(1)
                label = 'model_v13_finetune_cv_oof_rata3seed'
            else:
                frames = [pd.read_csv(test_directory(study, model, 'finetune', seed) / 'patient_predictions.csv',
                                      dtype={'hospital_id': str}).set_index('hospital_id') for seed in study.protocol['final_seeds']]
                p = pd.concat([(s[P13[1]] + s[P13[2]]).rename(i) for i, s in enumerate(frames)], axis=1).mean(1)
                label = 'model_v13_finetune_ensemble5'
            p = p.reindex(g.index)
            results.append({'pool': pool, 'method': f'{label}_{model}', **model_scores(g.truth, p)})
            for rule in ('tz', 'v2s_v3r', 'v2tr'):
                comparisons.append({'pool': pool, 'model': model, 'rule': rule,
                                    **paired_auc_bootstrap(g.truth.to_numpy(), g[f'{rule}_score'].to_numpy(), p.to_numpy())})
    results, comparisons = pd.DataFrame(results), pd.DataFrame(comparisons)
    results.to_csv(OUT / 'clinical_vs_model.csv', index=False); comparisons.to_csv(OUT / 'auc_model_minus_rule.csv', index=False)
    quality = m.groupby('pool').agg(patients=('truth', 'size'), with_pvc_beats=('n_pvc_beats', lambda x: int((x > 0).sum())),
                                    with_sinus_beats=('n_sinus_beats', lambda x: int((x > 0).sum()))).reset_index()
    quality.to_csv(OUT / 'beat_selection_quality.csv', index=False)
    write_json(OUT / 'manifest.json', {'code': code_hashes(), 'preregistration_sha256': sha256_file(PREREGISTRATION),
                                       'source_v13_hash': study.signature, 'trained_parameters': 0,
                                       'thresholds': {'pvc_min_qrs_ms': PVC_MIN_QRS_MS, 'tz_lvot_max': TZ_LVOT_MAX,
                                                      'tz_rvot_min': TZ_RVOT_MIN, 'v2s_v3r_lvot_max': V2S_V3R_LVOT_MAX,
                                                      'v2_transition_ratio_lvot_min': V2_TRANSITION_RATIO_LVOT_MIN}})
    pd.set_option('display.width', 200)
    print(quality.to_string(index=False)); print(results.round(3).to_string(index=False)); print(comparisons.round(3).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v13-run', default='20261008_proposal_baseline_vs_finetune_v13')
    run(parser.parse_args().v13_run)
