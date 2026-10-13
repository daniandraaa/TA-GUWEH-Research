"""Kalibrasi dan decision curve analysis (DCA) untuk keputusan LVOT, data development + test yang ada.

Kelas positif = LVOT (keputusan klinis: rencanakan pemetaan/ablasi sisi kiri). Probabilitas = P(LVOT).
Development: prediksi validasi CV (OOF) rata-rata 3 seed; test: ensemble 5 seed. Tidak ada rekalibrasi
yang dipasang ke test; kemiringan/intersep hanya deskriptif.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from src.artifacts import ROOT, read_json, write_json
from src.reports import plotting
from research_v13.common import context as v13_context, cv_directory, test_directory

OUT = ROOT / 'results/research_v14/calibration_dca_v14'
THRESHOLDS = np.round(np.arange(0.05, 0.61, 0.01), 2)


def lvot_probability(study, model, pool):
    fine = read_json(study.run / 'selection.json')['choices'][model]['finetune']['scenario']
    if pool == 'development':
        seeds = [pd.concat([pd.read_csv(cv_directory(study, model, fine, f, s) / 'validation_predictions.csv',
                                        dtype={'hospital_id': str}) for f in study.protocol['folds']]).set_index('hospital_id')
                 for s in study.protocol['cv_seeds']]
    else:
        seeds = [pd.read_csv(test_directory(study, model, 'finetune', s) / 'patient_predictions.csv',
                             dtype={'hospital_id': str}).set_index('hospital_id') for s in study.protocol['final_seeds']]
    p = pd.concat([s.probability_LVOT.rename(i) for i, s in enumerate(seeds)], axis=1).mean(axis=1)
    return p, (seeds[0].true_label.reindex(p.index) == 0).astype(int)


def calibration(y, p):
    eps = 1e-6; logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1))
    slope_fit = LogisticRegression(C=1e9).fit(logit.reshape(-1, 1), y)
    # Intersep "calibration-in-the-large": logit rata-rata observasi - rata-rata logit prediksi (pendekatan)
    citl = float(np.log(y.mean() / (1 - y.mean())) - np.log(p.mean() / (1 - p.mean())))
    return {'brier': float(brier_score_loss(y, p)), 'brier_reference_prevalence': float(y.mean() * (1 - y.mean())),
            'calibration_slope': float(slope_fit.coef_[0, 0]), 'calibration_in_the_large': citl,
            'mean_predicted_lvot': float(p.mean()), 'observed_lvot_rate': float(y.mean()), 'auc': float(roc_auc_score(y, p))}


def net_benefit(y, decision, t):
    n = len(y); tp = int(((decision == 1) & (y == 1)).sum()); fp = int(((decision == 1) & (y == 0)).sum())
    return tp / n - fp / n * t / (1 - t)


def run():
    study = v13_context('20261008_proposal_baseline_vs_finetune_v13')
    OUT.mkdir(parents=True, exist_ok=True)
    rules = pd.read_csv(ROOT / 'results/research_v14/clinical_criteria_v14/patient_measures_and_calls.csv',
                        dtype={'hospital_id': str}).set_index('hospital_id')
    plt = plotting(OUT); cal_rows, dca_rows = [], []
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, pool in enumerate(('development', 'test')):
        for model in study.protocol['models']:
            p, y = lvot_probability(study, model, pool)
            cal_rows.append({'pool': pool, 'model': model, 'n': len(y), **calibration(y.to_numpy(), p.to_numpy())})
            bins = pd.qcut(p, 5, duplicates='drop')
            g = pd.DataFrame({'p': p, 'y': y}).groupby(bins, observed=True).mean()
            axes[0, col].plot(g.p, g.y, 'o-', label=model)
            for t in THRESHOLDS:
                dca_rows.append({'pool': pool, 'strategy': f'model_{model}', 'threshold': t,
                                 'net_benefit': net_benefit(y.to_numpy(), (p.to_numpy() >= t).astype(int), t)})
            axes[1, col].plot(THRESHOLDS, [r['net_benefit'] for r in dca_rows if r['pool'] == pool and r['strategy'] == f'model_{model}'], label=model)
        r = rules[rules.pool == pool].reindex(y.index)
        call = (r.v2s_v3r_call == 0).astype(int).to_numpy()  # LVOT bila V2S/V3R <= 1,5; tak terukur = bukan LVOT
        for t in THRESHOLDS:
            dca_rows += [{'pool': pool, 'strategy': 'semua_LVOT', 'threshold': t, 'net_benefit': net_benefit(y.to_numpy(), np.ones(len(y), int), t)},
                         {'pool': pool, 'strategy': 'tidak_ada', 'threshold': t, 'net_benefit': 0.},
                         {'pool': pool, 'strategy': 'kriteria_V2S_V3R', 'threshold': t, 'net_benefit': net_benefit(y.to_numpy(), call, t)}]
        for s, style in (('semua_LVOT', ':'), ('kriteria_V2S_V3R', '--')):
            axes[1, col].plot(THRESHOLDS, [d['net_benefit'] for d in dca_rows if d['pool'] == pool and d['strategy'] == s], style, color='gray' if s == 'semua_LVOT' else 'black', label=s)
        axes[1, col].axhline(0, color='lightgray', lw=.8)
        axes[0, col].plot([0, 1], [0, 1], '--', color='gray', lw=.8)
        axes[0, col].set(title=f'Kalibrasi P(LVOT) — {pool}', xlabel='Prediksi rata-rata (kuintil)', ylabel='Proporsi LVOT teramati')
        axes[1, col].set(title=f'Decision curve LVOT — {pool}', xlabel='Ambang probabilitas', ylabel='Net benefit',
                         ylim=(-0.05, max(0.3, y.mean() + .02)))
        axes[0, col].legend(fontsize=8); axes[1, col].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'kalibrasi_dan_dca.png', dpi=180); plt.close(fig)
    cal, dca = pd.DataFrame(cal_rows), pd.DataFrame(dca_rows)
    cal.to_csv(OUT / 'calibration.csv', index=False); dca.to_csv(OUT / 'decision_curve.csv', index=False)
    key = dca[dca.threshold.isin([0.1, 0.2, 0.3, 0.4])].pivot_table(index=['pool', 'threshold'], columns='strategy', values='net_benefit')
    key.to_csv(OUT / 'decision_curve_key_thresholds.csv')
    write_json(OUT / 'manifest.json', {'positive_class': 'LVOT', 'development': 'CV OOF mean 3 seeds', 'test': 'ensemble 5 seeds',
                                       'recalibration_fitted_on_test': False})
    pd.set_option('display.width', 200)
    print(cal.round(3).to_string(index=False)); print(key.round(3).to_string())


if __name__ == '__main__':
    run()
