"""Tabel Bab IV, uji signifikansi baseline vs fine-tuning, dan grafik. Tanpa training."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from src.artifacts import read_json, write_json
from src.reports import plotting
from .common import baseline, grid, test_directory
from .experiment import cv_table, summarize

ARMS = ('baseline', 'finetune')


def holm(pvalues):
    """Koreksi Holm-Bonferroni; mengembalikan p terkoreksi dengan urutan asli."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p); adjusted = np.empty_like(p); running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def mcnemar_exact(correct_a, correct_b):
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1. if n == 0 else float(stats.binomtest(min(only_a, only_b), n, .5).pvalue)
    return {'baseline_correct_finetune_wrong': only_a, 'baseline_wrong_finetune_correct': only_b, 'p_value': p}


def wilcoxon(x, y):
    d = np.asarray(y, float) - np.asarray(x, float)
    if np.allclose(d, 0):
        return {'mean_difference': 0., 'statistic': None, 'p_value': 1.}
    r = stats.wilcoxon(y, x, zero_method='wilcox', alternative='two-sided')
    return {'mean_difference': float(d.mean()), 'statistic': float(r.statistic), 'p_value': float(r.pvalue)}


def bootstrap_difference(truth, prob_a, prob_b, resamples, seed):
    """Selisih macro-F1 (b - a) dengan bootstrap pasien terstratifikasi kelas."""
    truth = np.asarray(truth); rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(truth == c) for c in np.unique(truth)]
    pa, pb = np.asarray(prob_a).argmax(1), np.asarray(prob_b).argmax(1)
    labels = np.unique(truth)
    observed = f1_score(truth, pb, average='macro', labels=labels) - f1_score(truth, pa, average='macro', labels=labels)
    values = np.empty(resamples)
    for i in range(resamples):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values[i] = (f1_score(truth[idx], pb[idx], average='macro', labels=labels, zero_division=0)
                     - f1_score(truth[idx], pa[idx], average='macro', labels=labels, zero_division=0))
    low, high = np.percentile(values, [2.5, 97.5])
    return {'observed_difference': float(observed), 'ci95_low': float(low), 'ci95_high': float(high),
            'proportion_resamples_le_zero': float((values <= 0).mean())}


def cv_statistics(study, frame):
    selection = read_json(study.run / 'selection.json')['choices']
    same = next(s['id'] for s in grid(study.protocol)
                if (s['learning_rate'], s['batch_size']) == (baseline(study.protocol)['learning_rate'], baseline(study.protocol)['batch_size']))
    rows = []
    for model in study.protocol['models']:
        def scores(scenario):
            part = frame[(frame.model == model) & (frame.scenario == scenario)].sort_values(['fold', 'seed'])
            return part.f1.to_numpy()
        base = scores('baseline')
        for comparison, scenario in (('fine-tuning terpilih (GAN + grid)', selection[model]['finetune']['scenario']),
                                     ('efek GAN saja (LR/batch sama dengan baseline)', same)):
            rows.append({'model': model, 'comparison': comparison, 'finetune_scenario': scenario,
                         'n_pairs_fold_x_seed': len(base), 'baseline_f1_mean': float(base.mean()),
                         'finetune_f1_mean': float(scores(scenario).mean()), **wilcoxon(base, scores(scenario))})
    table = pd.DataFrame(rows)
    for comparison in table.comparison.unique():
        mask = table.comparison == comparison
        table.loc[mask, 'p_holm'] = holm(table.loc[mask, 'p_value'])
    friedman = []
    for arm in ARMS:
        blocks = []
        for model in study.protocol['models']:
            scenario = selection[model][arm]['scenario']
            blocks.append(frame[(frame.model == model) & (frame.scenario == scenario)].sort_values(['fold', 'seed']).f1.to_numpy())
        r = stats.friedmanchisquare(*blocks)
        friedman.append({'arm': arm, 'models': '/'.join(study.protocol['models']), 'n_blocks': len(blocks[0]),
                         'chi_square': float(r.statistic), 'p_value': float(r.pvalue),
                         **{f'{m}_f1_mean': float(b.mean()) for m, b in zip(study.protocol['models'], blocks)}})
    return table, pd.DataFrame(friedman)


def test_tables(study):
    rows, per_class = [], []
    for model in study.protocol['models']:
        for arm in ARMS:
            for seed in study.protocol['final_seeds']:
                r = read_json(test_directory(study, model, arm, seed) / 'metrics.json')
                rows.append({'model': model, 'arm': arm, 'seed': seed, **r['metrics']})
                for c in study.class_names:
                    per_class.append({'model': model, 'arm': arm, 'seed': seed, 'class': c,
                                      **{k: r['classification_report'][c][k] for k in ('precision', 'recall', 'f1-score', 'support')}})
    return pd.DataFrame(rows), pd.DataFrame(per_class)


def predictions(study, model, arm, seed):
    frame = pd.read_csv(test_directory(study, model, arm, seed) / 'patient_predictions.csv', dtype={'hospital_id': str})
    return frame.sort_values('hospital_id').reset_index(drop=True)


def test_statistics(study):
    p, rows = study.protocol, []
    columns = [f'probability_{c}' for c in study.class_names]
    for model in p['models']:
        for seed in p['final_seeds']:
            a, b = predictions(study, model, 'baseline', seed), predictions(study, model, 'finetune', seed)
            if not a.hospital_id.equals(b.hospital_id):
                raise RuntimeError('Pasien test baseline/fine-tuning tidak identik')
            row = {'model': model, 'seed': seed, 'primary_seed': seed == p['primary_seed'],
                   **mcnemar_exact(a.correct, b.correct)}
            if seed == p['primary_seed']:
                row.update(bootstrap_difference(a.true_label, a[columns], b[columns],
                                                p['statistics']['bootstrap_resamples'], p['statistics']['bootstrap_seed']))
            rows.append(row)
    table = pd.DataFrame(rows)
    primary = table.primary_seed
    table.loc[primary, 'p_holm_primary'] = holm(table.loc[primary, 'p_value'])
    return table


def figures(study, summary, frame, test, per_class):
    plt = plotting(study.run); out = study.run / 'reports'
    models = study.protocol['models']; selection = read_json(study.run / 'selection.json')['choices']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, model in zip(axes, models):
        data = [frame[(frame.model == model) & (frame.scenario == selection[model][arm]['scenario'])].f1 for arm in ARMS]
        ax.boxplot(data, tick_labels=['Baseline', f'Fine-tuning\n{selection[model]["finetune"]["scenario"]}'])
        ax.set_title(model); ax.set_ylabel('Macro-F1 validasi (5 fold x 3 seed)')
    fig.tight_layout(); fig.savefig(out / 'cv_baseline_vs_finetune.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, model in zip(axes, models):
        part = summary[(summary.model == model) & (summary.augmentation == 'gan')]
        pivot = part.pivot(index='learning_rate', columns='batch_size', values='f1_mean')
        image = ax.imshow(pivot.to_numpy(), cmap='viridis')
        ax.set_xticks(range(len(pivot.columns)), pivot.columns); ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f'{pivot.iloc[i, j]:.3f}', ha='center', va='center', color='white', fontsize=9)
        ax.set_xlabel('Batch (segmen)'); ax.set_ylabel('Learning rate'); ax.set_title(f'{model}: grid GAN, mean macro-F1 CV')
        fig.colorbar(image, ax=ax, fraction=.046)
    fig.tight_layout(); fig.savefig(out / 'grid_search_heatmap.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, arm in enumerate(ARMS):
        for i, model in enumerate(models):
            values = test[(test.model == model) & (test.arm == arm)].f1
            x = i + (k - .5) * .3
            ax.scatter(np.full(len(values), x), values, alpha=.6, color=f'C{k}', label=arm if i == 0 else None)
            ax.errorbar(x, values.mean(), values.std(ddof=1), color='black', capsize=4, marker='_')
    ax.set_xticks(range(len(models)), models); ax.set_ylabel('Macro-F1 test 20% (5 seed)'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'test_baseline_vs_finetune.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    short = ['LVOT', 'RVOT cusp', 'RVOT other']
    for row, arm in enumerate(ARMS):
        for col, model in enumerate(models):
            matrix = np.asarray(read_json(test_directory(study, model, arm, study.protocol['primary_seed']) / 'metrics.json')['confusion_matrix'])
            ax = axes[row, col]; ax.imshow(matrix, cmap='Blues')
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, int(matrix[i, j]), ha='center', va='center')
            ax.set_xticks(range(3), short, rotation=20); ax.set_yticks(range(3), short)
            ax.set_title(f'{model} — {arm} (seed {study.protocol["primary_seed"]})'); ax.set_xlabel('Prediksi'); ax.set_ylabel('Label')
    fig.tight_layout(); fig.savefig(out / 'test_confusion_matrices.png', dpi=180); plt.close(fig)
    recall = per_class.groupby(['model', 'arm', 'class']).recall.mean().unstack('class')
    ax = recall.plot.bar(figsize=(10, 4.5)); ax.set_ylabel('Recall test (rata-rata 5 seed)'); ax.figure.tight_layout()
    ax.figure.savefig(out / 'test_recall_per_class.png', dpi=180); plt.close(ax.figure)


def analyze(study):
    out = study.run / 'reports'; out.mkdir(parents=True, exist_ok=True)
    frame = cv_table(study); summary = summarize(study, frame)
    frame.to_csv(out / 'cv_cases.csv', index=False); summary.to_csv(out / 'cv_summary.csv', index=False)
    paired, friedman = cv_statistics(study, frame)
    paired.to_csv(out / 'stat_cv_wilcoxon.csv', index=False); friedman.to_csv(out / 'stat_cv_friedman_models.csv', index=False)
    test, per_class = test_tables(study)
    test.to_csv(out / 'test_all_seeds.csv', index=False); per_class.to_csv(out / 'test_per_class.csv', index=False)
    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc', 'precision']
    overview = test.groupby(['model', 'arm'])[metrics].agg(['mean', 'std'])
    overview.columns = [f'{m}_{s}' for m, s in overview.columns]
    overview = overview.reset_index()
    primary = test[test.seed == study.protocol['primary_seed']].set_index(['model', 'arm'])[metrics].add_suffix('_seed42')
    overview = overview.merge(primary.reset_index(), on=['model', 'arm'])
    overview.to_csv(out / 'bab4_tabel_test.csv', index=False)
    stat_test = test_statistics(study); stat_test.to_csv(out / 'stat_test_mcnemar_bootstrap.csv', index=False)
    figures(study, summary, frame, test, per_class)
    alpha = study.protocol['statistics']['alpha']
    lines = ['# Ringkasan hasil V13 (baseline vs fine-tuning)', '',
             'Angka utama: macro-F1 tingkat pasien. CV = 5 fold x 3 seed pada 230 pasien development; '
             'test = 59 pasien hold-out historis, 5 seed final.', '', '## Test 20% (mean ± SD 5 seed)', '']
    for r in overview.itertuples():
        lines.append(f'- {r.model} / {r.arm}: F1 {r.f1_mean:.4f} ± {r.f1_std:.4f}; akurasi {r.accuracy_mean:.4f}; '
                     f'sensitivitas {r.sensitivity_mean:.4f}; spesifisitas {r.specificity_mean:.4f}; AUC {r.auc_mean:.4f}')
    lines += ['', '## Uji signifikansi', '']
    for r in paired.itertuples():
        lines.append(f'- CV {r.model}, {r.comparison}: Δ={r.mean_difference:+.4f}, Wilcoxon p={r.p_value:.4f}, Holm p={r.p_holm:.4f}'
                     f' → {"signifikan" if r.p_holm < alpha else "tidak signifikan"} (α={alpha})')
    for r in stat_test[stat_test.primary_seed].itertuples():
        lines.append(f'- Test {r.model} seed {r.seed}: ΔF1={r.observed_difference:+.4f} [CI95 {r.ci95_low:+.4f}, {r.ci95_high:+.4f}], '
                     f'McNemar p={r.p_value:.4f}, Holm p={r.p_holm_primary:.4f}')
    for r in friedman.itertuples():
        lines.append(f'- Friedman antar-3-model ({r.arm}, CV): χ²={r.chi_square:.3f}, p={r.p_value:.4f}')
    lines += ['', 'Catatan: pasangan fold x seed tidak sepenuhnya independen (fold berbagi data training); '
              'test 59 pasien pernah diamati pada eksperimen lama sehingga disebut hold-out historis.']
    (out / 'RINGKASAN_V13.md').write_text('\n'.join(lines) + '\n')
    write_json(out / 'analysis.json', {'experiment_hash': study.signature, 'files': sorted(p.name for p in out.iterdir())})
    print('\n'.join(lines), flush=True)
