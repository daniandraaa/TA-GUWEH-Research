"""Saved-data scientific figures and patient-level uncertainty."""
import numpy as np
import pandas as pd
from src.artifacts import read_json
from src.reports import plotting
from research_v2.reporting import bootstrap_f1
from .experiment import summarize


def make_report(run, cfg, config, grid):
    summary = summarize(run, cfg, config, grid)
    out = run / 'reports'
    plt = plotting(run)
    fold_path = out / 'cv_folds.csv'
    if fold_path.stat().st_size > 1:
        folds = pd.read_csv(fold_path)
        paired = folds.pivot(index=['model', 'scenario', 'fold'], columns='arm', values='f1')
        for rep in ['window', 'r_candidate']:
            left, right = rep + '_patient', rep + '_segment'
            if left in paired and right in paired:
                paired[rep + '_patient_minus_segment'] = paired[left] - paired[right]
        for method in ['patient', 'segment']:
            left, right = 'r_candidate_' + method, 'window_' + method
            if left in paired and right in paired:
                paired[method + '_candidate_minus_window'] = paired[left] - paired[right]
        paired.to_csv(out / 'matched_cv_folds.csv')
    if len(summary):
        table = summary.groupby(['model', 'arm']).f1_mean.max().unstack()
        ax = table.plot.bar(figsize=(12, 5), ylim=(0, 1), rot=0)
        ax.set_ylabel('Best-grid development CV macro-F1 (selection optimistic)')
        ax.figure.tight_layout()
        ax.figure.savefig(out / 'cv_representation_sampling.png', dpi=180)
        plt.close(ax.figure)
    rows = []
    for model in cfg['models'] + ['extratrees']:
        for seed in cfg['final_seeds']:
            p = run / 'test' / model / f'seed_{seed}/metrics.json'
            if not p.exists():
                continue
            rows.append({'model': model, 'seed': seed, **read_json(p)['metrics']})
            if seed == cfg['primary_seed']:
                pred = pd.read_csv(p.parent / 'patient_predictions.csv')
                probabilities = pred[[f'probability_{c}' for c in config['class_names']]].to_numpy()
                lo, hi = bootstrap_f1(pred.true_label.to_numpy(), probabilities, 2000, 20261001)
                rows[-1].update(f1_ci_low=lo, f1_ci_high=hi)
                cm = np.array(read_json(p)['confusion_matrix'])
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.imshow(cm, cmap='Blues')
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(cm[i, j]), ha='center', va='center')
                ax.set_xticks(range(3), config['class_names'], rotation=25, ha='right')
                ax.set_yticks(range(3), config['class_names'])
                ax.set_xlabel('Predicted'); ax.set_ylabel('True patient class')
                ax.set_title(f'{model}, primary seed 42, historical test')
                fig.tight_layout(); fig.savefig(out / f'{model}_confusion_seed42.png', dpi=180)
                plt.close(fig)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(out / 'historical_test_patient_metrics.csv', index=False)
        frame.groupby('model')[['f1', 'accuracy', 'auc']].agg(['mean', 'std']).to_csv(out / 'seed_summary.csv')
    print(f'Laporan: {out}', flush=True)
