"""Read-only summaries and figures, also useful when a stage is incomplete."""
import os
import numpy as np
import pandas as pd
from src.artifacts import ROOT, lock, read_json, sha256_file, write_json
from src.evaluation import compute_metrics
from .common import progress, cv_directory, scenarios


def analyze(run, cfg, config):
    with lock(run / 'reports/.analysis.lock'):
        return _analyze(run, cfg, config)


def _analyze(run, cfg, config):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = run / 'reports'; out.mkdir(parents=True, exist_ok=True)
    diagnostics, stability, test, oof = [], [], [], []
    for p in sorted((run / 'diagnostic').glob('*/*/fold_*/seed_*/metrics.json')):
        result = read_json(p)
        identity = read_json(p.with_name('config.json'))
        diagnostics.append({'recipe': p.parents[3].name, 'model': identity['model'],
            'fold': identity['fold'], 'seed': identity['seed'], 'best_epoch': result['best_epoch'],
            'missing_predicted_classes': result['validation_missing_predicted_classes'], **result['metrics']})
    for p in sorted((run / 'stability').glob('*/fold_*/seed_*/metrics.json')):
        result, binding = read_json(p), read_json(p.with_name('config.json'))
        if 'reused_cv_case' in binding:
            origin = run / binding['reused_cv_case']
            if (binding['source_config_sha256'] != sha256_file(origin / 'config.json') or
                    binding['source_metrics_sha256'] != sha256_file(origin / 'metrics.json')):
                raise RuntimeError('Reused stability case changed')
        stability.append({'model': p.parents[2].name, 'fold': int(p.parents[1].name.split('_')[-1]),
            'seed': int(p.parent.name.split('_')[-1]), 'best_epoch': result['best_epoch'],
            'missing_predicted_classes': result['validation_missing_predicted_classes'], **result['metrics']})
    for p in sorted((run / 'test').glob('*/seed_*/metrics.json')):
        test.append({'model': p.parents[1].name, 'seed': int(p.parent.name.split('_')[-1]), **read_json(p)['metrics']})
    for name, rows in [('native_diagnostic', diagnostics), ('selected_stability', stability), ('historical_test', test)]:
        if rows:
            frame = pd.DataFrame(rows)
            frame.to_csv(out / f'{name}.csv', index=False)
            grouping = ['recipe', 'model'] if name == 'native_diagnostic' else ['model']
            frame.groupby(grouping).agg(f1_mean=('f1', 'mean'), f1_sd_cases=('f1', 'std'),
                accuracy_mean=('accuracy', 'mean'), n_cases=('f1', 'count')).reset_index().to_csv(out / f'{name}_summary.csv', index=False)
    if stability:
        frame = pd.DataFrame(stability)
        frame.groupby(['model', 'fold']).agg(f1_mean=('f1', 'mean'), f1_seed_sd=('f1', 'std'),
            seeds_completed=('seed', 'count'), collapse_cases=('missing_predicted_classes', lambda x: int((x > 0).sum()))
        ).reset_index().to_csv(out / 'stability_within_each_fold.csv', index=False)
    paired = []
    for model in cfg['models']:
        for sc in scenarios(cfg):
            for fold in range(1, 6):
                paths = [cv_directory(run, model, p, sc['id'], fold, cfg['cv_seed']) / 'metrics.json'
                         for p in ('mean', 'attention')]
                if all(p.exists() for p in paths):
                    a, b = [read_json(p)['metrics']['f1'] for p in paths]
                    paired.append({'model': model, 'scenario': sc['id'], 'fold': fold,
                                   'mean_f1': a, 'attention_f1': b, 'attention_minus_mean': b - a})
    if paired:
        pd.DataFrame(paired).to_csv(out / 'paired_pooling_deltas.csv', index=False)
    if (run / 'selection.json').exists():
        for choice in read_json(run / 'selection.json')['choices']:
            paths = [run / 'cv' / choice['model'] / choice['pooling'] / choice['scenario'] /
                     f'fold_{f}' / f'seed_{cfg["cv_seed"]}/validation_predictions.csv' for f in range(1, 6)]
            if all(p.exists() for p in paths):
                frame = pd.concat([pd.read_csv(p, dtype={'hospital_id': str}) for p in paths])
                expected = sum(r['pool'] == 'development' for r in read_json(run / 'data/patients.json'))
                if frame.hospital_id.duplicated().any() or len(frame) != expected:
                    raise RuntimeError('OOF coverage changed')
                probs = frame[[f'probability_{c}' for c in config['class_names']]].to_numpy()
                oof.append({'model': choice['model'], **compute_metrics(frame.true_label.to_numpy(), probs),
                            'selection_optimistic': True})
                frame.to_csv(out / f'selected_oof_{choice["model"]}.csv', index=False)
        pd.DataFrame(oof).to_csv(out / 'selected_oof_metrics.csv', index=False)
    cvpath = out / 'cv_summary.csv'
    if cvpath.exists() and cvpath.stat().st_size > 2:
        frame = pd.read_csv(cvpath)
        if len(frame):
            fig, axes = plt.subplots(1, len(cfg['models']), figsize=(20, 4.5), squeeze=False)
            for ax, model in zip(axes[0], cfg['models']):
                subset = frame[frame.model == model]
                if len(subset):
                    pivot = subset.pivot(index='pooling', columns='scenario', values='f1_mean').reindex(cfg['pooling'])
                    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, aspect='auto', cmap='viridis')
                    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
                    ax.set_yticks(range(len(pivot.index)), pivot.index)
                    for (i, j), value in np.ndenumerate(pivot.to_numpy()):
                        if np.isfinite(value):
                            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white', fontsize=8)
                ax.set_title(model)
            fig.suptitle('Development CV macro-F1; incomplete configurations are absent')
            fig.tight_layout(); fig.savefig(out / 'cv_grid.png', dpi=160); plt.close(fig)
    if (run / 'selection.json').exists():
        primary = read_json(run / 'selection.json')['primary']
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for fold in range(1, 6):
            directory = cv_directory(run, primary['model'], primary['pooling'], primary['scenario'], fold, cfg['cv_seed'])
            history = pd.read_csv(directory / 'history.csv')
            best = read_json(directory / 'metrics.json')['best_epoch']
            label = f'fold {fold}'
            axes[0, 0].plot(history.epoch, history.train_loss_minibatch_weighted_mean, label=label)
            axes[0, 1].plot(history.epoch, history.validation_f1, label=label)
            axes[0, 1].scatter([best], [history.loc[history.epoch == best, 'validation_f1'].iloc[0]], s=25)
            axes[0, 2].plot(history.epoch, history.gradient_norm_before_clip_mean, label=label)
            for c, ax in enumerate(axes[1]):
                ax.plot(history.epoch, history[f'validation_recall_class_{c}'], label=label)
                ax.set_title(f'Validation recall: {config["class_names"][c]}'); ax.set_ylim(0, 1)
        axes[0, 0].set_title('Weighted training minibatch loss')
        axes[0, 1].set_title('Patient validation macro-F1 / selected epoch'); axes[0, 1].set_ylim(0, 1)
        axes[0, 2].set_title('Mean gradient norm before clipping')
        for ax in axes.flat:
            ax.set_xlabel('Epoch'); ax.legend(fontsize=8)
        fig.suptitle(f'Development primary: {primary["model"]}/{primary["pooling"]}/{primary["scenario"]}')
        fig.tight_layout(); fig.savefig(out / 'learning_curves_primary.png', dpi=160); plt.close(fig)
    if diagnostics or stability:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, rows, title in zip(axes, [diagnostics, stability], ['Native V3 diagnosis', 'Selected V4 stability']):
            frame = pd.DataFrame(rows)
            if len(frame):
                if 'recipe' in frame:
                    frame['series'] = frame.recipe + '/' + frame.model
                else:
                    frame['series'] = frame.model
                groups = list(frame.groupby('series', sort=True))
                for i, (label, subset) in enumerate(groups):
                    ax.scatter(np.full(len(subset), i), subset.f1, alpha=.6)
                ax.set_xticks(range(len(groups)), [g[0] for g in groups], rotation=25, ha='right')
            ax.set_ylim(0, 1); ax.set_ylabel('Patient macro-F1 per fold/seed'); ax.set_title(title)
        fig.tight_layout(); fig.savefig(out / 'training_stability.png', dpi=160); plt.close(fig)
    # Show all seeds; seed 42 is the frozen primary, no best-test ranking.
    if test:
        frame = pd.DataFrame(test)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, model in enumerate(cfg['models']):
            subset = frame[frame.model == model]
            ax.scatter(np.full(len(subset), i), subset.f1, label=model)
            primary = subset[subset.seed == cfg['primary_seed']]
            ax.scatter(np.full(len(primary), i), primary.f1, marker='*', s=140, color='black')
        ax.set_xticks(range(len(cfg['models'])), cfg['models'], rotation=20)
        ax.set_ylim(0, 1); ax.set_ylabel('Patient macro-F1'); ax.set_title('Historical test — all seeds; black star: frozen seed 42')
        fig.tight_layout(); fig.savefig(out / 'historical_test_seeds.png', dpi=160); plt.close(fig)
    write_json(out / 'status.json', {'cv': {k: v for k, v in progress(run, cfg).items() if k != 'missing'},
        'native_diagnostic_completed': len(diagnostics), 'native_diagnostic_default_expected': 45,
        'stability_completed': len(stability), 'stability_expected': 75,
        'final_completed': len(list((run / 'final').glob('*/seed_*/metrics.json'))), 'final_expected': 25,
        'historical_test_completed': len(test), 'historical_test_expected': 25,
        'selection_optimistic_cv': True, 'external_anatomical_validation_done': False,
        'clinical_expert_review_done': False, 'q1_acceptance_or_0p9_guaranteed': False})
    print('Reports:', out, flush=True)
