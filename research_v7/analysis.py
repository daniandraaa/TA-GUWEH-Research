"""Development-only figures and a concise pass/fail report for manual GAN review."""
import numpy as np
import pandas as pd
from src.artifacts import read_json, sha256_file
from src.data import LEADS
from .common import require_prepared, verify_files
from .data import load_development


def report(study, fold):
    import os
    from src.artifacts import ROOT
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = study.folder(fold)
    require_prepared(study, fold)
    quality = read_json(folder / 'quality.json'); verify_files(folder, quality)
    generated = read_json(folder / 'generated.json'); verify_files(folder, generated)
    completed = read_json(folder / 'completed.json'); verify_files(folder, completed)
    if (quality['experiment_hash'] != study.signature or completed['experiment_hash'] != study.signature
            or quality['generated_manifest_sha256'] != sha256_file(folder / 'generated.json')
            or generated['generator_sha256'] != sha256_file(folder / 'generator.pth')):
        raise RuntimeError('Report input identity changed')
    output = folder / 'figures'; output.mkdir(exist_ok=True)
    history = pd.read_csv(folder / 'history.csv')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for key in ('d_loss', 'g_adversarial', 'g_fidelity'): axes[0].plot(history.epoch, history[key], label=key)
    axes[0].set(title='Training losses (not a quality score)', xlabel='Epoch'); axes[0].legend(fontsize=8)
    for c in range(3): axes[1].plot(history.epoch, history[f'class_{c}_draws'], label=f'Class {c}')
    axes[1].set(title='Class-uniform draws', xlabel='Epoch'); axes[1].legend()
    axes[2].plot(history.epoch, history.unique_fit_patients_drawn)
    axes[2].set(title='Unique fit patients drawn', xlabel='Epoch')
    fig.tight_layout(); fig.savefig(output / 'training_process.png', dpi=150); plt.close(fig)
    if not quality['numerical_screen']['finite']:
        (folder / 'REPORT.md').write_text('Synthetic values are nonfinite. Gate FAIL. Inspect training losses; no downstream classifier.\n')
        return
    X, table = load_development(study)
    synthetic = np.load(folder / 'synthetic.npy', mmap_mode='r')
    labels = pd.read_csv(folder / 'synthetic_labels.csv').label.to_numpy()
    rows = {name: pd.read_csv(folder / f'{name}_rows.csv', dtype={'hospital_id': str}) for name in ('fit', 'reference')}
    names = {r['label']: r['class_name'] for r in study.records}
    time = np.arange(5000) / 2000
    for c in range(3):
        # Predefined first row per class; no best-looking sample selection.
        samples = [np.asarray(X[rows[name][rows[name].label.eq(c)].iloc[0].segment_index]) for name in ('fit', 'reference')]
        samples.append(np.asarray(synthetic[np.flatnonzero(labels == c)[0]]))
        fig, axes = plt.subplots(12, 3, figsize=(14, 16), sharex=True)
        for lead, name in enumerate(LEADS):
            low = min(v[lead].min() for v in samples); high = max(v[lead].max() for v in samples)
            pad = max((high - low) * .05, 1.)
            for column, value in enumerate(samples):
                axes[lead, column].plot(time, value[lead], linewidth=.6)
                axes[lead, column].set_ylim(low - pad, high + pad)
                axes[lead, column].grid(alpha=.15)
            axes[lead, 0].set_ylabel(name)
        for column, title in enumerate(('Real inner-fit', 'Real inner-reference', 'Synthetic (unverified condition)')):
            axes[0, column].set_title(title); axes[-1, column].set_xlabel('Seconds (2000 Hz)')
        fig.suptitle(f'{names[c]} | native amplitude units; first predefined samples', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .98)); fig.savefig(output / f'waveforms_class_{c}.png', dpi=120); plt.close(fig)
    distances = pd.read_csv(folder / 'feature_distances.csv')
    families = ['amplitude', 'roughness', 'spectrum', 'correlation', 'limb_residual']
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in range(3):
        values = [float((~distances[(distances.label == c) & (distances.family == f)].passed).mean()) for f in families]
        ax.bar(np.arange(len(families)) + (c - 1) * .25, values, width=.25, label=names[c])
    ax.axhline(study.protocol['quality_gate']['max_failed_feature_fraction_per_family_and_class'], color='red', linestyle='--', label='Engineering gate')
    ax.set(xticks=np.arange(len(families)), xticklabels=families, ylabel='Fraction of feature screens failed', ylim=(0, 1))
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / 'quality_families.png', dpi=150); plt.close(fig)
    a, b, fake = (pd.read_csv(folder / n) for n in ('fit_features.csv', 'reference_features.csv', 'synthetic_features.csv'))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for c in range(3):
        for index, (name, data) in enumerate((('Fit', a), ('Reference', b), ('Synthetic', fake))):
            selected = data[data.label.eq(c)]
            columns = [n for n in data.columns if n.startswith('power_')]
            powers = []
            from research_v6.quality import BANDS
            for lo, hi in BANDS:
                names_band = [n for n in columns if n.startswith(f'power_{lo:g}_{hi:g}_')]
                mean_lead_power = selected[names_band].mean(axis=1)
                powers.append(float(np.average(mean_lead_power, weights=selected.weight)))
            axes[c].bar(np.arange(7) + (index - 1) * .25, powers, width=.25, label=name)
        axes[c].set(title=names[c], xticks=np.arange(7), xticklabels=[f'{lo:g}-{hi:g}' for lo, hi in BANDS], xlabel='Hz'); axes[c].tick_params(axis='x', labelrotation=45)
    axes[0].set_ylabel('Patient-balanced relative Welch power (mean 12 leads)'); axes[-1].legend()
    fig.tight_layout(); fig.savefig(output / 'frequency_bands.png', dpi=150); plt.close(fig)
    ns, nr = (pd.read_csv(folder / n) for n in ('synthetic_nearest_fit.csv', 'reference_nearest_fit.csv'))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for c in range(3):
        axes[c].hist(ns[ns.label == c].relative_rmse, bins=15, density=True, alpha=.5, label='Synthetic to fit')
        selected = nr[nr.label == c]
        axes[c].hist(selected.relative_rmse, bins=15, weights=selected.weight, density=True, alpha=.5, label='Reference to fit')
        axes[c].axvline(study.protocol['quality_gate']['near_copy_relative_rmse'], color='red', linestyle='--')
        axes[c].set(title=names[c], xlabel='Full native zero-lag RMSE / real RMS')
    axes[0].set_ylabel('Density'); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'copy_distance.png', dpi=150); plt.close(fig)
    message = 'PASS: eligible for the remaining GAN folds; classifier effect remains untested.' if quality['passed'] else 'FAIL: diagnose the failed families; do not proceed to other folds/classifier.'
    lines = [f'# V7 GAN quality — fold {fold}', '', message, '',
        f'Run hash: `{study.signature}`', '',
        'Engineering screening only. No PVC/origin clinical validation, privacy certification, or classifier benefit has been established.', '',
        'Waveform examples are the first deterministic samples, in native units at 2000 Hz (2.5 seconds).', '',
        'Welch spectra use 2000 samples, 50% overlap and 1 Hz resolution. Real distributions give each patient total weight one.', '',
        'Nearest neighbours search every inner-fit window of the same class at full native resolution without phase alignment.', '',
        '## Failed families', '', str(quality['failed_families']), '',
        '## Numerical checks', '', str(quality['numerical_screen']), '',
        '## Figures', '']
    for path in sorted(output.glob('*.png')): lines.extend((f'![{path.stem}](figures/{path.name})', ''))
    (folder / 'REPORT.md').write_text('\n'.join(lines))
    print(f'Report: {folder / "REPORT.md"}\nGate: {message}', flush=True)
    compare(study, fold)


def compare(study, fold):
    """Same QC and patients, no seed/test selection. Statistics are exploratory."""
    old = study.baseline / f'fold_{fold}'
    new = study.folder(fold)
    if not (old / 'quality.json').exists():
        print('No V6 completed comparator for this fold; report V7 QC only.', flush=True)
        return
    a, b = read_json(old / 'quality.json'), read_json(new / 'quality.json')
    for path, record in ((old, a), (new, b)): verify_files(path, record)
    frames = []
    for label, folder in (('V6', old), ('V7', new)):
        data = pd.read_csv(folder / 'feature_distances.csv')
        summary = data.assign(failed=~data.passed).groupby(['label','family']).agg(
            failed_fraction=('failed','mean'), normalized_wasserstein_mean=('synthetic_reference_distance','mean')).reset_index()
        summary['version'] = label; frames.append(summary)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(new / 'comparison_v6_v7.csv', index=False)
    import matplotlib.pyplot as plt
    keys = [(c, f) for c in range(3) for f in ['amplitude','roughness','correlation','spectrum','limb_residual']]
    fig, ax = plt.subplots(figsize=(14, 5))
    for offset, version in enumerate(('V6','V7')):
        values = [float(combined[(combined.version == version) & (combined.label == c) & (combined.family == f)].iloc[0].failed_fraction) for c,f in keys]
        ax.bar(np.arange(len(keys)) + (offset - .5) * .35, values, width=.35, label=version)
    ax.axhline(study.protocol['quality_gate']['max_failed_feature_fraction_per_family_and_class'], color='red', linestyle='--')
    ax.set(xticks=np.arange(len(keys)), xticklabels=[f'C{c} {f}' for c,f in keys], ylabel='Fraction of features failed (lower is better)', ylim=(0,1))
    ax.tick_params(axis='x', labelrotation=65); ax.legend(); fig.tight_layout()
    fig.savefig(new / 'figures/comparison_v6_v7.png', dpi=150); plt.close(fig)
    report = new / 'REPORT.md'
    with report.open('a') as stream:
        stream.write(f'\n## V6 versus V7, same fold\n\nV6 gate: {a["passed"]}; V7 gate: {b["passed"]}. '
            'This is an objective-bundle ablation; individual component benefit and classifier performance remain untested.\n\n'
            '![V6 versus V7](figures/comparison_v6_v7.png)\n')


def status(study):
    states = []
    for fold in study.protocol['folds']:
        directory = study.folder(fold)
        for marker in ('preparation.json', 'completed.json', 'generated.json', 'quality.json'):
            if (directory / marker).exists():
                saved = read_json(directory / marker)
                verify_files(directory, saved)
                if saved['experiment_hash'] != study.signature:
                    raise RuntimeError(f'Wrong identity: {directory / marker}')
        states.append({'fold': fold, 'prepared': (directory / 'preparation.json').exists(),
            'gan_training_complete': (directory / 'completed.json').exists(),
            'generated': (directory / 'generated.json').exists(),
            'quality_passed': read_json(directory / 'quality.json')['passed'] if (directory / 'quality.json').exists() else None})
    return {'experiment_hash': study.signature, 'folds': states,
        'classifier_enabled': False, 'test_enabled': False,
        'next_step': '37 prepare/smoke -> 38 fold1 train/generate/qc -> 39 compare/visualize -> 40 remaining folds only after PASS'}
