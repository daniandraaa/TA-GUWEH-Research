"""Source audit and patient-level uncertainty; no test-based tuning."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import f1_score
from src.artifacts import ROOT, read_json, write_json
from src.data import LEADS
from .data import relabel, denoise

SOURCES = {
    'method_paper_2022': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8962834/',
    'dataset_descriptor_2020': 'https://www.nature.com/articles/s41597-020-0440-8',
    'collection_link_in_paper': 'https://doi.org/10.6084/m9.figshare.c.4668086.v2',
}


def audit(run, source, cfg, config):
    out = run / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    records = relabel(read_json(source / 'data/patients.json'), config['class_names'])
    frame = pd.DataFrame(records)
    frame.to_csv(out / 'patient_label_audit.csv', index=False)
    table = frame.groupby(['class_name_historical', 'class_name', 'sublocation', 'pool'], dropna=False).size().reset_index(name='n')
    table.to_csv(out / 'label_mapping_counts.csv', index=False)
    diagnosis = pd.read_excel(ROOT / 'Diagnosis.xlsx')
    # Cohort metadata only. Signal examples below are development only.
    write_json(out / 'source_evidence.json', {
        'sources': SOURCES, 'local_diagnosis_rows': len(diagnosis),
        'local_raw_files': len(list((ROOT / 'PVCVTRawECGData').glob('*.csv'))),
        'local_left_right': diagnosis.LeftRight.value_counts().to_dict(),
        'published_dataset_patients': 334, 'published_method_cohort': 545,
        'published_sampling_rate_hz': 2000, 'assumed_historical_sampling_rate_hz': 1000,
        'local_original_download_checksum_verified': False,
        'membership_changed': False, 'three_classes_are_custom_anatomical_groups': True,
        'segment_seconds': cfg['segment_length'] / cfg['sampling_rate'],
        'legacy_filter_effective_cutoff_hz': 1.0,
        'caveat': 'Dataset label validation does not validate local preprocessing, GAN or classifier.'})
    examples = frame[frame.pool == 'development'].groupby('label', sort=True).head(1)
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 3.2 * len(examples)), squeeze=False)
    for row, (_, record) in enumerate(examples.iterrows()):
        hid = record.hospital_id
        raw = pd.read_csv(ROOT / 'PVCVTRawECGData' / f'{hid}.csv')[LEADS].to_numpy()
        legacy = pd.read_csv(ROOT / 'preprocessed_data/dwt_denoised' / f'{hid}.csv')[LEADS].to_numpy()
        corrected = denoise(raw, cfg['sampling_rate'], cfg['preprocessing'])
        if len(raw) != len(legacy):
            raise ValueError('Legacy length differs: investigate resampling')
        n = min(len(raw), cfg['segment_length'])
        for label, values in [('Raw', raw), ('Historical DWT', legacy), ('Candidate fs=2000', corrected)]:
            axes[row, 0].plot(np.arange(n) / cfg['sampling_rate'], values[:n, 1], lw=.7, label=label)
            f, p = signal.welch(values[:, 1], fs=cfg['sampling_rate'], nperseg=min(2048, len(values)))
            axes[row, 1].semilogy(f, p + 1e-12, lw=.8, label=label)
        axes[row, 0].set_title(f'Development example {row+1}: {record.class_name}, Lead II')
        axes[row, 0].set_xlabel('Seconds (2000 Hz)')
        axes[row, 0].set_ylabel('CSV amplitude units (not independently verified)')
        axes[row, 1].set_xlim(0, 150)
        axes[row, 1].set_xlabel('Frequency (Hz)')
        axes[row, 1].set_ylabel('PSD')
        axes[row, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'preprocessing_review.png', dpi=160)
    plt.close(fig)
    counts = frame[frame.pool.isin(['development', 'test'])].groupby(['class_name', 'pool']).size().unstack(fill_value=0)
    ax = counts.plot.bar(rot=12, figsize=(10, 4))
    ax.set_ylabel('Patients'); ax.set_title('Corrected anatomical group names; historical membership retained')
    ax.figure.tight_layout(); ax.figure.savefig(out / 'cohort_groups.png', dpi=160); plt.close(ax.figure)
    return table


def bootstrap_f1(truth, probabilities, repeats=2000, seed=20260930):
    truth, probabilities = np.asarray(truth), np.asarray(probabilities)
    if probabilities.shape != (len(truth), 3) or set(np.unique(truth)) != {0, 1, 2}:
        raise ValueError('Patient bootstrap requires all three classes and Nx3 probabilities')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(truth == k) for k in range(3)]
    pred = probabilities.argmax(1)
    values = []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values.append(f1_score(truth[idx], pred[idx], labels=[0, 1, 2], average='macro', zero_division=0))
    return np.quantile(values, [.025, .975]).tolist()


def final_report(run, cfg, config):
    rows = []
    for name in config['models']:
        for seed in cfg['final_seeds']:
            directory = run / 'test' / name / f'seed_{seed}'
            table = pd.read_csv(directory / 'patient_predictions.csv', dtype={'hospital_id': str})
            if table.hospital_id.duplicated().any():
                raise ValueError('Bootstrap unit must be a unique patient')
            probabilities = table[[f'probability_{c}' for c in config['class_names']]].to_numpy()
            truth = table.true_label.to_numpy()
            low, high = bootstrap_f1(truth, probabilities, cfg['bootstrap_repeats'], cfg['bootstrap_seed'])
            metric = read_json(directory / 'metrics.json')['metrics']
            rows.append({'model': name, 'seed': seed, 'primary_seed': seed == cfg['final_seeds'][0],
                         **metric, 'f1_ci_low': low, 'f1_ci_high': high,
                         'brier_multiclass': float(np.mean(np.sum((probabilities - np.eye(3)[truth])**2, axis=1)))})
    data = pd.DataFrame(rows)
    out = run / 'reports'
    data.to_csv(out / 'historical_test_all_seeds.csv', index=False)
    data.groupby('model').agg(f1_mean=('f1', 'mean'), f1_sd=('f1', 'std')).to_csv(out / 'seed_variation.csv')
    primary = data[data.primary_seed].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(primary))
    ax.scatter(x, primary.f1, label='Primary seed 42')
    ax.vlines(x, primary.f1_ci_low, primary.f1_ci_high, label='95% stratified patient bootstrap CI')
    ax.set_xticks(x, primary.model); ax.set_ylim(0, 1); ax.set_ylabel('Patient macro-F1')
    ax.set_title('Historical test; not external validation'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'historical_test_uncertainty.png', dpi=160); plt.close(fig)
    return data
