"""Read-only audit and descriptive analysis of a completed INCART benchmark.

Checks saved counts against annotations and recomputes metrics, not detector
peaks. Bootstrap resamples complete source-Holter groups identified by headers.
No training, download, threshold optimization or new detector inference.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import yaml

from research_v3.common import context
from src.artifacts import ROOT, canonical_hash, read_json, sha256_file, write_json
from src.data import LEADS


BEAT_SYMBOLS = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count_metrics(counts):
    """Last axis: TP, FP, FN, V-reference, V-detected; scalar or batched."""
    counts = np.asarray(counts, dtype=float)
    tp, fp, fn, vr, vh = np.moveaxis(counts, -1, 0)
    nonv_reference = tp + fn - vr
    def ratio(a, b):
        return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=b > 0)
    return {'micro_precision': ratio(tp, tp + fp), 'micro_recall': ratio(tp, tp + fn),
            'micro_f1': ratio(2 * tp, 2 * tp + fp + fn),
            'ventricular_detection_recall': ratio(vh, vr),
            'non_V_annotation_detection_recall': ratio(tp - vh, nonv_reference)}


def cluster_bootstrap(groups, repeats=5000, seed=20261002):
    groups = np.asarray(groups, dtype=float)
    require(groups.ndim == 2 and groups.shape[1] == 5 and len(groups) > 0, 'Expected nonempty [Holter,5] counts')
    require(np.isfinite(groups).all() and (groups >= 0).all(), 'Invalid cluster counts')
    require((groups[:, 4] <= groups[:, 3]).all() and (groups[:, 4] <= groups[:, 0]).all()
            and (groups[:, 3] <= groups[:, 0] + groups[:, 2]).all()
            and ((groups[:, 3] - groups[:, 4]) <= groups[:, 2]).all(), 'Impossible V/reference counts')
    require(repeats > 0, 'Bootstrap repeats must be positive')
    rng = np.random.default_rng(seed)
    samples = groups[rng.integers(0, len(groups), size=(repeats, len(groups)))].sum(axis=1)
    point, draws = count_metrics(groups.sum(axis=0)), count_metrics(samples)
    rows = []
    for metric, array in draws.items():
        finite = array[np.isfinite(array)]
        bounds = np.quantile(finite, [.025, .975]) if len(finite) else [None, None]
        rows.append({'metric': metric, 'point_estimate': float(point[metric]) if np.isfinite(point[metric]) else None,
                     'ci95_lower': None if bounds[0] is None else float(bounds[0]),
                     'ci95_upper': None if bounds[1] is None else float(bounds[1]),
                     'valid_draws': len(finite)})
    return rows


def analyze(benchmark, data_dir=None):
    benchmark = Path(benchmark).resolve()
    data_dir = Path(data_dir or ROOT / 'external_data/incartdb_1.0.0').resolve()
    protocol = read_json(benchmark / 'protocol.json')
    identity = {key: value for key, value in protocol.items() if key != 'signature'}
    require(canonical_hash(identity) == protocol['signature'] and benchmark.name == protocol['signature'][:12], 'Benchmark signature mismatch')
    records = [f'I{i:02}' for i in range(1, 76)]
    require(protocol['records'] == records, 'Expected all 75 records in fixed order')
    expected_assets = {f'{r}.{ext}' for r in records for ext in ['hea', 'dat', 'atr']}
    require(set(protocol['assets']) == expected_assets, 'Incomplete benchmark asset manifest')
    for name, expected in protocol['assets'].items():
        require(sha256_file(data_dir / name) == expected, f'Changed dataset file: {name}')
    for key, name in [('algorithm_sha256', 'candidates.py'), ('benchmark_sha256', 'benchmark.py')]:
        require(sha256_file(ROOT / 'research_v3' / name) == protocol[key], f'Changed frozen code: {name}')
    spec = yaml.safe_load((ROOT / 'research_v3/protocol.yaml').read_text())['candidate']
    require(protocol['spec_at_2000hz'] == spec, 'Changed detector specification')
    require(protocol['tolerance_seconds'] == .1, 'Changed matching tolerance')
    run, *_ = context()  # Also checks frozen training/source identities.
    sys.path.insert(0, str(ROOT / '.ecgqa'))
    import wfdb
    require(wfdb.__version__ == protocol['wfdb_version'], 'Different WFDB annotation-reader version')
    frame = pd.read_csv(benchmark / 'per_record.csv')
    require(frame.record.tolist() == records and frame.record.is_unique, 'Incomplete/duplicate/reordered CSV records')
    require({p.stem for p in benchmark.glob('I*.json')} == set(records), 'Incomplete per-record JSON inventory')
    count_columns = ['tp', 'fp', 'fn', 'n_ventricular_reference_beats', 'n_ventricular_detected']
    counts = frame[count_columns + ['excluded_edge_reference_beats']].to_numpy(dtype=float)
    require(np.isfinite(counts).all() and (counts >= 0).all() and np.equal(counts, np.floor(counts)).all(), 'Invalid beat counts')
    metric_error = 0.0
    holters = []
    outside_reference_counts = []
    for i, r in enumerate(records):
        row = frame.iloc[i]
        saved = read_json(benchmark / f'{r}.json')
        require(set(saved) == set(frame.columns), f'CSV/JSON column mismatch: {r}')
        for key, value in saved.items():
            if value is None:
                require(pd.isna(row[key]), f'CSV/JSON null mismatch: {r}/{key}')
            elif isinstance(value, str):
                require(row[key] == value, f'CSV/JSON text mismatch: {r}/{key}')
            else:
                require(np.isclose(row[key], value, rtol=0, atol=1e-12), f'CSV/JSON numeric mismatch: {r}/{key}')
        header = wfdb.rdheader(str(data_dir / r))
        require(header.fs == row.sampling_rate == 257 and header.n_sig == 12, f'Wrong header sampling/leads: {r}')
        require({s.upper() for s in header.sig_name} == {s.upper() for s in LEADS}, f'Wrong lead names: {r}')
        numbers = [int(m.group(1)) for c in header.comments if (m := re.fullmatch(r'patient\s+(\d+)', c.strip(), re.I))]
        require(len(numbers) == 1, f'Missing source-Holter identifier: {r}')
        holters.append(numbers[0])
        ann = wfdb.rdann(str(data_dir / r), 'atr')
        require(np.isfinite(ann.sample).all() and (np.diff(ann.sample) >= 0).all(), f'Invalid annotation order: {r}')
        eligible = (ann.sample >= round(header.fs * spec['before_seconds'])) & (ann.sample + round(header.fs * spec['after_seconds']) <= header.sig_len)
        beat = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
        ventricular = np.array([s == 'V' for s in ann.symbol])
        # Some source annotations precede the clip's first sample. They are
        # excluded by the original complete-crop rule, not altered or dropped
        # from eligible reference counts during this audit.
        outside = (ann.sample < 0) | (ann.sample >= header.sig_len)
        require(not (outside & eligible).any(), f'Out-of-waveform annotation entered eligible references: {r}')
        outside_reference_counts.append(int((outside & beat).sum()))
        require(int((beat & eligible).sum()) == row.tp + row.fn, f'Reference count mismatch: {r}')
        require(int((ventricular & eligible).sum()) == row.n_ventricular_reference_beats, f'V annotation count mismatch: {r}')
        require(int((beat & ~eligible).sum()) == row.excluded_edge_reference_beats, f'Excluded edge count mismatch: {r}')
        recomputed = count_metrics(row[count_columns].to_numpy(dtype=float))
        for source_name, stored_name in [('micro_precision', 'precision'), ('micro_recall', 'recall'), ('micro_f1', 'f1'), ('ventricular_detection_recall', 'ventricular_detection_recall')]:
            value = recomputed[source_name]
            if not np.isfinite(value):
                require(pd.isna(row[stored_name]), f'Undefined metric should be null: {r}')
            else:
                error = abs(float(value) - float(row[stored_name]))
                metric_error = max(metric_error, error)
                require(error < 1e-12, f'Wrong record metric: {r}/{stored_name}')
        require(0 <= row.detector_agreement <= 1, f'Invalid agreement: {r}')
        if row.tp:
            require(0 <= row.median_absolute_timing_ms <= round(header.fs * .1) * 1000 / header.fs, f'Invalid matched timing median: {r}')
    frame['source_holter_id'] = holters
    frame['out_of_waveform_reference_beats'] = outside_reference_counts
    require(set(holters) == set(range(1, 33)), 'Expected 32 source-Holter groups')
    totals = frame[count_columns].sum().to_numpy(dtype=float)
    metrics = count_metrics(totals)
    summary = read_json(benchmark / 'summary.json')
    require(summary['n_records'] == 75, 'Wrong summary record count')
    for key in ['tp', 'fp', 'fn']:
        require(summary[key] == int(frame[key].sum()), f'Wrong summary {key}')
    for key in ['micro_precision', 'micro_recall', 'micro_f1']:
        require(abs(float(metrics[key]) - summary[key]) < 1e-12, f'Wrong summary metric: {key}')
    require(summary['site_classifier_external_validation'] is False and summary['thresholds_optimized_on_benchmark'] is False, 'Unsupported validation/tuning claim')
    groups = frame.groupby('source_holter_id')[count_columns].sum()
    uncertainty = cluster_bootstrap(groups.to_numpy(), repeats=5000, seed=20261002)
    output = benchmark / 'analysis_20261002'
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'record_diagnostics.csv', index=False)
    groups.to_csv(output / 'source_holter_counts.csv')
    frame.sort_values('f1').to_csv(output / 'records_by_f1.csv', index=False)
    frame[frame.n_ventricular_reference_beats > 0].sort_values(['ventricular_detection_recall', 'record']).to_csv(output / 'records_by_V_recall.csv', index=False)
    pd.DataFrame(uncertainty).to_csv(output / 'source_holter_bootstrap.csv', index=False)
    notebook = read_json(ROOT / 'notebooks/17_Benchmark_Detektor_INCART.ipynb')
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    report = {'audited_utc': datetime.now(timezone.utc).isoformat(), 'benchmark_signature': protocol['signature'],
        'audit_code_sha256': sha256_file(__file__), 'protocol_sha256': sha256_file(benchmark / 'protocol.json'),
        'summary_sha256': sha256_file(benchmark / 'summary.json'), 'per_record_csv_sha256': sha256_file(benchmark / 'per_record.csv'),
        'training_experiment_hash': read_json(run / 'protocol.json')['experiment_hash'],
        'records_verified': 75, 'source_holter_groups': len(groups), 'dataset_assets_sha256_verified': len(expected_assets),
        'annotation_reference_counts_verified': 75, 'maximum_saved_record_metric_error': metric_error,
        'notebook_executed_cells': sum(c.get('execution_count') is not None for c in cells),
        'notebook_code_cells': len(cells), 'notebook_saved_errors': sum(o['output_type'] == 'error' for c in cells for o in c.get('outputs', [])),
        'tp': int(totals[0]), 'fp': int(totals[1]), 'fn': int(totals[2]),
        **{k: float(v) for k, v in metrics.items()},
        'V_reference_beats': int(totals[3]), 'V_detected': int(totals[4]), 'V_missed': int(totals[3] - totals[4]),
        'V_share_of_all_missed': float((totals[3] - totals[4]) / totals[2]),
        'excluded_edge_reference_beats': int(frame.excluded_edge_reference_beats.sum()),
        'out_of_waveform_reference_beats_in_excluded_edges': int(frame.out_of_waveform_reference_beats.sum()),
        'records_with_out_of_waveform_reference': frame.loc[frame.out_of_waveform_reference_beats > 0, 'record'].tolist(),
        'record_mean_f1': float(frame.f1.mean()), 'record_median_f1': float(frame.f1.median()),
        'record_minimum_f1': float(frame.f1.min()), 'records_without_V': frame.loc[frame.n_ventricular_reference_beats == 0, 'record'].tolist(),
        'median_of_record_matched_timing_medians_ms': float(frame.median_absolute_timing_ms.median()),
        'source_holter_bootstrap': {'repeats': 5000, 'seed': 20261002, 'groups': 32, 'ci': uncertainty,
            'scope': 'complete source-Holter groups, fixed detector and observed cohort; no independent-beat resampling'},
        'reference_symbols': sorted(BEAT_SYMBOLS), 'matching_tolerance_seconds_requested': .1,
        'matching_tolerance_samples': round(257 * .1), 'detector_peaks_or_matches_recomputed': False,
        'classifier_training_run': False, 'site_classifier_external_validation': False,
        'local_PVC_annotations_validated': False, 'benchmark_64_crop_cap_applied': False}
    write_json(output / 'audit.json', report)
    plot(frame, report, output)
    print(f'INCART audit: 75 records, 32 source Holters, 225 hashes and annotation counts verified; F1={metrics["micro_f1"]:.6f}; V recall={metrics["ventricular_detection_recall"]:.6f}', flush=True)
    print('Analysis:', output, flush=True)
    return report


def plot(frame, report, output):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    keys = ['micro_precision', 'micro_recall', 'micro_f1', 'ventricular_detection_recall']
    labels = ['All-beat precision', 'All-beat recall', 'All-beat F1', 'V-annotated beat recall']
    axes[0].barh(labels[::-1], [report[k] for k in keys[::-1]], color=['#bc5a45', '#29608c', '#29608c', '#29608c'])
    for i, key in enumerate(keys[::-1]):
        axes[0].text(report[key] - .02, i, f'{report[key]*100:.2f}%', ha='right', va='center', color='white', weight='bold')
    axes[0].set_xlim(0, 1); axes[0].set_title('Pooled beat counts (75 records)')
    worst = frame.nsmallest(8, 'f1').iloc[::-1]
    axes[1].barh(worst.record, worst.f1, color='#ba7054')
    for i, row in enumerate(worst.itertuples()):
        axes[1].text(row.f1 - .02, i, f'{row.f1:.3f}', ha='right', va='center', color='white')
    axes[1].set_xlim(0, 1); axes[1].set_title('Lowest all-beat F1 records')
    lowv = frame[frame.n_ventricular_reference_beats >= 50].nsmallest(8, 'ventricular_detection_recall').iloc[::-1]
    axes[2].barh(lowv.record, lowv.ventricular_detection_recall, color='#7c708c')
    for i, row in enumerate(lowv.itertuples()):
        axes[2].text(row.ventricular_detection_recall + .01, i, f'{row.n_ventricular_detected}/{row.n_ventricular_reference_beats}', ha='left', va='center', fontsize=9)
    axes[2].set_xlim(0, 1.05); axes[2].set_title('Lowest V recall (at least 50 V beats)')
    for ax in axes:
        ax.set_xlabel('Proportion'); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Frozen INCART detector benchmark: overall performance and failures', fontsize=14)
    fig.text(.5, .015, '75 records from 32 source Holters; 257 Hz; matching ±26 samples (~101 ms). V recall measures beat detection, not PVC classification or anatomical origin.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    fig.savefig(output / 'benchmark_overview.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    analyze(args.benchmark, args.data_dir)
