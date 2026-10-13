"""Optional annotated INCART benchmark for the detector, not PVC-site models.

Run manually. Download requires an explicit --download flag and wfdb installed.
Record selection is fixed before scoring; no threshold optimization on INCART.
"""
import argparse
import copy
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from src.data import LEADS
from .candidates import detect_candidates, consensus_matches

BEAT_SYMBOLS = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q'}


def score_peaks(reference, detected, fs, tolerance_seconds=.1):
    pairs = consensus_matches(np.sort(reference), np.sort(detected), round(fs * tolerance_seconds))
    tp, fp, fn = len(pairs), len(detected) - len(pairs), len(reference) - len(pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'median_absolute_timing_ms': float(np.median([abs(a-b)*1000/fs for a,b in pairs])) if pairs else None}, pairs


def benchmark(data_dir, output_dir, download=False, records=None):
    # Notebook 17 optionally installs only QA dependencies into this target,
    # preserving the frozen training environment's packages.
    qa_dependencies = ROOT / '.ecgqa'
    if qa_dependencies.is_dir():
        sys.path.insert(0, str(qa_dependencies))
    try:
        import wfdb
    except ImportError as error:
        raise RuntimeError('Benchmark opsional memerlukan wfdb/dependensinya. Jalankan cell instalasi manual notebook 17 dahulu.') from error
    records = records or [f'I{i:02}' for i in range(1, 76)]
    if len(set(records)) != len(records) or any(r not in {f'I{i:02}' for i in range(1, 76)} for r in records):
        raise ValueError('Expected unique INCART record names I01..I75')
    data_dir, output = Path(data_dir), Path(output_dir)
    if download:
        data_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database('incartdb', str(data_dir), records=records, annotators=['atr'])
    missing = [str(data_dir / f'{r}.{ext}') for r in records for ext in ['hea', 'dat', 'atr']
               if not (data_dir / f'{r}.{ext}').exists()]
    if missing:
        raise RuntimeError('INCART files missing; use --download manually or supply files: ' + ', '.join(missing[:3]))
    spec = yaml.safe_load((Path(__file__).parent / 'protocol.yaml').read_text())['candidate']
    identity = {'records': records, 'spec_at_2000hz': spec, 'dataset': 'INCART 1.0.0',
                'purpose': 'all-beat QRS detection; not PVC classification or anatomical validation',
                'tolerance_seconds': .1, 'wfdb_version': wfdb.__version__,
                'algorithm_sha256': sha256_file(Path(__file__).parent / 'candidates.py'),
                'benchmark_sha256': sha256_file(Path(__file__)),
                'assets': {f'{r}.{ext}': sha256_file(data_dir / f'{r}.{ext}')
                           for r in records for ext in ['hea', 'dat', 'atr']}}
    signature = canonical_hash(identity)
    output = output / signature[:12]
    with lock(output / '.lock'):
        write_json(output / 'protocol.json', {**identity, 'signature': signature})
        rows = []
        for r in records:
            path = output / f'{r}.json'
            if path.exists():
                rows.append(read_json(path))
                continue
            record = wfdb.rdrecord(str(data_dir / r))
            annotation = wfdb.rdann(str(data_dir / r), 'atr')
            lookup = {name.upper(): i for i, name in enumerate(record.sig_name)}
            if any(name.upper() not in lookup for name in LEADS):
                raise ValueError(f'Unexpected lead set: {record.sig_name}')
            raw = record.p_signal[:, [lookup[n.upper()] for n in LEADS]]
            current = copy.deepcopy(spec)
            current.update(fs=float(record.fs), maximum_per_patient=int(record.sig_len))
            peaks, qa = detect_candidates(raw, current)
            # Both reference and predictions use complete-crop eligibility;
            # report the number of edge annotations excluded by this rule.
            before, after = round(record.fs * spec['before_seconds']), round(record.fs * spec['after_seconds'])
            beat = np.array([s in BEAT_SYMBOLS for s in annotation.symbol])
            eligible = (annotation.sample >= before) & (annotation.sample + after <= len(raw))
            reference = annotation.sample[beat & eligible]
            metrics, pairs = score_peaks(reference, peaks, record.fs)
            ventricular = set(annotation.sample[eligible & np.array([s == 'V' for s in annotation.symbol])])
            hit = len({a for a,b in pairs} & ventricular)
            row = {'record': r, 'sampling_rate': record.fs, **metrics,
                   'n_ventricular_reference_beats': len(ventricular), 'n_ventricular_detected': hit,
                   'ventricular_detection_recall': hit/len(ventricular) if ventricular else None,
                   'excluded_edge_reference_beats': int((beat & ~eligible).sum()),
                   'detector_agreement': qa['agreement']}
            write_json(path, row); rows.append(row)
            print(f'INCART {r}: F1={metrics["f1"]:.4f}, PVC-reference detected {hit}/{len(ventricular)}', flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(output / 'per_record.csv', index=False)
        tp, fp, fn = [int(frame[k].sum()) for k in ['tp','fp','fn']]
        summary = {'n_records': len(rows), 'tp': tp, 'fp': fp, 'fn': fn,
                   'micro_precision': tp/(tp+fp) if tp+fp else 0.0,
                   'micro_recall': tp/(tp+fn) if tp+fn else 0.0,
                   'micro_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                   'site_classifier_external_validation': False,
                   'thresholds_optimized_on_benchmark': False}
        write_json(output / 'summary.json', summary)
    print(f'INCART report: {output}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--records', nargs='+')
    args = parser.parse_args()
    benchmark(args.data_dir, args.output_dir, args.download, args.records)
