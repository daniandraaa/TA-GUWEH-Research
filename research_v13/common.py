"""Identitas run V13: data/fold V4, GAN per fold V7, kode dan runtime dibekukan."""
from dataclasses import dataclass
from pathlib import Path
import importlib.metadata
import itertools
import json
import platform
import yaml
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from research_v4.common import context as v4_context, disk_check
from research_v6.common import require_quality

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = '20261008_proposal_baseline_vs_finetune_v13'


@dataclass
class Study:
    run: Path
    source: Path
    protocol: dict
    config: dict
    records: list
    folds: list
    gan_folds: list
    gan: object
    identity: dict

    @property
    def signature(self):
        return canonical_hash(self.identity)

    @property
    def class_names(self):
        return self.config['class_names']


def grid(p):
    return [{'id': f'G{i:02}', 'learning_rate': float(lr), 'batch_size': int(bs), 'augmentation': 'gan'}
            for i, (lr, bs) in enumerate(itertools.product(p['grid']['learning_rates'], p['grid']['batch_sizes']), 1)]


def baseline(p):
    b = p['baseline']
    return {'id': b['id'], 'learning_rate': float(b['learning_rate']), 'batch_size': int(b['batch_size']),
            'augmentation': 'none'}


def all_scenarios(p):
    return [baseline(p), *grid(p)]


def validate(p):
    if (p['models'] != ['encoder', 'patchtst', 'hierarchical'] or p['pooling'] != 'mean'
            or p['folds'] != [1, 2, 3, 4, 5] or p['sampling_rate'] != 2000 or p['segment_length'] != 5000
            or p['segments_per_patient'] != 4 or p['primary_seed'] not in p['final_seeds']
            or p['gan_augmentation']['validation_augmented']
            or p['gan_augmentation']['replacements_per_full_bag'] != 1):
        raise ValueError('Protokol V13 hanya mendukung desain proposal yang dibekukan')
    if any(bs % p['segments_per_patient'] for bs in [*p['grid']['batch_sizes'], p['baseline']['batch_size']]):
        raise ValueError('Batch (anggaran segmen) harus kelipatan segmen per pasien')
    if not any(s['learning_rate'] == p['baseline']['learning_rate'] and s['batch_size'] == p['baseline']['batch_size']
               for s in grid(p)):
        raise ValueError('Konfigurasi baseline harus ada di grid agar efek GAN murni dapat diisolasi')


def context(run_id=DEFAULT_RUN, initialize=False):
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run-id harus satu nama folder')
    p = yaml.safe_load((PACKAGE / 'protocol.yaml').read_text())
    validate(p)
    v4run, source, _, _, config = v4_context(p['source_v4_run'])
    records, folds = read_json(v4run / 'data/patients.json'), read_json(v4run / 'folds/manifest.json')
    from research_v7.common import context as v7_context
    gan = v7_context(p['source_v7_run'])
    for fold, inner in zip(folds, gan.folds):
        if (fold['fold'] != inner['fold']
                or sorted(fold['train_ids']) != sorted(inner['gan_fit_ids'] + inner['gan_reference_ids'])
                or sorted(fold['validation_ids']) != sorted(inner['outer_validation_ids'])):
            raise RuntimeError('Fold GAN V7 tidak identik dengan fold V4')
    gan_files = {}
    for inner in gan.folds:
        f = inner['fold']; require_quality(gan, f)
        for name in ('generated.json', 'quality.json', 'synthetic.npy', 'synthetic_labels.csv'):
            gan_files[f'fold_{f}/{name}'] = sha256_file(gan.folder(f) / name)
    identity = {'protocol': p, 'source_v4_hash': read_json(v4run / 'protocol.json')['experiment_hash'],
        'source_v7_hash': gan.signature, 'source_v7_files': gan_files,
        'development_cache': {n: sha256_file(source / 'data/development' / n) for n in ('X.npy', 'segments.csv', 'manifest.json')},
        'records_hash': canonical_hash(records), 'folds_hash': canonical_hash(folds),
        'model_config': config['model'], 'class_names': config['class_names'],
        'code': {q.name: sha256_file(q) for q in sorted(PACKAGE.glob('*.py'))},
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
            **{n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'shap', 'PyYAML')}},
        'historical_test_previously_observed': True, 'pretrained_weights_used': False}
    study = Study(ROOT / 'results/research_v13' / run_id, source, p, config, records, folds, gan.folds, gan, identity)
    expected = json.loads(json.dumps({**identity, 'experiment_hash': study.signature}))
    path = study.run / 'protocol.json'
    if not path.exists():
        if not initialize:
            raise RuntimeError('Jalankan notebook 69 / tahap prepare dahulu')
        disk_check(study.run, p)
        with lock(study.run / '.initialize.lock'):
            if not path.exists():
                write_json(study.run / 'data/patients.json', records)
                write_json(study.run / 'folds/manifest.json', folds)
                write_json(path, expected)
    if read_json(path) != expected:
        raise RuntimeError('Kode/protokol/data/runtime V13 berubah. Pertahankan run lama; gunakan run-id baru.')
    return study


def cv_directory(study, model, scenario_id, fold, seed):
    return study.run / 'cv' / model / scenario_id / f'fold_{fold}' / f'seed_{seed}'


def final_directory(study, model, arm, seed):
    return study.run / 'final' / model / arm / f'seed_{seed}'


def test_directory(study, model, arm, seed):
    return study.run / 'test' / model / arm / f'seed_{seed}'


def cv_cases(study, models=None, scenario_ids=None, folds=None, seeds=None):
    p = study.protocol
    for m in models or p['models']:
        for s in all_scenarios(p):
            if scenario_ids and s['id'] not in scenario_ids:
                continue
            for f in folds or p['folds']:
                for seed in seeds or p['cv_seeds']:
                    yield m, s, f, seed


def progress(study):
    p = study.protocol
    cv_missing = [f'{m}/{s["id"]}/fold{f}/seed{seed}' for m, s, f, seed in cv_cases(study)
                  if not (cv_directory(study, m, s['id'], f, seed) / 'metrics.json').is_file()]
    n_cv = sum(1 for _ in cv_cases(study))
    finals = [(m, a, s) for m in p['models'] for a in ('baseline', 'finetune') for s in p['final_seeds']]
    final_missing = [f'{m}/{a}/seed{s}' for m, a, s in finals
                     if not (final_directory(study, m, a, s) / 'metrics.json').is_file()]
    test_missing = [f'{m}/{a}/seed{s}' for m, a, s in finals
                    if not (test_directory(study, m, a, s) / 'metrics.json').is_file()]
    gan_qc = study.run / 'gan_final/quality.json'
    return {'experiment_hash': study.signature,
            'cv_complete': n_cv - len(cv_missing), 'cv_expected': n_cv,
            'selection_frozen': (study.run / 'selection.json').is_file(),
            'final_gan_quality_passed': bool(gan_qc.is_file() and read_json(gan_qc)['passed']),
            'final_complete': len(finals) - len(final_missing), 'final_expected': len(finals),
            'test_complete': len(finals) - len(test_missing), 'test_expected': len(finals),
            'cv_missing_first': cv_missing[:10], 'final_missing_first': final_missing[:10]}
