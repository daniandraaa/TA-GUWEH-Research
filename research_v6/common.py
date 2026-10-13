"""Bind data, split, runtime and new code without touching V4/V5 artifacts."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import importlib.metadata
import platform
import yaml
from sklearn.model_selection import train_test_split
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from research_v5.common import context as v5_context
from research_v4.common import disk_check

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = '20261004_conditional_gan_qc_v6'


def inner_splits(records, folds, fraction, seed):
    labels = {r['hospital_id']: r['label'] for r in records}
    output = []
    for fold in folds:
        ids = sorted(fold['train_ids'])
        fit, reference = train_test_split(ids, test_size=fraction,
            stratify=[labels[x] for x in ids], random_state=seed + fold['fold'])
        fit, reference = sorted(fit), sorted(reference)
        outer = sorted(fold['validation_ids'])
        if set(fit) & set(reference) or set(fit + reference) != set(ids) or set(ids) & set(outer):
            raise RuntimeError('GAN fit/reference/outer validation overlap')
        if set(labels[x] for x in fit) != {0, 1, 2} or set(labels[x] for x in reference) != {0, 1, 2}:
            raise RuntimeError('Each inner partition must contain all three classes')
        output.append({'fold': fold['fold'], 'gan_fit_ids': fit, 'gan_reference_ids': reference,
            'outer_validation_ids': outer,
            'fit_class_patients': dict(Counter(labels[x] for x in fit)),
            'reference_class_patients': dict(Counter(labels[x] for x in reference))})
    return output


@dataclass
class Study:
    run: Path
    source: Path
    protocol: dict
    records: list
    folds: list
    identity: dict

    @property
    def signature(self):
        return canonical_hash(self.identity)

    def folder(self, fold):
        return self.run / f'fold_{fold}'


def context(run_id=DEFAULT_RUN, initialize=False):
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run-id must be a single directory name')
    p = yaml.safe_load((PACKAGE / 'protocol.yaml').read_text())
    if (p['folds'] != [1, 2, 3, 4, 5] or p['sampling_rate'] != 2000 or p['segment_length'] != 5000
            or p['classes'] != 3 or p['clipping'] or p['test_signals_enabled']
            or p['outer_validation_signals_enabled'] or p['classifier_training_enabled']
            or p['training']['discriminator_steps_per_generator_step'] != 1
            or p['generator']['independent_leads'] != ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            or p['generator']['output'] != 'linear_eight_leads_then_algebraic_twelve_leads'
            or p['generator']['normalization'] != 'GroupNorm'
            or p['discriminator'] != 'spectral_normalized_projection_hinge'
            or p['amplitude_scale'] != 'median_inner_fit_patient_global_rms'
            or p['training']['sampling'] != 'uniform_class_then_uniform_patient_then_uniform_window'
            or p['training']['checkpoint_selection'] != 'fixed_last_epoch_no_classifier_scores'
            or not p['full_fold_training_requires_fold1_quality_pass']):
        raise ValueError('This endpoint is only the fixed native development GAN quality pilot')
    old = v5_context(p['source_v5_run'])
    folds = inner_splits(old.records, old.folds, p['inner_reference_fraction'], p['inner_split_seed'])
    identity = {'protocol': p, 'source_v5_hash': old.signature,
        'development_cache': old.identity['development_cache'],
        'development_records_hash': canonical_hash(old.records), 'inner_folds': folds,
        'code': {f.name: sha256_file(f) for f in sorted(PACKAGE.glob('*.py'))},
        'runner_sha256': sha256_file(ROOT / 'notebooks/_v6_runner.py'),
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
            **{n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'PyYAML', 'matplotlib')}},
        'test_signals_read': False, 'outer_validation_signals_read': False}
    study = Study(ROOT / 'results/research_v6' / run_id, old.source, p, old.records, folds, identity)
    path = study.run / 'protocol.json'
    expected = {**identity, 'experiment_hash': study.signature}
    # JSON object keys in class counts are normalized before comparing saved content.
    import json
    expected = json.loads(json.dumps(expected))
    if not path.exists():
        if not initialize:
            raise RuntimeError('Jalankan notebook33 / prepare dahulu')
        disk_check(study.run, p)
        with lock(study.run / '.initialize.lock'):
            if not path.exists():
                write_json(study.run / 'folds.json', folds)
                write_json(path, expected)
    if read_json(path) != expected or canonical_hash(read_json(study.run / 'folds.json')) != canonical_hash(expected['inner_folds']):
        raise RuntimeError('Kode/protokol/data/runtime V6 berubah. Pertahankan run lama, gunakan run-id baru.')
    return study


def write_manifest(path, value):
    write_json(path, {**value, 'record_hash': canonical_hash(value)})


def verify_files(directory, saved):
    unsigned = {k: v for k, v in saved.items() if k != 'record_hash'}
    if saved.get('record_hash') != canonical_hash(unsigned):
        raise RuntimeError('V6 manifest content changed')
    for name, digest in saved['files'].items():
        if not (directory / name).is_file() or sha256_file(directory / name) != digest:
            raise RuntimeError(f'Frozen V6 artifact changed: {directory / name}')


def require_prepared(study, fold):
    directory = study.folder(fold)
    saved = read_json(directory / 'preparation.json')
    if saved['experiment_hash'] != study.signature or saved['fold'] != fold:
        raise RuntimeError('Preparation identity changed')
    verify_files(directory, saved)
    return saved


def require_quality(study, fold):
    directory = study.folder(fold)
    require_prepared(study, fold)
    completed = read_json(directory / 'completed.json')
    if completed['experiment_hash'] != study.signature:
        raise RuntimeError('Completed GAN identity changed')
    verify_files(directory, completed)
    generated = read_json(directory / 'generated.json')
    if generated['experiment_hash'] != study.signature or generated['generator_sha256'] != sha256_file(directory / 'generator.pth'):
        raise RuntimeError('Generated samples do not belong to this generator')
    verify_files(directory, generated)
    saved = read_json(directory / 'quality.json')
    if saved['experiment_hash'] != study.signature or saved['generated_manifest_sha256'] != sha256_file(directory / 'generated.json'):
        raise RuntimeError('Quality provenance changed')
    verify_files(directory, saved)
    if not saved['passed']:
        raise RuntimeError(f'Fold{fold} gagal screening mutu. Baca quality.json dan visualisasi; jangan lanjut classifier/full fold.')
    return saved
