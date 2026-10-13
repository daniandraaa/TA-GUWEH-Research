"""Identitas run V14: data, fold, GAN dan skenario dibaca dari run V13 yang sudah dibekukan."""
from dataclasses import dataclass
from pathlib import Path
import json
import yaml
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from research_v4.common import disk_check
from research_v13.common import all_scenarios, context as v13_context

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = '20261008_binary_lvot_rvot_cv_v14'


@dataclass
class Study:
    run: Path
    v13: object
    protocol: dict
    scenarios: dict
    identity: dict

    @property
    def signature(self):
        return canonical_hash(self.identity)

    @property
    def class_names(self):
        return self.protocol['class_names']

    @property
    def model_config(self):
        return {**self.v13.config['model'], 'n_classes': len(self.class_names)}


def validate(p):
    if (p['models'] != ['encoder', 'patchtst', 'hierarchical'] or p['arms'] != ['baseline', 'finetune']
            or p['folds'] != [1, 2, 3, 4, 5] or p['cv_seeds'] != [42, 43, 44]
            or p['label_map'] != {0: 0, 1: 1, 2: 1} or p['class_names'] != ['LVOT', 'RVOT'] or p['test_pool_read']):
        raise ValueError('Protokol V14 hanya mendukung desain biner yang dipra-registrasi')


def context(run_id=DEFAULT_RUN, initialize=False):
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run-id harus satu nama folder')
    p = yaml.safe_load((PACKAGE / 'protocol.yaml').read_text())
    validate(p)
    v13 = v13_context(p['source_v13_run'])
    selection = read_json(v13.run / 'selection.json')
    if selection['experiment_hash'] != v13.signature or selection['test_used_for_selection']:
        raise RuntimeError('Seleksi V13 tidak sah untuk dipakai V14')
    known = {s['id']: s for s in all_scenarios(v13.protocol)}
    scenarios = {m: {a: known[selection['choices'][m][a]['scenario']] for a in p['arms']} for m in p['models']}
    identity = {'protocol': p, 'source_v13_hash': v13.signature, 'source_v13_selection_hash': canonical_hash(selection),
                'scenarios': scenarios, 'development_cache': v13.identity['development_cache'],
                'model_config': {**v13.config['model'], 'n_classes': 2}, 'class_names': p['class_names'],
                'code': {q.name: sha256_file(q) for q in sorted(PACKAGE.glob('*.py'))},
                'runtime': v13.identity['runtime'], 'test_pool_read': False}
    study = Study(ROOT / 'results/research_v14' / run_id, v13, p, scenarios, identity)
    expected = json.loads(json.dumps({**identity, 'experiment_hash': study.signature}))
    path = study.run / 'protocol.json'
    if not path.exists():
        if not initialize:
            raise RuntimeError('Jalankan tahap prepare V14 dahulu')
        disk_check(study.run, p)
        with lock(study.run / '.initialize.lock'):
            if not path.exists():
                write_json(path, expected)
    if read_json(path) != expected:
        raise RuntimeError('Kode/protokol/data/runtime V14 berubah. Pertahankan run lama; gunakan run-id baru.')
    return study


def cv_directory(study, model, arm, fold, seed):
    return study.run / 'cv' / model / arm / f'fold_{fold}' / f'seed_{seed}'


def cv_cases(study, models=None, arms=None, folds=None, seeds=None):
    p = study.protocol
    for m in models or p['models']:
        for a in arms or p['arms']:
            for f in folds or p['folds']:
                for seed in seeds or p['cv_seeds']:
                    yield m, a, f, seed


def progress(study):
    cases = list(cv_cases(study))
    missing = [f'{m}/{a}/fold{f}/seed{s}' for m, a, f, s in cases
               if not (cv_directory(study, m, a, f, s) / 'metrics.json').is_file()]
    return {'experiment_hash': study.signature, 'cv_complete': len(cases) - len(missing), 'cv_expected': len(cases),
            'scenarios': {m: {a: s['id'] for a, s in v.items()} for m, v in study.scenarios.items()},
            'comparison_written': (study.run / 'reports/decision.json').is_file(), 'cv_missing_first': missing[:10]}
