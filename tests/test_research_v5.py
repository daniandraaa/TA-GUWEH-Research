"""Critical control, readiness, and real endpoint checks on synthetic signals."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import tempfile
import unittest
from unittest.mock import patch
import nbformat
import numpy as np
import pandas as pd
import torch
import yaml
from src.artifacts import canonical_hash, read_json, sha256_file, write_json
from src.evaluation import prediction_table, report
from research_v4.training import class_diagnostics
from research_v5.common import Study, case_directory, case_identity, require_smoke, validate_control
from research_v5.experiment import train
from research_v5.analysis import analyze, checked_result, priority_decision
from tests.test_research_v4 import LEGACY, NEW

ROOT = Path(__file__).resolve().parents[1]


class ResearchV5Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_reuse_rejects_recipe_changes_even_with_recomputed_hash(self):
        base = yaml.safe_load((ROOT / 'research_v4/protocol.yaml').read_text())
        fold = {'fold': 1, 'train_ids': ['a', 'b'], 'validation_ids': ['c']}
        choice = {'model': 'hierarchical', 'pooling': 'mean', 'scenario': 'S05',
                  'learning_rate': .0005, 'batch_size': 32}
        sc = {'id': 'S05', 'learning_rate': .0005, 'batch_size': 32}
        case = {'experiment_hash': 'v4', 'device': 'mps', 'phase': 'cv',
            'model': 'hierarchical', 'pooling': 'mean', 'scenario': sc, 'fold': 1, 'seed': 42,
            'effective_seed': 142, 'training_mode': 'bag', 'train_patient_ids': ['a', 'b'],
            'validation_patient_ids': ['c'], 'settings': {**base['training'], **sc, 'scheduler_t_max': 100},
            'segment_budget': 32, 'patient_batch_size': 8, 'diagnostic_v3_choice': None}
        validate_control({**case, 'case_hash': canonical_hash(case)}, 'v4', base, choice, fold, 42)
        for key, value in [('patient_batch_size', 4), ('validation_patient_ids', ['b']), ('effective_seed', 42)]:
            changed = {**case, key: value}
            with self.assertRaises(RuntimeError):
                validate_control({**changed, 'case_hash': canonical_hash(changed)}, 'v4', base, choice, fold, 42)
        changed = deepcopy(case); changed['settings']['learning_rate'] = .001
        with self.assertRaises(RuntimeError):
            validate_control({**changed, 'case_hash': canonical_hash(changed)}, 'v4', base, choice, fold, 42)

    def test_smoke_device_and_tampering_gate(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            study = SimpleNamespace(run=Path(tmp), signature='sig', identity={'paired_device': 'mps'})
            record = {'experiment_hash': 'sig', 'device': 'mps', 'model': 'encoder',
                'batch_shape': [8, 8, 12, 5000], 'valid_segments': 64,
                'disposable_optimizer_steps': 1, 'passed': True}
            path = study.run / 'smoke/encoder.json'
            write_json(path, {'record': record, 'record_hash': canonical_hash(record)})
            require_smoke(study, 'encoder', torch.device('mps'))
            with self.assertRaises(RuntimeError): require_smoke(study, 'encoder', torch.device('cpu'))
            record['valid_segments'] = 32
            write_json(path, {'record': record, 'record_hash': canonical_hash(record)})
            with self.assertRaises(RuntimeError): require_smoke(study, 'encoder', torch.device('mps'))

    def test_incomplete_comparison_does_not_read_predictions(self):
        study = SimpleNamespace()
        with patch('research_v5.analysis.require_prepared'), patch('research_v5.analysis.progress',
                  return_value={'ready_for_comparison': False, 'complete': 2, 'missing': ['fold2']}), \
             patch('research_v5.analysis.pd.read_csv') as read:
            with self.assertRaises(RuntimeError): analyze(study)
            read.assert_not_called()

    def test_priority_requires_repeated_positive_effect_and_RVOT_recall(self):
        rules = {'minimum_positive_seeds': 2}
        self.assertTrue(priority_decision(.03, 2, .01, rules))
        self.assertFalse(priority_decision(.03, 1, .01, rules))
        self.assertFalse(priority_decision(.03, 3, -.01, rules))
        self.assertFalse(priority_decision(-.01, 2, .01, rules))

    def test_synthetic_k8_fit_and_completed_case_reuse(self):
        protocol = yaml.safe_load((ROOT / 'research_v5/protocol.yaml').read_text())
        base = yaml.safe_load((ROOT / 'research_v4/protocol.yaml').read_text())
        cfg = {**base, 'segment_length': 128, 'segments_per_patient': 8, 'new_model': NEW,
               'minimum_free_gib': 0, 'training': {**base['training'], 'epochs': 2, 'patience': 1}}
        ids = [f'p{i}' for i in range(9)]
        records = [{'hospital_id': p, 'label': i % 3, 'pool': 'development'} for i, p in enumerate(ids)]
        fold = {'fold': 1, 'train_ids': ids[:6], 'validation_ids': ids[6:]}
        folds = [{**fold, 'fold': f} for f in range(1, 6)]
        choices = {m: {'model': m, 'pooling': 'mean' if m == 'hierarchical' else 'attention',
                       'learning_rate': .0005, 'scenario': 'S05', 'batch_size': 32, 'patient_batch_size': 8}
                   for m in protocol['models']}
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            run, source = Path(tmp) / 'run', Path(tmp) / 'source'
            study = Study(run, source, source, protocol, cfg,
                {'model': LEGACY, 'class_names': ['a', 'b', 'c']}, choices, records, folds, [],
                {'paired_device': 'cpu', 'development_cache': {}})
            values = np.random.default_rng(12).normal(size=(81, 12, 128)).astype('float32')
            table = pd.DataFrame({'hospital_id': np.repeat(ids, 9), 'label': np.repeat(np.arange(9) % 3, 9),
                                  'valid_samples': 128})
            target = source / 'data/development'; target.mkdir(parents=True)
            np.save(target / 'X.npy', values); table.to_csv(target / 'segments.csv', index=False)
            write_json(run / 'data/patients.json', records)
            write_json(run / 'folds/manifest.json', folds)
            write_json(run / 'protocol.json', {'experiment_hash': study.signature, 'development_cache': {}})
            with patch('research_v5.experiment.require_prepared'), patch('research_v5.experiment.require_smoke'):
                train(study, ['hierarchical'], [1], [42], torch.device('cpu'), pilot=True)
                path = case_directory(study, 'hierarchical', 1, 42)
                result = read_json(path / 'config.json')
                expected = case_identity(study, 'hierarchical', 1, 42, 'cpu')
                self.assertEqual(result, {**expected, 'case_hash': canonical_hash(expected)})
                self.assertEqual(result['patient_batch_size'], 8)
                self.assertEqual(result['segment_budget'], 64)
                weights = pd.read_csv(path / 'validation_segment_weights.csv')
                self.assertEqual(len(weights), 27)  # all nine windows of each validation patient
                _, verified = checked_result(path, study, 1, table)
                self.assertEqual(verified['epochs_run'], 2)
                self.assertTrue(torch.load(path / 'last.pt', weights_only=False)['completed'])
                before = {p.name: sha256_file(p) for p in path.iterdir() if p.suffix in {'.json', '.csv', '.pt'}}
                with patch('research_v5.experiment.fit_case') as fit_mock:
                    train(study, ['hierarchical'], [1], [42], torch.device('cpu'), pilot=True)
                    fit_mock.assert_not_called()
                self.assertEqual(before, {p.name: sha256_file(p) for p in path.iterdir() if p.suffix in {'.json', '.csv', '.pt'}})

    def test_complete_synthetic_comparison_and_result_freeze(self):
        protocol = yaml.safe_load((ROOT / 'research_v5/protocol.yaml').read_text())
        cfg = yaml.safe_load((ROOT / 'research_v4/protocol.yaml').read_text())
        ids = [f'p{i:02}' for i in range(15)]
        records = [{'hospital_id': p, 'pool': 'development', 'label': i % 3} for i, p in enumerate(ids)]
        folds = [{'fold': f + 1, 'train_ids': ids[:3*f] + ids[3*f+3:],
                  'validation_ids': ids[3*f:3*f+3]} for f in range(5)]
        choices = {m: {'model': m, 'pooling': 'mean' if m == 'hierarchical' else 'attention',
                       'learning_rate': .0005} for m in protocol['models']}
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            run, source = Path(tmp) / 'run', Path(tmp) / 'source'
            study = Study(run, source, source, protocol, cfg, {'class_names': ['a', 'b', 'c']},
                          choices, records, folds, [], {'paired_device': 'cpu'})
            table = pd.DataFrame({'hospital_id': np.repeat(ids, 2), 'label': np.repeat(np.arange(15) % 3, 2)})
            target = source / 'data/development'; target.mkdir(parents=True)
            table.to_csv(target / 'segments.csv', index=False)
            for model in protocol['models']:
                for fold in folds:
                    f = fold['fold']; valid_ids = fold['validation_ids']
                    for seed in protocol['seeds']:
                        origin = source / model / f'fold{f}_seed{seed}'
                        study.controls.append({'model': model, 'fold': f, 'seed': seed,
                                               'origin': str(origin.relative_to(ROOT))})
                        for arm, directory in [('k4', origin), ('k8', case_directory(study, model, f, seed))]:
                            truth = np.arange(3)
                            pred = np.asarray([0, 0, 2]) if arm == 'k4' else truth
                            p = np.full((3, 3), .15); p[np.arange(3), pred] = .7
                            identity = case_identity(study, model, f, seed, 'cpu')
                            write_json(directory / 'config.json', {**identity, 'case_hash': canonical_hash(identity)})
                            result = {**report(truth, p, ['a', 'b', 'c']), **class_diagnostics(truth, p),
                                      'best_epoch': 1, 'epochs_run': 2, 'elapsed_seconds': 2 if arm == 'k4' else 4}
                            write_json(directory / 'metrics.json', result)
                            prediction_table(valid_ids, truth, p, ['a', 'b', 'c']).to_csv(directory / 'validation_predictions.csv', index=False)
                            pd.DataFrame({'epoch': [1, 2], 'validation_f1': [result['metrics']['f1']] * 2}).to_csv(directory / 'history.csv', index=False)
                            indices = np.flatnonzero(table.hospital_id.isin(valid_ids).to_numpy())
                            pd.DataFrame({'hospital_id': table.iloc[indices].hospital_id.to_numpy(),
                                'segment_index': indices, 'weight': .5, 'n_patient_segments': 2}).to_csv(directory / 'validation_segment_weights.csv', index=False)
            with patch('research_v5.analysis.require_prepared'):
                summary = analyze(study)
                self.assertTrue(summary.prioritize_k8_for_further_development.all())
                self.assertTrue((summary.positive_seeds == 3).all())
                self.assertEqual(len(pd.read_csv(run / 'reports/folds_paired.csv')), 30)
                self.assertEqual(len(read_json(run / 'reports/comparison_inputs_sha256.json')), 300)
                self.assertFalse(read_json(run / 'reports/analysis.json')['test_signals_read'])
                self.assertTrue((run / 'reports/k4_k8_development.pdf').exists())
                candidate = case_directory(study, 'hierarchical', 1, 42) / 'metrics.json'
                changed = read_json(candidate); changed['elapsed_seconds'] = 5
                write_json(candidate, changed)
                with self.assertRaisesRegex(RuntimeError, 'Results changed'):
                    analyze(study)

    def test_manual_notebooks_valid_and_cli_has_no_test_stage(self):
        files = [ROOT / 'notebooks' / name for name in (
            '29_Persiapan_Ablation_Window_V5.ipynb', '30_Ablation_Window_Hierarchical_V5.ipynb',
            '31_Ablation_Window_Encoder_V5.ipynb', '32_Analisis_Ablation_Window_V5.ipynb')]
        for path in files:
            book = nbformat.read(path, as_version=4); nbformat.validate(book)
            for cell in book.cells:
                if cell.cell_type == 'code': ast.parse(cell.source)
        from research_v5.__main__ import main
        with patch('sys.argv', ['research_v5', 'test']), patch('research_v5.__main__.context') as ctx:
            with self.assertRaises(SystemExit): main()
            ctx.assert_not_called()


if __name__ == '__main__': unittest.main()
