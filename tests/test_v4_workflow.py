"""Endpoint paths, completed-grid selection, and executable manual notebooks."""
import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import nbformat
import numpy as np
import pandas as pd
import torch
import yaml
from src.artifacts import read_json, write_json
from research_v4.common import cv_directory, progress, scenarios, require_final_complete
from research_v4.experiment import fit_case, summarize, stability, evaluate

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = yaml.safe_load((ROOT / 'research_v4/protocol.yaml').read_text())

    def test_notebooks_valid_executable_no_hidden_skip_flags(self):
        files = sorted((ROOT / 'notebooks').glob('2[0-7]_*.ipynb'))
        self.assertEqual(len(files), 8)
        for path in files:
            book = nbformat.read(path, as_version=4); nbformat.validate(book)
            for cell in book.cells:
                if cell.cell_type == 'code':
                    ast.parse(cell.source)
                    self.assertNotIn('= False', cell.source)
                    # User execution outputs are research artefacts to preserve.

    def test_complete_grid_selects_development_and_reuses_stability_case(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            run = Path(tmp)
            for model in self.cfg['models']:
                for pooling in self.cfg['pooling']:
                    for sc in scenarios(self.cfg):
                        for fold in range(1, 6):
                            path = cv_directory(run, model, pooling, sc['id'], fold, 42)
                            value = .6 + .05 * (pooling == 'attention') + .01 * (sc['id'] == 'S05')
                            write_json(path / 'metrics.json', {'metrics': {'f1': value, 'accuracy': .7},
                                'best_epoch': fold, 'epochs_run': 6, 'validation_missing_predicted_classes': 0})
                            write_json(path / 'config.json', {'fixture': True})
                for fold in range(1, 6):
                    for seed in (42, 43, 44):
                        write_json(run / 'diagnostic/Adam' / model / f'fold_{fold}' / f'seed_{seed}/metrics.json', {})
            # Historical test scores deliberately do not participate in selection.
            write_json(run / 'test/encoder/seed_42/metrics.json', {'metrics': {'f1': .99}})
            summarize(run, self.cfg, select=True)
            decision = read_json(run / 'selection.json')
            self.assertEqual(len(decision['choices']), 5)
            self.assertTrue(all(c['scenario'] == 'S05' and c['pooling'] == 'attention' for c in decision['choices']))
            self.assertTrue(all(c['epochs'] == 3 for c in decision['choices']))
            self.assertFalse(decision['test_used_for_selection']); self.assertEqual(decision['primary_seed'], 42)
            with patch('research_v4.experiment.fit_case') as fit_mock:
                stability(run, run, self.cfg, {}, torch.device('cpu'), ['encoder'])
                self.assertEqual(fit_mock.call_count, 10)
            self.assertTrue((run / 'stability/encoder/fold_1/seed_42/metrics.json').exists())
            # No final model yet: gate must fail before loading any test signals.
            with patch('research_v4.experiment.load_pool') as data:
                with self.assertRaises(RuntimeError):
                    evaluate(run, run, self.cfg, {}, torch.device('cpu'))
                data.assert_not_called()

    def test_small_fixture_cv_and_final_paths_are_consistent(self):
        from tests.test_research_v4 import LEGACY, NEW
        cfg = {**self.cfg, 'segment_length': 128, 'new_model': NEW,
               'training': {**self.cfg['training'], 'epochs': 1, 'patience': 1}}
        config = {'model': LEGACY, 'class_names': ['a', 'b', 'c']}
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            run, source = Path(tmp) / 'run', Path(tmp) / 'source'
            values = np.random.default_rng(3).normal(size=(12, 12, 128)).astype('float32')
            ids = [f'p{i}' for i in range(6)]
            table = pd.DataFrame({'hospital_id': np.repeat(ids, 2), 'label': np.repeat([0, 1, 2, 0, 1, 2], 2), 'valid_samples': 128})
            directory = source / 'data/development'; directory.mkdir(parents=True)
            np.save(directory / 'X.npy', values); table.to_csv(directory / 'segments.csv', index=False)
            write_json(run / 'protocol.json', {'experiment_hash': 'fixture', 'development_cache': {}})
            write_json(run / 'data/patients.json', [{'hospital_id': hid, 'label': i % 3, 'pool': 'development'} for i, hid in enumerate(ids)])
            write_json(run / 'folds/manifest.json', [{'fold': i + 1, 'train_ids': ids[1:], 'validation_ids': [ids[0]]} for i in range(5)])
            sc = scenarios(cfg)[0]
            result = fit_case(run, source, cfg, config, 'residual_multiscale', 'attention', sc, torch.device('cpu'), fold=1)
            self.assertEqual(result['n_validation_patients'], 1)
            cv = cv_directory(run, 'residual_multiscale', 'attention', 'S01', 1, 42)
            self.assertTrue((cv / 'validation_predictions.csv').exists())
            self.assertEqual(read_json(cv / 'config.json')['device'], 'cpu')
            result = fit_case(run, source, cfg, config, 'residual_multiscale', 'attention', sc,
                              torch.device('cpu'), seed=42, phase='final', epochs=1)
            self.assertTrue(result['fit_complete'])
            self.assertTrue((run / 'final/residual_multiscale/seed_42/model.pth').is_file())
            self.assertFalse((run / 'final/residual_multiscale/final').exists())
            with self.assertRaises(RuntimeError):
                require_final_complete(run, cfg)
            # Read-only analysis must also work with only part of the grid.
            from research_v4.analysis import analyze
            summarize(run, cfg)
            analyze(run, cfg, config)
            status = read_json(run / 'reports/status.json')
            self.assertEqual(status['cv']['complete'], 1)
            self.assertEqual(status['final_completed'], 1)
            self.assertEqual(status['historical_test_completed'], 0)

    def test_native_length_model_inference_is_supported(self):
        from research_v4.models import build_model
        from research_v3.common import context
        _, _, _, config, _, _ = context()
        for name in self.cfg['models']:
            model = build_model(name, 'attention', config['model'], self.cfg['new_model']).eval()
            with torch.no_grad():
                output = model(torch.zeros(1, 1, 12, 5000))
            self.assertEqual(output.shape, (1, 3))
            self.assertTrue(torch.isfinite(output).all())
            self.assertAlmostEqual(float(output.exp().sum()), 1, places=5)


if __name__ == '__main__':
    unittest.main()
