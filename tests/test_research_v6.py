"""Critical isolation, scale, numerical gates and exact resume on synthetic signals."""
from copy import deepcopy
from pathlib import Path
import ast
import tempfile
import unittest
from unittest.mock import patch
import nbformat
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
from src.artifacts import cpu_weights, read_json, seed_everything, sha256_file
from research_v6.common import Study, inner_splits, require_prepared, require_quality, verify_files, write_manifest
from research_v6.data import BalancedPatients, global_scale, reference_rows
from research_v6.models import Discriminator, Generator, twelve_leads
from research_v6.quality import compare_features, features, native_nearest, numerical_screen
from research_v6.training import fit
from research_v6 import experiment

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load((ROOT / 'research_v6/protocol.yaml').read_text())


class ResearchV6Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_inner_patients_are_disjoint_and_deterministic(self):
        records = [{'hospital_id': f'p{i:03}', 'label': i % 3} for i in range(90)]
        ids = [r['hospital_id'] for r in records]
        folds = [{'fold': f, 'train_ids': [x for i, x in enumerate(ids) if i % 5 != f - 1],
                  'validation_ids': [x for i, x in enumerate(ids) if i % 5 == f - 1]} for f in range(1, 6)]
        one = inner_splits(records, folds, .2, 6042)
        self.assertEqual(one, inner_splits(records, folds, .2, 6042))
        for part, original in zip(one, folds):
            fit_ids, ref, outer = (set(part[k]) for k in ('gan_fit_ids', 'gan_reference_ids', 'outer_validation_ids'))
            self.assertFalse(fit_ids & ref or fit_ids & outer or ref & outer)
            self.assertEqual(fit_ids | ref, set(original['train_ids']))
            self.assertEqual(set(part['fit_class_patients']), {0, 1, 2})

    def test_scale_uses_only_fit_rows_and_does_not_clip(self):
        X = np.ones((6, 12, 128), dtype='float32')
        X[0] *= 2; X[1] *= 4; X[2] *= 1000
        table = pd.DataFrame({'hospital_id': ['a', 'b', 'c', 'd', 'e', 'f'], 'label': [0, 1, 2, 0, 1, 2]})
        rows = reference_rows(table, ['a', 'b'], 3)
        scale = global_scale(X, rows)
        self.assertEqual(scale, 3.)
        ds = BalancedPatients(X, table, ['a', 'b', 'c'], scale, 1, 42)
        samples = [ds[i][0] for i in range(30)]
        self.assertGreater(max(float(x.max()) for x in samples), 300.)
        self.assertTrue(set(ds.groups) == {0, 1, 2})

    def test_native_generator_and_limb_algebra(self):
        seed_everything(12)
        network = Generator(base=16)
        output = network(torch.randn(2, 64), torch.tensor([0, 2]))
        self.assertEqual(tuple(output.shape), (2, 12, 5000))
        torch.testing.assert_close(output[:, 2], output[:, 1] - output[:, 0])
        torch.testing.assert_close(output[:, 3], -(output[:, 0] + output[:, 1]) / 2)
        torch.testing.assert_close(output[:, 4], output[:, 0] - output[:, 1] / 2)
        torch.testing.assert_close(output[:, 5], output[:, 1] - output[:, 0] / 2)
        self.assertFalse(any(isinstance(m, (torch.nn.Tanh, torch.nn.BatchNorm1d)) for m in network.modules()))

    def test_epoch_resume_matches_uninterrupted_gan(self):
        cfg = {**PROTOCOL['training'], 'epochs': 2, 'batch_size': 3, 'draws_per_fit_patient_per_epoch': 1}
        X = np.random.default_rng(12).normal(size=(12, 12, 128)).astype('float32')
        table = pd.DataFrame({'hospital_id': [f'p{i}' for i in range(12)], 'label': [i % 3 for i in range(12)]})
        def setup():
            seed_everything(142)
            g, d = Generator(base=16, length=128), Discriminator()
            dataset = BalancedPatients(X, table, table.hospital_id.tolist(), 1., 1, 10142)
            loader = DataLoader(dataset, batch_size=3, generator=torch.Generator().manual_seed(11142))
            return g, d, loader
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            a, b = Path(temp) / 'full', Path(temp) / 'resume'; a.mkdir(); b.mkdir()
            g, d, loader = setup(); fit(g, d, loader, torch.device('cpu'), cfg, a, 'case', 142)
            expected_g, expected_d = cpu_weights(g), cpu_weights(d)
            g, d, loader = setup(); fit(g, d, loader, torch.device('cpu'), cfg, b, 'case', 142, stop_after=1)
            g, d, loader = setup(); fit(g, d, loader, torch.device('cpu'), cfg, b, 'case', 142)
            for name, value in g.state_dict().items(): torch.testing.assert_close(value, expected_g[name], rtol=0, atol=0)
            for name, value in d.state_dict().items(): torch.testing.assert_close(value, expected_d[name], rtol=0, atol=0)
            h1, h2 = (pd.read_csv(p / 'history.csv').drop(columns='seconds') for p in (a, b))
            pd.testing.assert_frame_equal(h1, h2)

    def test_nearest_is_full_native_and_copy_collapse_nonfinite_fail(self):
        values = np.random.default_rng(1).normal(size=(6, 12, 5000)).astype('float32')
        labels = np.repeat(np.arange(3), 2)
        table = pd.DataFrame({'hospital_id': [f'p{i}' for i in range(6)], 'label': labels})
        ids = table.hospital_id.tolist()
        nearest = native_nearest(values, values, table, ids, 1., labels)
        self.assertLess(nearest.relative_rmse.max(), 1e-6)
        report, _ = numerical_screen(values, labels, values, table, ids, 1., PROTOCOL['quality_gate'])
        self.assertFalse(report['passed']); self.assertIn('exact_training_window_copy', report['errors'])
        constant = np.zeros_like(values)
        report, _ = numerical_screen(constant, labels, values, table, ids, 1., PROTOCOL['quality_gate'])
        self.assertFalse(report['passed']); self.assertIn('class_0_duplicates', report['errors'])
        nonfinite = values.copy(); nonfinite[0, 0, 0] = np.nan
        report, neighbours = numerical_screen(nonfinite, labels, values, table, ids, 1., PROTOCOL['quality_gate'])
        self.assertFalse(report['passed']); self.assertIsNone(neighbours)

    def test_spectral_gate_rejects_noisy_synthesis(self):
        rng = np.random.default_rng(3)
        t = np.arange(5000) / 2000
        clean = np.stack([np.stack([(1 + rng.uniform(-.1, .1)) * np.sin(2 * np.pi * (6 + c) * t + rng.uniform(-.2, .2))
                            for _ in range(8)]) for c in np.repeat(np.arange(3), 4)])
        real = twelve_leads(torch.from_numpy(clean)).numpy()
        frame, mapping = features(real)
        frame['label'] = np.repeat(np.arange(3), 4); frame['weight'] = 1.
        _, failure = compare_features(frame, frame, frame, mapping, PROTOCOL['quality_gate'])
        self.assertFalse(failure)
        noisy, _ = features(rng.normal(size=real.shape))
        noisy['label'] = frame.label; noisy['weight'] = 1.
        _, failure = compare_features(frame, frame, noisy, mapping, PROTOCOL['quality_gate'])
        self.assertTrue(any(r['family'] == 'spectrum' for r in failure))

    def test_manifest_tampering_rejected_and_failed_qc_blocks_fold2_before_data_read(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            directory = Path(temp); (directory / 'x.txt').write_text('x')
            write_manifest(directory / 'manifest.json', {'passed': False, 'files': {'x.txt': sha256_file(directory / 'x.txt')}})
            saved = read_json(directory / 'manifest.json'); verify_files(directory, saved)
            saved['passed'] = True
            with self.assertRaises(RuntimeError): verify_files(directory, saved)
        study = Study(Path('/unused'), Path('/unused'), PROTOCOL, [], [], {})
        with patch('research_v6.experiment.require_smoke'), patch('research_v6.experiment.require_prepared'), \
                patch('research_v6.experiment.require_quality', side_effect=RuntimeError('FAIL')) as gate, \
                patch('research_v6.experiment.load_development') as load:
            with self.assertRaises(RuntimeError): experiment.train(study, 2, torch.device('cpu'))
            gate.assert_called_once_with(study, 1); load.assert_not_called()

    def test_endpoints_synthetic_native_data_preserve_outer_waveforms(self):
        p = deepcopy(PROTOCOL); p['training'].update(epochs=1, batch_size=3, draws_per_fit_patient_per_epoch=1)
        p['generator']['base_channels'] = 16; p['generation']['per_class'] = 2; p['minimum_free_gib'] = 0
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            root = Path(temp); source = root / 'source'; cache = source / 'data/development'; cache.mkdir(parents=True)
            ids = [f'p{i}' for i in range(12)]
            records = [{'hospital_id': hid, 'label': i % 3, 'class_name': f'class{i%3}'} for i, hid in enumerate(ids)]
            table = pd.DataFrame({'hospital_id': ids, 'label': [r['label'] for r in records]})
            table.to_csv(cache / 'segments.csv', index=False)
            X = np.random.default_rng(13).normal(size=(12, 12, 5000)).astype('float32')
            # Outer validation waveforms must never be indexed, even though present in the cache.
            X[9:] = np.nan; np.save(cache / 'X.npy', X)
            fold = {'fold': 1, 'gan_fit_ids': ids[:6], 'gan_reference_ids': ids[6:9], 'outer_validation_ids': ids[9:]}
            study = Study(root / 'run', source, p, records, [fold], {'test': 'synthetic'})
            experiment.prepare(study); require_prepared(study, 1)
            experiment.smoke(study, torch.device('cpu'))
            experiment.train(study, 1, torch.device('cpu'))
            original = sha256_file(study.folder(1) / 'generator.pth')
            experiment.train(study, 1, torch.device('cpu'))
            self.assertEqual(original, sha256_file(study.folder(1) / 'generator.pth'))
            experiment.generate(study, 1, torch.device('cpu')); experiment.qc(study, 1)
            quality = read_json(study.folder(1) / 'quality.json')
            self.assertFalse(quality['passed']); self.assertFalse(quality['outer_validation_signals_read'])
            with self.assertRaises(RuntimeError): require_quality(study, 1)
            from research_v6.analysis import report
            report(study, 1)
            self.assertTrue((study.folder(1) / 'figures/waveforms_class_0.png').exists())

    def test_manual_notebooks_are_unexecuted_valid_python(self):
        notebooks = sorted((ROOT / 'notebooks').glob('3[3-6]_*.ipynb'))
        self.assertEqual(len(notebooks), 4)
        for path in notebooks:
            notebook = nbformat.read(path, as_version=4); nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    ast.parse(cell.source); self.assertIsNone(cell.execution_count); self.assertFalse(cell.outputs)


if __name__ == '__main__': unittest.main()
