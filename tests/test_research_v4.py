"""Small synthetic fixtures only; no real ECG training or historical test access."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from src.artifacts import read_json, seed_everything, write_json
from research_v4.models import PatientModel, LegacyFeatures, build_model
from research_v4.data import PatientBags, collate_bags, load_pool, scaler_for
from research_v4.common import progress, require_final_complete
from research_v4.experiment import summarize
from research_v4.training import fit, predict, train_epoch

ROOT = Path(__file__).resolve().parents[1]
LEGACY = {'n_leads': 12, 'n_classes': 3, 'cnn_channels': [8, 16], 'cnn_kernels': [7, 3],
          'transformer_heads': 4, 'transformer_layers': 1, 'transformer_dropout': 0.,
          'fc_dropout': 0., 'patch_size': 4}
NEW = {'dimension': 32, 'heads': 4, 'layers': 1, 'dropout': 0., 'temporal_bins': 4}


class TinyBackbone(nn.Module):
    dimension = 8
    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(12, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        z = torch.tanh(self.feature(x.mean(-1)))
        return z, self.head(z)


class ResearchV4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = yaml.safe_load((ROOT / 'research_v4/protocol.yaml').read_text())

    def fixture(self):
        generator = np.random.default_rng(9)
        X = generator.normal(size=(15, 12, 64)).astype('float32')
        # Unequal bag lengths, including patients with fewer than K segments.
        ids = ['p0'] + ['p1'] * 2 + ['p2'] * 3 + ['p3'] * 4 + ['p4'] * 2 + ['p5'] * 3
        labels = [int(i[1:]) % 3 for i in ids]
        table = pd.DataFrame({'hospital_id': ids, 'label': labels, 'valid_samples': 64})
        scaler = SimpleNamespace(mean_=np.zeros(12), scale_=np.ones(12))
        return X, table, scaler

    def test_legacy_logits_are_preserved(self):
        x = torch.randn(3, 12, 129)
        for name in ('encoder', 'patchtst', 'hierarchical'):
            model = LegacyFeatures(name, LEGACY).eval()
            with torch.no_grad():
                _, logits = model(x)
                self.assertTrue(torch.allclose(logits, model.base(x), atol=1e-6))

    def test_bag_mask_permutation_chunking_and_gradients(self):
        for pool in ('mean', 'attention'):
            model = PatientModel(TinyBackbone(), pool).eval()
            x = torch.randn(2, 5, 12, 64)
            mask = torch.tensor([[True, True, True, False, False], [True] * 5])
            original = model(x, mask)
            perm = torch.tensor([4, 2, 0, 3, 1])
            self.assertTrue(torch.allclose(original, model(x[:, perm], mask[:, perm]), atol=1e-6))
            changed = x.clone(); changed[~mask] = 1e8
            self.assertTrue(torch.allclose(original, model(changed, mask), atol=1e-6))
            chunked, w = model.all_segments(x[1], 2)
            full, _ = model.all_segments(x[1], 20)
            self.assertTrue(torch.allclose(chunked, full, atol=1e-6))
            self.assertTrue(torch.allclose(original[1].exp(), full, atol=1e-6))
            self.assertAlmostEqual(float(w.sum()), 1., places=6)
            loss = nn.NLLLoss()(original, torch.tensor([0, 1])); loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            with self.assertRaises(ValueError):
                model(x, torch.zeros_like(mask))

    def test_new_hybrids_train_with_attention(self):
        for name in ('residual_multiscale', 'lead_time'):
            model = build_model(name, 'attention', LEGACY, NEW)
            x = torch.randn(2, 2, 12, 193)
            logp = model(x)
            self.assertTrue(torch.allclose(logp.exp().sum(1), torch.ones(2), atol=1e-5))
            nn.NLLLoss()(logp, torch.tensor([0, 2])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_train_bags_and_evaluation_use_all_segments(self):
        X, table, scaler = self.fixture()
        training = PatientBags(X, table, sorted(table.hospital_id.unique()), scaler, 2)
        evaluation = PatientBags(X, table, sorted(table.hospital_id.unique()), scaler)
        self.assertEqual(sum(len(evaluation[i][0]) for i in range(len(evaluation))), len(X))
        self.assertTrue(all(len(training[i][0]) <= 2 for i in range(len(training))))
        batch, labels, mask, ids = collate_bags([training[0], training[1]])
        self.assertEqual(mask.sum().item(), 3)
        model = PatientModel(TinyBackbone(), 'attention')
        output_ids, truth, probs, weights = predict(model, evaluation, torch.device('cpu'), 2)
        self.assertEqual(len(output_ids), 6); self.assertEqual(len(weights), len(X))
        self.assertTrue(np.allclose(probs.sum(1), 1))
        for hid in output_ids:
            self.assertAlmostEqual(sum(w['weight'] for w in weights if w['hospital_id'] == hid), 1, places=6)

    def test_matched_crop_sampling_is_independent_of_model_rng(self):
        X, table, scaler = self.fixture()
        left = PatientBags(X, table, table.hospital_id.unique(), scaler, 2, sample_seed=44)
        right = PatientBags(X, table, table.hospital_id.unique(), scaler, 2, sample_seed=44)
        for _ in range(4):
            for i in range(len(left)):
                expected = left[i][0]
                torch.randn(1000)  # architecture/dropout consumption must not change crops
                self.assertTrue(torch.equal(expected, right[i][0]))

    def test_training_only_scaler_and_tamper_guard(self):
        X, table, _ = self.fixture()
        train = np.flatnonzero(table.hospital_id.isin(['p0', 'p1']))
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            run = Path(tmp)
            write_json(run / 'protocol.json', {'development_cache': {'fixture': 'native'}})
            original = scaler_for(run, 'fold_1', X, table, train, ['p0', 'p1'])
            shifted = X.copy(); shifted[~table.hospital_id.isin(['p0', 'p1'])] += 1000
            other = scaler_for(run, 'fold_2', shifted, table, train, ['p0', 'p1'])
            self.assertTrue(np.allclose(original.mean_, other.mean_))
            self.assertTrue(np.allclose(original.scale_, other.scale_))
            with self.assertRaises(ValueError):
                scaler_for(run, 'bad', X, table, train, ['p2'])
            marker = run / 'scalers/fold_1/provenance.json'
            value = read_json(marker); value['sha256'] = 'changed'; write_json(marker, value)
            with self.assertRaises(RuntimeError):
                scaler_for(run, 'fold_1', X, table, train, ['p0', 'p1'])

    def test_incomplete_grid_and_test_are_blocked_before_read(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            run = Path(tmp)
            self.assertEqual(progress(run, self.cfg)['expected'], 450)
            self.assertEqual(progress(run, self.cfg)['complete'], 0)
            with self.assertRaises(RuntimeError):
                summarize(run, self.cfg, select=True)
            with patch('research_v4.data.original_load_pool') as raw:
                with self.assertRaises(RuntimeError):
                    load_pool(run, run, self.cfg, 'test')
                raw.assert_not_called()
            with self.assertRaises(RuntimeError):
                require_final_complete(run, self.cfg)

    def test_epoch_resume_restores_bag_sampling_and_optimizer(self):
        X, table, scaler = self.fixture()
        settings = {'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': .0001,
                    'scheduler_t_max': 3, 'minimum_lr': 1e-6, 'gradient_clip': 1., 'epochs': 3, 'patience': 2}
        def setup():
            seed_everything(72)
            model = PatientModel(TinyBackbone(), 'attention')
            data = PatientBags(X, table, table.hospital_id.unique(), scaler, 2)
            loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_bags,
                                generator=torch.Generator().manual_seed(72), num_workers=0)
            return model, loader
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            root = Path(tmp); device = torch.device('cpu'); classes = ['a', 'b', 'c']; weights = torch.ones(3)
            model, loader = setup()
            fit(model, loader, None, device, settings, root / 'full', classes, weights, 2)
            reference = {k: v.clone() for k, v in model.state_dict().items()}
            model, loader = setup()
            calls = 0
            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt
                return train_epoch(*args, **kwargs)
            with patch('research_v4.training.train_epoch', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    fit(model, loader, None, device, settings, root / 'resume', classes, weights, 2)
            self.assertEqual(torch.load(root / 'resume/last.pt', weights_only=False)['epoch'], 1)
            model, loader = setup()
            fit(model, loader, None, device, settings, root / 'resume', classes, weights, 2)
            self.assertTrue(all(torch.equal(reference[k], v) for k, v in model.state_dict().items()))
            self.assertTrue(torch.load(root / 'resume/last.pt', weights_only=False)['completed'])

    def test_validation_checkpoint_has_class_logs_and_no_large_cv_checkpoint(self):
        X, table, scaler = self.fixture()
        seed_everything(11)
        model = PatientModel(TinyBackbone(), 'mean')
        data = PatientBags(X, table, table.hospital_id.unique(), scaler, 2)
        val = PatientBags(X, table, table.hospital_id.unique(), scaler)
        loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_bags,
                            generator=torch.Generator().manual_seed(11))
        settings = {'optimizer': 'Adam', 'learning_rate': .001, 'weight_decay': .0001,
            'scheduler_t_max': 2, 'minimum_lr': 1e-6, 'gradient_clip': 1., 'epochs': 2, 'patience': 2}
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            path = Path(tmp)
            result = fit(model, loader, val, torch.device('cpu'), settings, path, ['a', 'b', 'c'], torch.ones(3), 2)
            self.assertIn('validation_missing_predicted_classes', result)
            self.assertTrue((path / 'validation_predictions.csv').is_file())
            history = pd.read_csv(path / 'history.csv')
            self.assertIn('validation_recall_class_2', history)
            self.assertIn('validation_loss_patient_weighted', history)
            self.assertEqual(history.unique_training_patients_seen.tolist(), [6, 6])
            self.assertFalse((path / 'best.pth').exists())
            self.assertLess((path / 'last.pt').stat().st_size, 10000)


if __name__ == '__main__':
    unittest.main()
