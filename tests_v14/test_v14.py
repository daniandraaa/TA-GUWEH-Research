"""Uji V14 tanpa training: metrik biner, augmentasi GAN biner, protokol."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from src import evaluation
import research_v4.training as v4_training
from research_v14 import experiment
from research_v14.common import validate

ROOT = Path(__file__).resolve().parents[1]


class FakeScaler:
    mean_ = np.zeros(12); scale_ = np.ones(12)


class V14Tests(unittest.TestCase):
    def test_binary_metrics_match_sklearn_and_three_class_untouched(self):
        y = np.array([0, 1, 1, 0, 1, 1]); p = np.array([[.8, .2], [.3, .7], [.6, .4], [.4, .6], [.1, .9], [.2, .8]])
        m = experiment.compute_metrics(y, p)
        self.assertAlmostEqual(m['auc'], roc_auc_score(y, p[:, 1]))
        self.assertAlmostEqual(m['f1'], f1_score(y, p.argmax(1), average='macro'))
        self.assertIs(v4_training.compute_metrics, experiment.compute_metrics)
        y3 = np.array([0, 1, 2, 0, 1, 2]); p3 = np.eye(3)[y3] * .8 + .2 / 3
        self.assertEqual(experiment.compute_metrics(y3, p3), evaluation.compute_metrics(y3, p3))

    def test_binary_gan_bags_draw_same_side_synthetic(self):
        X = np.random.default_rng(0).normal(size=(24, 12, 50)).astype(np.float32)
        table = pd.DataFrame({'hospital_id': np.repeat([f'p{i}' for i in range(6)], 4), 'label': np.repeat([0, 1, 1, 0, 1, 1], 4)})
        synthetic = np.stack([np.full((12, 50), c, np.float32) for c in (0, 1, 2, 0, 1, 2)])
        bags = experiment.BinaryGanBags(X, table, table.hospital_id.unique(), FakeScaler(), synthetic,
                                        np.array([0, 1, 2, 0, 1, 2]), {0: 0, 1: 1, 2: 1}, 4, sample_seed=1)
        for i in range(len(bags)):
            value, label, _ = bags[i]
            fake = [w for w in value if np.allclose(w, w[0, 0])]
            self.assertEqual(len(fake), 1)
            self.assertEqual(int(fake[0][0, 0] > 0), label)
        self.assertEqual(bags.replacements, 6)

    def test_protocol_is_the_preregistered_design(self):
        p = yaml.safe_load((ROOT / 'research_v14/protocol.yaml').read_text())
        validate(p)
        with self.assertRaises(ValueError):
            validate({**p, 'test_pool_read': True})


if __name__ == '__main__':
    unittest.main()
