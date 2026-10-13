"""Invarian sintetis V13: tanpa sinyal EKG asli, training, atau test."""
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
from research_v13.analysis import bootstrap_difference, holm, mcnemar_exact, wilcoxon
from research_v13.common import all_scenarios, baseline, grid, validate
from research_v13.data import GanBags

P = yaml.safe_load((Path(__file__).resolve().parents[1] / 'research_v13/protocol.yaml').read_text())


def toy_bags(windows_per_patient=(4, 6, 2)):
    rows, X = [], []
    for p, n in enumerate(windows_per_patient):
        for _ in range(n):
            rows.append({'hospital_id': f'p{p}', 'label': p % 3}); X.append(np.full((12, 50), float(p)))
    scaler = SimpleNamespace(mean_=np.zeros(12), scale_=np.ones(12))
    synthetic = np.stack([np.full((12, 50), 100. + c) for c in range(3)])
    data = GanBags(np.asarray(X, np.float32), pd.DataFrame(rows), [f'p{i}' for i in range(len(windows_per_patient))],
                   scaler, synthetic, np.arange(3), size=4, sample_seed=7)
    return data


class ProtocolTests(unittest.TestCase):
    def test_grid_has_nine_and_contains_baseline_configuration(self):
        validate(P)
        self.assertEqual(len(grid(P)), 9)
        b = baseline(P)
        self.assertEqual(sum((g['learning_rate'], g['batch_size']) == (b['learning_rate'], b['batch_size']) for g in grid(P)), 1)
        self.assertEqual(len(all_scenarios(P)), 10)
        self.assertEqual(b['augmentation'], 'none')

    def test_validation_never_augmented(self):
        bad = {**P, 'gan_augmentation': {**P['gan_augmentation'], 'validation_augmented': True}}
        with self.assertRaises(ValueError):
            validate(bad)


class GanBagTests(unittest.TestCase):
    def test_exactly_one_same_class_synthetic_window_in_full_bags(self):
        data = toy_bags()
        for item in range(len(data)):
            value, label, _ = data[item]
            synthetic = [(w == 100. + label).all().item() for w in value]
            other_synth = [any((w == 100. + c).all().item() for c in range(3) if c != label) for w in value]
            self.assertFalse(any(other_synth))
            self.assertEqual(sum(synthetic), 1 if len(value) == 4 else 0)

    def test_rng_resume_is_exact(self):
        a, b = toy_bags(), toy_bags()
        a[1]; state = a.generator.get_state()
        b.generator.set_state(state)
        for x, y in zip(a[1][0], b[1][0]):
            self.assertTrue(np.array_equal(x.numpy(), y.numpy()))


class StatisticsTests(unittest.TestCase):
    def test_holm_monotone_and_bounded(self):
        adjusted = holm([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[2], 0.06)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertTrue(all(0 <= v <= 1 for v in holm([0.9, 0.8])))

    def test_mcnemar_counts_discordant_pairs_only(self):
        r = mcnemar_exact([1, 1, 0, 0, 1], [1, 0, 1, 1, 1])
        self.assertEqual((r['baseline_correct_finetune_wrong'], r['baseline_wrong_finetune_correct']), (1, 2))
        self.assertEqual(mcnemar_exact([1, 0], [1, 0])['p_value'], 1.)

    def test_wilcoxon_identical_is_one(self):
        self.assertEqual(wilcoxon([.5, .6], [.5, .6])['p_value'], 1.)

    def test_bootstrap_detects_clear_improvement(self):
        truth = np.repeat(np.arange(3), 20)
        good = np.eye(3)[truth]
        bad = np.eye(3)[(truth + 1) % 3]
        r = bootstrap_difference(truth, bad, good, 200, 1)
        self.assertAlmostEqual(r['observed_difference'], 1.)
        self.assertGreater(r['ci95_low'], 0)


if __name__ == '__main__':
    unittest.main()
