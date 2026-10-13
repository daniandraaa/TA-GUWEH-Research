"""Verify vectorized paired patient bootstrap against an independent reference."""
import unittest
import numpy as np
from sklearn.metrics import f1_score
from analysis_v3.analyze_final import bootstrap_scores


class PatientBootstrapTests(unittest.TestCase):
    def test_perfect_and_fully_incorrect_predictions(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        scores = bootstrap_scores(truth, {'perfect': truth, 'incorrect': (truth + 1) % 3}, repeats=200, seed=4)
        np.testing.assert_array_equal(scores['perfect'], np.ones(200))
        np.testing.assert_array_equal(scores['incorrect'], np.zeros(200))

    def test_identical_models_have_zero_paired_difference(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 1, 1, 0, 2, 1])
        scores = bootstrap_scores(truth, {'a': pred, 'b': pred}, repeats=200, seed=4)
        np.testing.assert_array_equal(scores['a'] - scores['b'], np.zeros(200))

    def test_nontrivial_case_matches_sklearn_per_draw(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 1, 1, 0, 2, 1])
        scores = bootstrap_scores(truth, {'a': pred}, repeats=200, seed=4)['a']
        rng = np.random.default_rng(4)
        draws = np.concatenate([rng.choice(np.flatnonzero(truth == k), size=(200, 2), replace=True)
                                for k in range(3)], axis=1)
        reference = [f1_score(truth[i], pred[i], labels=[0, 1, 2], average='macro', zero_division=0)
                     for i in draws]
        np.testing.assert_allclose(scores, reference, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
