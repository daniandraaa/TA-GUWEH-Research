import unittest

import numpy as np

from analysis_v3.analyze_incart import cluster_bootstrap


class HolterBootstrapTests(unittest.TestCase):
    def test_one_source_keeps_exact_count_metrics(self):
        rows = {r['metric']: r for r in cluster_bootstrap([[8, 2, 2, 5, 3]], repeats=50)}
        for name, expected in [('micro_f1', .8), ('ventricular_detection_recall', .6),
                               ('non_V_annotation_detection_recall', 1.)]:
            self.assertAlmostEqual(rows[name]['point_estimate'], expected)
            self.assertAlmostEqual(rows[name]['ci95_lower'], expected)
            self.assertAlmostEqual(rows[name]['ci95_upper'], expected)

    def test_pooled_point_uses_counts_instead_of_mean_record_scores(self):
        groups = np.array([[9, 1, 1, 1, 1], [50, 50, 50, 50, 25]])
        row = next(r for r in cluster_bootstrap(groups, repeats=200, seed=17) if r['metric'] == 'micro_f1')
        self.assertAlmostEqual(row['point_estimate'], 2 * 59 / (2 * 59 + 51 + 51))
        self.assertNotAlmostEqual(row['point_estimate'], (.9 + .5) / 2)
        self.assertAlmostEqual(row['ci95_lower'], .5)
        self.assertAlmostEqual(row['ci95_upper'], .9)

    def test_no_V_annotations_produce_undefined_V_recall(self):
        row = next(r for r in cluster_bootstrap([[10, 0, 0, 0, 0]], repeats=30) if r['metric'] == 'ventricular_detection_recall')
        self.assertIsNone(row['point_estimate'])
        self.assertIsNone(row['ci95_lower'])
        self.assertIsNone(row['ci95_upper'])
        self.assertEqual(row['valid_draws'], 0)

    def test_impossible_V_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            cluster_bootstrap([[8, 2, 2, 5, 6]], repeats=30)


if __name__ == '__main__':
    unittest.main()
