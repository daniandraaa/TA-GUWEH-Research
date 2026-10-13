"""Engineering audit invariants; no real ECG or training data fixtures."""
import unittest
import numpy as np
import pandas as pd
from analysis_v4.input_audit import matched_controls, reconstruct_windows, signal_profile, window_candidate_counts


class InputAuditTests(unittest.TestCase):
    def test_controls_never_cross_class_and_disclose_reuse(self):
        table = pd.DataFrame([
            {'hospital_id': 'h1', 'class_name': 'a', 'sublocation': 'x', 'n_wrong_selected_models': 5},
            {'hospital_id': 'h2', 'class_name': 'a', 'sublocation': 'x', 'n_wrong_selected_models': 4},
            {'hospital_id': 'h3', 'class_name': 'b', 'sublocation': 'z', 'n_wrong_selected_models': 4},
            {'hospital_id': 'c1', 'class_name': 'a', 'sublocation': 'x', 'n_wrong_selected_models': 0},
            {'hospital_id': 'c2', 'class_name': 'c', 'sublocation': 'z', 'n_wrong_selected_models': 0}])
        pairs = matched_controls(table, dict.fromkeys(table.hospital_id, 10))
        self.assertEqual(pairs.iloc[0].control_id, 'c1')
        self.assertTrue(pairs.iloc[1].reused_control)
        self.assertTrue(pd.isna(pairs.iloc[2].control_id))

    def test_overlap_is_counted_once_and_corruption_is_rejected(self):
        signal = np.arange(72, dtype=np.float32).reshape(6, 12)
        blocks = np.stack([signal[:4].T, signal[2:].T])
        rows = pd.DataFrame({'start_sample': [0, 2], 'valid_samples': [4, 4]})
        np.testing.assert_array_equal(reconstruct_windows(blocks, rows), signal)
        corrupt = blocks.copy(); corrupt[1, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'disagree'):
            reconstruct_windows(corrupt, rows)

    def test_gaps_and_nonfinite_signals_are_rejected(self):
        rows = pd.DataFrame({'start_sample': [0, 3], 'valid_samples': [2, 2]})
        with self.assertRaisesRegex(ValueError, 'gap'):
            reconstruct_windows(np.ones((2, 12, 2), dtype=np.float32), rows)
        invalid = np.ones((10, 12)); invalid[0, 0] = np.nan
        with self.assertRaises(ValueError): signal_profile(invalid, 'raw')

    def test_candidates_use_half_open_window_boundaries(self):
        np.testing.assert_array_equal(window_candidate_counts([0, 5, 10, 15], [0, 5, 10], 10), [2, 2, 2])
        np.testing.assert_array_equal(window_candidate_counts([], [0, 5], 10), [0, 0])


if __name__ == '__main__': unittest.main()
