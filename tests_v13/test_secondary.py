"""Uji analisis sekunder V13 dengan run palsu (tanpa sinyal EKG/training)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pandas as pd
import yaml
from src.artifacts import write_json
import analysis_v13.secondary as sec
from research_v13.common import cv_directory, test_directory

P = yaml.safe_load((Path(__file__).resolve().parents[1] / 'research_v13/protocol.yaml').read_text())
CLASSES = ['LVOT', 'RVOT_cusp', 'RVOT_septal_freewall_other']


def predictions(seed, n=30):
    rng = np.random.default_rng(seed); truth = np.repeat([0, 1, 2], n // 3)
    prob = rng.dirichlet(np.ones(3), len(truth)); prob[np.arange(len(truth)), truth] += 1.5
    prob /= prob.sum(1, keepdims=True)
    frame = pd.DataFrame({'hospital_id': [f'p{i:03}' for i in range(len(truth))], 'true_label': truth})
    for k, c in enumerate(CLASSES): frame[f'probability_{c}'] = prob[:, k]
    return frame


class SecondaryTests(unittest.TestCase):
    def test_binary_metrics_and_majority(self):
        truth = np.array([0, 0, 1, 2]); prob = np.array([[.9, .05, .05], [.4, .3, .3], [.1, .8, .1], [.2, .2, .6]])
        m = sec.binary_metrics(truth, prob)
        self.assertEqual(m['sensitivity_LVOT'], .5); self.assertEqual(m['sensitivity_RVOT'], 1.)
        self.assertEqual(sec.majority_baseline(truth)['accuracy'], .5)

    def test_full_analysis_on_fake_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = SimpleNamespace(run=Path(tmp), protocol=P, class_names=CLASSES, signature='x')
            write_json(study.run / 'selection.json', {'choices': {m: {'baseline': {'scenario': 'baseline'},
                       'finetune': {'scenario': 'G05'}} for m in P['models']}})
            for i, m in enumerate(P['models']):
                for a, s in (('baseline', 'baseline'), ('finetune', 'G05')):
                    for f in P['folds']:
                        for seed in P['cv_seeds']:
                            d = cv_directory(study, m, s, f, seed); d.mkdir(parents=True)
                            predictions(seed + f + i).to_csv(d / 'validation_predictions.csv', index=False)
                    for seed in P['final_seeds']:
                        d = test_directory(study, m, a, seed); d.mkdir(parents=True)
                        predictions(seed * 7 + i).to_csv(d / 'patient_predictions.csv', index=False)
                        write_json(d / 'metrics.json', {})
            P2 = {**P, 'statistics': {**P['statistics'], 'bootstrap_resamples': 50}}
            study.protocol = P2
            with mock.patch.object(sec, 'context', return_value=study), mock.patch.object(sec, 'require_preregistered'):
                sec.analyze('fake')
            out = study.run / 'reports_secondary'
            for name in ('bab4_tabel_sekunder.csv', 'stat_sekunder_ensemble.csv', 'cv_binary_summary.csv',
                         'RINGKASAN_SEKUNDER_V13.md', 'roc_biner_ensemble.png'):
                self.assertTrue((out / name).is_file(), name)
            table = pd.read_csv(out / 'bab4_tabel_sekunder.csv')
            self.assertEqual(len(table), 6)
            self.assertTrue(table.binary_macro_f1_ensemble.between(0, 1).all())


if __name__ == '__main__':
    unittest.main()
