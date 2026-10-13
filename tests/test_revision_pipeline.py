"""Verifikasi pemisahan pasien, scaler, metrik, dan checkpoint resume."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.artifacts import canonical_hash, read_json, seed_everything
from src.evaluation import aggregate_patients, compute_metrics
from src.experiment import bind_case
from src.preprocessing import fit_training_scaler
from src.splits import make_patient_folds
from src.training import fit_classifier


class RevisionTests(unittest.TestCase):
    def test_patient_folds_disjoint_and_exhaustive(self):
        patients = [{"hospital_id": str(i), "label": i % 3} for i in range(60)]
        folds = make_patient_folds(patients)
        self.assertEqual(sorted(x for f in folds for x in f["validation_ids"]), sorted(p["hospital_id"] for p in patients))
        for fold in folds:
            self.assertFalse(set(fold["train_ids"]) & set(fold["validation_ids"]))

    def test_scaler_ignores_validation_distribution(self):
        X = np.array([[[0, 2]], [[2, 4]], [[1000, 2000]]], dtype=np.float32)
        scaler = fit_training_scaler(X, np.array([0, 1]))
        self.assertAlmostEqual(scaler.mean_[0], 2)
        self.assertGreater(scaler.transform([[1000]])[0, 0], 100)

    def test_patient_aggregation_equal_patient_weight(self):
        labels = [0, 0, 1, 2]
        probabilities = np.array([[.8,.1,.1],[.6,.2,.2],[.1,.8,.1],[.1,.1,.8]])
        ids, truth, probs = aggregate_patients(labels, probabilities, ["A","A","B","C"])
        self.assertEqual(ids, ["A","B","C"])
        self.assertAlmostEqual(probs[0,0], .7)
        self.assertEqual(compute_metrics(truth, probs)["f1"], 1)
        with self.assertRaises(ValueError):
            aggregate_patients([0,1], probabilities[:2], ["A","A"])

    def test_case_identity_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bind_case(directory, {"learning_rate": .001})
            with self.assertRaises(RuntimeError):
                bind_case(directory, {"learning_rate": .0001})

    def test_best_checkpoint_and_resume_after_interruption(self):
        seed_everything(42)
        X = torch.eye(3).repeat(4,1)
        y = torch.arange(3).repeat(4)
        groups = np.array([str(i) for i in range(12)])
        settings = {"optimizer":"Adam", "learning_rate":.01, "weight_decay":0., "scheduler_t_max":3, "minimum_lr":0., "gradient_clip":1., "label_smoothing":0., "epochs":3, "patience":10}
        dataset = TensorDataset(X,y)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model = nn.Linear(3,3)
            loader = DataLoader(dataset, batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(42))
            calls = 0
            def interrupt_forward(module, args):
                nonlocal calls
                calls += 1
                if calls == 9:
                    raise RuntimeError("simulated interruption after checkpoint")
            hook = model.register_forward_pre_hook(interrupt_forward)
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                fit_classifier(model, loader, DataLoader(dataset,batch_size=3), groups, torch.device("cpu"), settings, directory, ["a","b","c"])
            hook.remove()
            resumed = nn.Linear(3,3)
            result = fit_classifier(resumed, loader, DataLoader(dataset,batch_size=3), groups, torch.device("cpu"), settings, directory, ["a","b","c"], resume=True)
            self.assertEqual(result["epochs_ran"], 3)
            self.assertTrue((directory / "best.pth").exists())
            self.assertEqual(read_json(directory / "metrics.json")["f1"], result["f1"])
            self.assertTrue(torch.load(directory / "last.pt", weights_only=True)["completed"])


if __name__ == "__main__":
    unittest.main()
