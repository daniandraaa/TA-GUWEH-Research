"""Jalankan alur CV -> seleksi -> final -> test pada fixture kecil."""
import copy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.artifacts import load_configs, read_json, write_json
from src.experiment import evaluate_test, prepare_fold, require_selection, run_cv_case, train_final


class IntegrationTest(unittest.TestCase):
    def test_complete_grid_required_before_final_and_test(self):
        config, scenarios = load_configs()
        config = copy.deepcopy(config)
        config["models"] = ["encoder"]
        config["segment_length"] = 16
        config["training"].update(epochs=2, patience=1, scheduler_t_max=2, eval_batch_size=16)
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            write_json(run / "protocol.json", {"experiment_hash": "fixture"})
            patients = [{"hospital_id":str(i), "label":i%3, "pool":"development", "class_name":config["class_names"][i%3], "sublocation":"fixture"} for i in range(60)]
            patients += [{"hospital_id":str(100+i), "label":i%3, "pool":"test", "class_name":config["class_names"][i%3], "sublocation":"fixture"} for i in range(9)]
            write_json(run / "data/patients.json", patients)
            def make_pool(run, config, pool):
                directory = run / "data" / pool
                directory.mkdir(parents=True,exist_ok=True)
                chosen = [r for r in patients if r["pool"]==pool]
                if not (directory / "X.npy").exists():
                    X=rng.normal(size=(len(chosen),12,16)).astype(np.float32)
                    np.save(directory / "X.npy",X)
                    pd.DataFrame([{ "hospital_id":r["hospital_id"], "label":r["label"], "start_sample":0, "valid_samples":16} for r in chosen]).to_csv(directory / "segments.csv",index=False)
                    write_json(directory / "manifest.json", {"patients":chosen})
                return directory
            def fake_gan(dataset,directory,config,device,patient_ids,seed,resume):
                directory.mkdir(parents=True,exist_ok=True)
                np.save(directory / "synthetic_X.npy",np.zeros((1,12,16),dtype=np.float32))
                np.save(directory / "synthetic_y.npy",np.array([0]))
                write_json(directory / "manifest.json", {"fit_patient_ids":patient_ids})
            with patch("src.experiment.prepare_pool",side_effect=make_pool), patch("src.experiment.train_fold_gan",side_effect=fake_gan), patch("src.experiment.build_model",side_effect=lambda name,cfg: nn.Sequential(nn.Flatten(),nn.Linear(12*16,3))), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    require_selection(run,config,scenarios)
                for scenario in scenarios:
                    for fold in range(1,6):
                        run_cv_case(run,config,"encoder",scenario,fold,torch.device("cpu"),resume=True)
                self.assertFalse((run / "data/test/X.npy").exists())
                choice=require_selection(run,config,scenarios)
                self.assertFalse(choice["test_used_for_selection"])
                self.assertEqual(len(pd.read_csv(run / "encoder/cv_fold_results.csv")),45)
                self.assertEqual(len(pd.read_csv(run / "encoder/cv_summary.csv")),9)
                with self.assertRaises(RuntimeError):
                    evaluate_test(run,config,scenarios,"encoder",torch.device("cpu"))
                train_final(run,config,scenarios,"encoder",torch.device("cpu"),resume=True)
                self.assertFalse((run / "data/test/X.npy").exists())
                result=evaluate_test(run,config,scenarios,"encoder",torch.device("cpu"))
                self.assertEqual(result["n_patients"],9)
                self.assertFalse(result["test_used_for_selection"])
                for fold in range(1,6):
                    manifest=read_json(run / "folds/manifest.json")[fold-1]
                    scaler=read_json(run / "folds" / f"fold_{fold}" / "scaler_provenance.json")
                    gan=read_json(run / "augmentation" / f"fold_{fold}" / "manifest.json")
                    self.assertEqual(set(scaler["fit_patient_ids"]),set(manifest["train_ids"]))
                    self.assertEqual(set(gan["fit_patient_ids"]),set(manifest["train_ids"]))
                    self.assertFalse(set(gan["fit_patient_ids"]) & set(manifest["validation_ids"]))


if __name__ == "__main__":
    unittest.main()
