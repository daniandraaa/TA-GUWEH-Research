"""Synthetic CPU checks only; never read patient waveforms or alter V8 files."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from src.artifacts import read_json, seed_everything, sha256_file, write_json
from src.evaluation import prediction_table, report
from research_v8.data import collate
from research_v8.training import fit
from research_v9.common import median_epoch, settings_for, validate_reuse


def check_median_rule_does_not_use_outer_scores():
    assert median_epoch([22, 3, 11]) == 11
    assert median_epoch([65, 18, 17]) == 18
    assert median_epoch([4, 11, 3]) == 4
    assert median_epoch([10, 10, 9]) == 10
    for values in ([1, 2], [0, 2, 3], [1.5, 2, 3], [101, 2, 3], [True, 2, 3]):
        with unittest.TestCase().assertRaises(ValueError): median_epoch(values)


def check_fixed_budget_preserves_the_original_scheduler():
    study = SimpleNamespace(training={'epochs': 100, 'scheduler_t_max': 100, 'patience': 15, 'weight_decay': 1e-4},
        choices={'encoder': {'learning_rate': 1e-4}})
    original = deepcopy(study.training)
    result = settings_for(study, {'model': 'encoder', 'epochs': 11})
    assert result == {**original, 'epochs': 11, 'learning_rate': 1e-4}
    assert study.training == original


def check_reuse_rejects_wrong_epoch_and_device(tmp_path):
    import research_v9.common as common
    folder = tmp_path / 'v8'; folder.mkdir()
    row = {'model': 'encoder', 'fold': 1, 'seed': 43, 'reuse': True, 'epochs': 10, 'v8_selected_epoch': 10}
    case = {'model': 'encoder', 'fold': 1, 'seed': 43, 'arm': 'real_only', 'device': 'mps', 'effective_seed': 143,
        'settings': {'epochs': 10}, 'pooling': 'attention', 'train_ids': ['a'], 'evaluation_ids': ['b'],
        'scaler_sha256': 's', 'source_gan_generated_sha256': 'g', 'test_signals_read': False, 'outer_epoch_selection': False}
    write_json(folder / 'case.json', case)
    study = SimpleNamespace(parent=object())
    with patch.object(common, 'v8_directory', return_value=folder), patch.object(common, 'v8_complete',
            return_value={'selected_epoch': 10, 'epochs_run': 10, 'inner_selection': False}):
        assert validate_reuse(study, row, case) == folder
        with unittest.TestCase().assertRaises(RuntimeError): validate_reuse(study, {**row, 'epochs': 11}, case)
        with unittest.TestCase().assertRaises(RuntimeError): validate_reuse(study, row, {**case, 'device': 'cpu'})
        with unittest.TestCase().assertRaises(RuntimeError): validate_reuse(study, row, {**case, 'train_ids': ['a', 'b']})
        with unittest.TestCase().assertRaises(RuntimeError): validate_reuse(study, row, {**case, 'settings': {'epochs': 10, 'scheduler_t_max': 10}})


class SyntheticBags(Dataset):
    def __init__(self):
        self.generator = torch.Generator().manual_seed(123)
        self.slot_rng = torch.Generator().manual_seed(456)
        self.source_rng = torch.Generator().manual_seed(789)
        self.values = torch.randn(9, 6, 3, 16, generator=torch.Generator().manual_seed(300))
    def __len__(self): return 9
    def __getitem__(self, index):
        k = 2 if index == 0 else 4
        chosen = torch.randperm(6, generator=self.generator)[:k]
        if k == 4: torch.randint(4, (), generator=self.slot_rng)
        meta = {'eligible': k == 4, 'replacements': 0, 'donor_hospital_id': None,
                'synthetic_index': None, 'label': index % 3}
        return self.values[index, chosen], index % 3, f'synthetic_{index}', meta
    def rng_states(self):
        return {'crop': self.generator.get_state(), 'slot': self.slot_rng.get_state(), 'source': self.source_rng.get_state()}
    def restore_rng_states(self, state):
        self.generator.set_state(state['crop']); self.slot_rng.set_state(state['slot']); self.source_rng.set_state(state['source'])


class ToyPatientModel(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Flatten(), nn.Linear(48, 12), nn.Dropout(.2), nn.Linear(12, 3))
    def forward(self, values, mask):
        b, k, _, _ = values.shape
        logp = F.log_softmax(self.net(values.reshape(b * k, 3, 16)), -1).reshape(b, k, 3)
        weights = mask.float() / mask.sum(1, keepdim=True)
        return (logp.exp() * weights[..., None]).sum(1).clamp_min(1e-8).log()


def toy_setup():
    seed_everything(142)
    model = ToyPatientModel()
    loader = DataLoader(SyntheticBags(), batch_size=4, shuffle=True, num_workers=0,
        collate_fn=collate, generator=torch.Generator().manual_seed(142))
    seed_everything(20142)
    return model, loader


def check_exact_cpu_resume_and_last_checkpoint_without_validation(tmp_path):
    settings = {'epochs': 3, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
        'scheduler_t_max': 100, 'minimum_lr': 1e-6, 'patience': 15, 'gradient_clip': 1.}
    device = torch.device('cpu'); weights = torch.ones(3)
    full = tmp_path / 'full'; resumed = tmp_path / 'resumed'
    model, loader = toy_setup()
    expected = fit(model, loader, None, device, settings, full, 'synthetic_full', weights, 16)
    model, loader = toy_setup()
    partial = fit(model, loader, None, device, settings, resumed, 'synthetic_resume', weights, 16, stop_after=1)
    assert partial == {'partial': True, 'epochs_run': 1}
    # Simulate stopping before the CSV commit; last.pt must repair it on resume.
    (resumed / 'history.csv').write_text('epoch\n')
    model, loader = toy_setup()
    actual = fit(model, loader, None, device, settings, resumed, 'synthetic_resume', weights, 16)
    assert actual['epochs_run'] == expected['epochs_run'] == actual['selected_epoch'] == 3
    assert actual['optimizer_steps'] == 9 and not actual['inner_selection']
    state1 = torch.load(full / 'model.pth', map_location='cpu', weights_only=True)
    state2 = torch.load(resumed / 'model.pth', map_location='cpu', weights_only=True)
    assert all(torch.equal(state1[k], state2[k]) for k in state1)
    history = pd.read_csv(resumed / 'history.csv')
    assert history.epoch.tolist() == [1, 2, 3]
    assert not any(c.startswith(('inner_', 'validation_')) for c in history.columns)
    assert history.replacements.eq(0).all()
    assert history.anchor_patients.eq(9).all()
    assert history.optimizer_steps.eq(3).all()
    before = read_json(resumed / 'fit_finished.json')
    assert fit(model, loader, None, device, settings, resumed, 'synthetic_resume', weights, 16) == actual
    assert read_json(resumed / 'fit_finished.json') == before


def check_prediction_audit_rejects_patient_budget_and_trace_corruption(tmp_path):
    import research_v9.analysis as analysis
    settings = {'epochs': 2, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
        'scheduler_t_max': 100, 'minimum_lr': 1e-6, 'patience': 15, 'gradient_clip': 1.}
    folder = tmp_path / 'synthetic_new'; old = tmp_path / 'synthetic_v8'; old.mkdir()
    model, loader = toy_setup()
    fitted = fit(model, loader, None, torch.device('cpu'), settings, folder, 'synthetic_audit', torch.ones(3), 16)
    (old / 'history.csv').write_bytes((folder / 'history.csv').read_bytes())
    train_ids = [f'synthetic_{i}' for i in range(9)]
    eval_ids = [f'synthetic_eval_{i}' for i in range(6)]
    classes = ['LVOT', 'RVOT_cusp', 'RVOT_other']
    truth = np.array([i % 3 for i in range(6)])
    probability = np.eye(3)[truth] * .7 + .1
    frame = prediction_table(eval_ids, truth, probability, classes)
    frame.to_csv(folder / 'predictions.csv', index=False)
    table = pd.DataFrame({'hospital_id': train_ids + eval_ids, 'label': [i % 3 for i in range(15)]})
    pd.DataFrame({'hospital_id': eval_ids, 'segment_index': range(9, 15), 'weight': 1.,
        'n_patient_segments': 1}).to_csv(folder / 'segment_weights.csv', index=False)
    write_json(folder / 'case.json', {'train_ids': train_ids, 'case_hash': 'synthetic_audit'})
    result = {**report(truth, probability, classes), **fitted,
        **{f'validation_recall_class_{c}': 1. for c in range(3)}}
    records = [{'hospital_id': h, 'label': i % 3} for i, h in enumerate(train_ids + eval_ids)]
    study = SimpleNamespace(parent=object(), records=records, inherited={'class_names': classes},
        protocol={'patient_batch_size': 4}, folds=[{'outer_validation_ids': eval_ids}], choices={'encoder': {'pooling': 'mean'}})
    row = {'model': 'encoder', 'fold': 1, 'seed': 42, 'epochs': 2, 'reuse': False}
    with patch.object(analysis, 'checked_complete', return_value=(folder, result)), patch.object(analysis, 'v8_directory', return_value=old):
        _, score = analysis.checked_new_predictions(study, row, table)
        assert score['f1'] == 1. and score['optimizer_steps'] == 6
        # Patient coverage and labels are checked separately from scalar metrics.
        wrong = frame.copy(); wrong.loc[0, 'true_label'] = 1
        wrong.to_csv(folder / 'predictions.csv', index=False)
        with unittest.TestCase().assertRaises(RuntimeError): analysis.checked_new_predictions(study, row, table)
        frame.to_csv(folder / 'predictions.csv', index=False)
        history = pd.read_csv(folder / 'history.csv')
        wrong = history.copy(); wrong.loc[0, 'replacements'] = 1
        wrong.to_csv(folder / 'history.csv', index=False)
        with unittest.TestCase().assertRaises(RuntimeError): analysis.checked_new_predictions(study, row, table)
        wrong = history.copy(); wrong.loc[0, 'learning_rate'] *= 2
        wrong.to_csv(folder / 'history.csv', index=False)
        with unittest.TestCase().assertRaises(RuntimeError): analysis.checked_new_predictions(study, row, table)
        history.to_csv(folder / 'history.csv', index=False)
        assert analysis.checked_new_predictions(study, row, table)[1]['f1'] == 1.


class TestV9(unittest.TestCase):
    def test_inner_median(self): check_median_rule_does_not_use_outer_scores()
    def test_original_scheduler(self): check_fixed_budget_preserves_the_original_scheduler()
    def test_reuse_rejects_wrong_recipe(self):
        with tempfile.TemporaryDirectory(prefix='v9_synthetic_reuse_') as name:
            check_reuse_rejects_wrong_epoch_and_device(Path(name))
    def test_exact_resume_without_validation(self):
        with tempfile.TemporaryDirectory(prefix='v9_synthetic_resume_') as name:
            check_exact_cpu_resume_and_last_checkpoint_without_validation(Path(name))
    def test_new_prediction_audit(self):
        with tempfile.TemporaryDirectory(prefix='v9_synthetic_prediction_') as name:
            check_prediction_audit_rejects_patient_budget_and_trace_corruption(Path(name))


if __name__ == '__main__': unittest.main()
