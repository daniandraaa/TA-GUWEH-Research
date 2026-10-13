"""Synthetic fixtures only: no real training or hold-out inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import unittest
import tempfile
from unittest.mock import patch
from contextlib import ExitStack
from types import SimpleNamespace
import torch
import yaml
from scipy import signal
from research_v2.data import denoise, relabel, require_final_complete
from research_v2.experiment import class_weights, fit, summarize, train_finals
from research_v2.reporting import bootstrap_f1

ROOT = Path(__file__).resolve().parents[1]


def test_anatomical_membership_preserved():
    rows = [{'hospital_id': str(i), 'class_name': old, 'label': k, 'sublocation': sub}
            for i, (old, k, sub) in enumerate([('LVOT', 0, 'LCC'), ('Non-OT', 1, 'LC'), ('RVOT', 2, 'FreeWall')])]
    names = ['LVOT', 'RVOT_cusp', 'RVOT_septal_freewall_other']
    result = relabel(rows, names)
    assert [r['label'] for r in result] == [0, 1, 2]
    assert result[1]['class_name'] == 'RVOT_cusp'
    assert rows[1]['class_name'] == 'Non-OT'
    with unittest.TestCase().assertRaises(ValueError):
        relabel([{**rows[1], 'label': 2}], names)


def test_denoising_zero_finite_and_length():
    spec = yaml.safe_load((ROOT / 'research_v2/protocol.yaml').read_text())['preprocessing']
    x = np.zeros((5001, 12))
    assert np.array_equal(denoise(x, 2000, spec), x)
    rng = np.random.default_rng(1)
    y = denoise(rng.normal(size=x.shape), 2000, spec)
    assert y.shape == x.shape and np.isfinite(y).all()
    with unittest.TestCase().assertRaises(ValueError):
        denoise(np.full_like(x, np.nan), 2000, spec)


def test_filter_physical_cutoff():
    old = signal.butter(4, .5 / (1000 / 2), btype='highpass', output='sos')
    new = signal.butter(4, .5, btype='highpass', fs=2000, output='sos')
    _, old_response = signal.sosfreqz(old, worN=[1.0], fs=2000)
    _, new_response = signal.sosfreqz(new, worN=[.5], fs=2000)
    assert abs(abs(old_response[0]) - 1 / np.sqrt(2)) < 1e-6
    assert abs(abs(new_response[0]) - 1 / np.sqrt(2)) < 1e-6


def test_fold_weights_and_bootstrap():
    assert torch.allclose(class_weights([0, 0, 1, 2], 3), torch.tensor([.6, 1.2, 1.2]))
    with unittest.TestCase().assertRaises(ValueError):
        class_weights([0, 1], 3)
    truth = np.tile([0, 1, 2], 4)
    probs = np.eye(3)[truth]
    assert bootstrap_f1(truth, probs, 20, 1) == [1.0, 1.0]
    probs[0] = [0, 1, 0]
    assert bootstrap_f1(truth, probs, 20, 1) == bootstrap_f1(truth, probs, 20, 1)


def test_end_to_end_tiny_fixture(tmp_path, monkeypatch):
    import research_v2.experiment as experiment
    torch.set_num_threads(1)
    config = yaml.safe_load((ROOT / 'configs/finetune_fixed.yaml').read_text())
    cfg = yaml.safe_load((ROOT / 'research_v2/protocol.yaml').read_text())
    cfg.update(minimum_free_gib=0, final_seeds=[42, 43])
    config.update(models=['encoder'], class_names=cfg['class_names'])
    config['training'].update(epochs=1, patience=1, eval_batch_size=6)
    monkeypatch.setattr(experiment, 'build_model', lambda *args: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12 * 32, 3)))
    pool = tmp_path / 'data/development'; pool.mkdir(parents=True)
    (tmp_path / 'folds').mkdir()
    ids = [str(i) for i in range(30)]
    np.save(pool / 'X.npy', np.random.default_rng(4).normal(size=(30, 12, 32)).astype('float32'))
    pd.DataFrame({'hospital_id': ids, 'label': np.tile([0, 1, 2], 10)}).to_csv(pool / 'segments.csv', index=False)
    folds = [{'train_ids': [s for s in ids if s not in ids[f*6:(f+1)*6]], 'validation_ids': ids[f*6:(f+1)*6]} for f in range(5)]
    (tmp_path / 'folds/manifest.json').write_text(json.dumps(folds))
    (tmp_path / 'protocol.json').write_text(json.dumps({'experiment_hash': 'fixture'}))
    scenarios = [{'id': 'S01', 'learning_rate': .001, 'batch_size': 6}]
    with unittest.TestCase().assertRaises(RuntimeError):
        require_final_complete(tmp_path, cfg, config)
    with unittest.TestCase().assertRaises(RuntimeError):
        summarize(tmp_path, cfg, config, scenarios, select=True)
    for arm in cfg['arms']:
        for fold in range(1, 6):
            fit(tmp_path, cfg, config, 'encoder', arm, scenarios[0], torch.device('cpu'), fold=fold)
    summarize(tmp_path, cfg, config, scenarios, select=True)
    with unittest.TestCase().assertRaises(RuntimeError):
        require_final_complete(tmp_path, cfg, config)
    train_finals(tmp_path, cfg, config, scenarios, 'encoder', torch.device('cpu'))
    require_final_complete(tmp_path, cfg, config)
    assert len(list((tmp_path / 'final').rglob('model.pth'))) == 2
    assert not list((tmp_path / 'cv').rglob('best.pth'))
    assert not (tmp_path / 'data/test').exists()
    # Evaluate only after all final fits; six independent synthetic test patients.
    test_pool = tmp_path / 'data/test'; test_pool.mkdir()
    np.save(test_pool / 'X.npy', np.random.default_rng(8).normal(size=(6, 12, 32)).astype('float32'))
    pd.DataFrame({'hospital_id': [str(100+i) for i in range(6)], 'label': [0,1,2,0,1,2]}).to_csv(test_pool / 'segments.csv', index=False)
    (test_pool / 'manifest.json').write_text('{}')
    experiment.evaluate(tmp_path, tmp_path, cfg, config, torch.device('cpu'))
    from research_v2.reporting import final_report
    cfg['bootstrap_repeats'] = 20
    report = final_report(tmp_path, cfg, config)
    assert len(report) == 2 and report.primary_seed.sum() == 1
    assert (tmp_path / 'reports/historical_test_uncertainty.png').exists()
    # Completed cases are idempotent and identities cannot silently change.
    path = tmp_path / 'cv/real_only/encoder/S01/fold_1/metrics.json'
    before = path.read_bytes()
    fit(tmp_path, cfg, config, 'encoder', 'real_only', scenarios[0], torch.device('cpu'), fold=1)
    assert before == path.read_bytes()
    with unittest.TestCase().assertRaises(RuntimeError):
        fit(tmp_path, cfg, config, 'encoder', 'real_only', {**scenarios[0], 'learning_rate': .1}, torch.device('cpu'), fold=1)


class ResearchV2Tests(unittest.TestCase):
    def test_labels(self):
        test_anatomical_membership_preserved()

    def test_denoising(self):
        test_denoising_zero_finite_and_length()

    def test_cutoff(self):
        test_filter_physical_cutoff()

    def test_statistics(self):
        test_fold_weights_and_bootstrap()

    def test_prepare_development_without_test_access(self):
        import research_v2.data as data
        from src.data import LEADS
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'; (source / 'data').mkdir(parents=True)
            (source / 'folds').mkdir()
            raw_dir = root / 'PVCVTRawECGData'; raw_dir.mkdir()
            cfg = yaml.safe_load((ROOT / 'research_v2/protocol.yaml').read_text())
            cfg.update(minimum_free_gib=0, segment_length=500)
            config = {'class_names': cfg['class_names'], 'models': ['encoder']}
            records = []
            for i in range(9):
                records.append({'hospital_id': str(i), 'label': i % 3,
                                'sublocation': ['LCC', 'LC', 'FreeWall'][i % 3],
                                'class_name': ['LVOT', 'Non-OT', 'RVOT'][i % 3],
                                'pool': 'development' if i < 6 else 'test'})
                if i < 6:
                    t = np.arange(1024) / 2000
                    raw = np.tile(np.sin(2 * np.pi * 10 * t)[:, None], (1, 12))
                    pd.DataFrame(raw, columns=LEADS).to_csv(raw_dir / f'{i}.csv', index=False)
            (source / 'data/patients.json').write_text(json.dumps(records))
            folds = [{'train_ids': ['0','1','2'], 'validation_ids': ['3','4','5']},
                     {'train_ids': ['3','4','5'], 'validation_ids': ['0','1','2']}]
            (source / 'folds/manifest.json').write_text(json.dumps(folds))
            run = root / 'new'; run.mkdir()
            with patch.object(data, 'ROOT', root):
                data.prepare_pool(run, source, cfg, config)
                X = np.load(run / 'data/development/X.npy')
                self.assertEqual(X.shape, (18, 12, 500))
                self.assertTrue(np.isfinite(X).all())
                self.assertFalse((run / 'data/test').exists())
                with self.assertRaises(RuntimeError):
                    data.prepare_pool(run, source, cfg, config, 'test')
                self.assertEqual(data.prepare_pool(run, source, cfg, config), run / 'data/development')

    def test_workflow(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            monkeypatch = SimpleNamespace(setattr=lambda obj, name, value: stack.enter_context(patch.object(obj, name, value)))
            test_end_to_end_tiny_fixture(Path(temporary), monkeypatch)


if __name__ == '__main__':
    unittest.main()
