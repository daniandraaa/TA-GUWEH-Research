"""Check manual workflow routing and readiness without starting training."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('manual_v3_runner_test', ROOT / 'notebooks/_v3_runner.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Recorder:
    def __init__(self, complete=150):
        self.complete, self.calls = complete, []

    def run_stage(self, stage, *options):
        self.calls.append((stage, options))

    def cv_progress(self):
        return {'complete': self.complete, 'expected': 630, 'ready': self.complete == 630,
                'missing': ['pending'] * (630 - self.complete)}


class NotebookWorkflowTests(unittest.TestCase):
    def notebook(self, prefix):
        path, = (ROOT / 'notebooks').glob(prefix + '_*.ipynb')
        return json.loads(path.read_text())

    def final_cells(self, count):
        book = self.notebook('15')
        recorder = Recorder(count)
        scope = {'runner': recorder}
        for i in [2, 4, 5, 6, 7]:
            exec(compile(''.join(book['cells'][i]['source']), f'notebook15_cell{i}', 'exec'), scope)
        return recorder.calls

    def test_full_grid_defaults_route_to_all_scenarios(self):
        for number, name in [('12', 'encoder'), ('13', 'patchtst'), ('14', 'hierarchical')]:
            with self.subTest(model=name):
                book, runner = self.notebook(number), Recorder()
                scope = {'runner': runner}
                for i in [2, 5]:
                    exec(compile(''.join(book['cells'][i]['source']), f'notebook{number}_cell{i}', 'exec'), scope)
                self.assertIs(scope['RUN_FULL_GRID'], True)
                self.assertEqual(runner.calls, [('cv', ('--model', name))])

    def test_partial_grid_does_not_route_to_selection_final_or_test(self):
        self.assertEqual(self.final_cells(150), [('status', ())])

    def test_full_grid_routes_to_selection_then_all_finals_then_test(self):
        self.assertEqual(self.final_cells(630), [('status', ()), ('select', ())] +
                         [('final', ('--model', m)) for m in ['encoder', 'patchtst', 'hierarchical', 'extratrees']] +
                         [('test', ())])

    def test_unexpected_metrics_cannot_replace_a_missing_expected_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / '.venv/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            directory = root / 'results/research_v3/fixture'
            directory.mkdir(parents=True)
            protocol = {'protocol': {'models': ['encoder'], 'arms': ['window_segment'],
                        'classical_arms': ['window_patient'],
                        'classical': {'max_features': [.3, 1.], 'min_samples_leaf': [1]}},
                        'config': {'n_folds': 2}, 'scenarios': [{'id': 'S01'}]}
            (directory / 'protocol.json').write_text(json.dumps(protocol))
            runner = module.V3Runner(root, 'fixture', device='cpu')
            self.assertEqual(runner.cv_progress()['expected'], 6)
            for path in runner.cv_progress()['missing']:
                target = directory / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('{}')
            self.assertTrue(runner.cv_progress()['ready'])
            missing = directory / 'cv/window_segment/encoder/S01/fold_2/metrics.json'
            missing.unlink()
            unexpected = directory / 'cv/unknown/encoder/S99/fold_2/metrics.json'
            unexpected.parent.mkdir(parents=True)
            unexpected.write_text('{}')
            self.assertEqual(len(list((directory / 'cv').rglob('metrics.json'))), 6)
            progress = runner.cv_progress()
            self.assertFalse(progress['ready'])
            self.assertEqual(progress['complete'], 5)
            self.assertEqual(progress['missing'], ['cv/window_segment/encoder/S01/fold_2/metrics.json'])


if __name__ == '__main__':
    unittest.main()
