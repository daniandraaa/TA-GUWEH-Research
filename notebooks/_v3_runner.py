"""Manual subprocess endpoints with streamed output, logs and interruption."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shlex
import signal
import subprocess


class V3Runner:
    def __init__(self, root, run_id, device='mps', label='manual_v3'):
        self.root, self.run_id, self.device, self.label = Path(root), run_id, device, label
        self.python = self.root / '.venv/bin/python'
        self.run_dir = self.root / 'results/research_v3' / run_id
        if not self.python.is_file():
            raise FileNotFoundError(self.python)
        print(f'Project: {self.root}\nPython: {self.python}\nRun: {run_id}\nDevice: {device}', flush=True)

    def run_stage(self, stage, *options):
        return self.run_module('research_v3', stage, '--run-id', self.run_id,
                               '--device', self.device, *options)

    def cv_progress(self):
        """Read completion for every expected case in the saved protocol.

        This is a notebook readiness check. The training endpoints separately
        enforce frozen experiment identity and validate selection/test access.
        """
        info = json.loads((self.run_dir / 'protocol.json').read_text())
        protocol, config = info['protocol'], info['config']
        models = protocol['models'] + ['extratrees']
        tree = protocol['classical']
        tree_ids = [f'T{i:02d}' for i in range(1, len(tree['max_features']) * len(tree['min_samples_leaf']) + 1)]
        expected = []
        for model in models:
            arms = protocol['classical_arms'] if model == 'extratrees' else protocol['arms']
            ids = tree_ids if model == 'extratrees' else [s['id'] for s in info['scenarios']]
            expected.extend(self.run_dir / 'cv' / arm / model / scenario / f'fold_{fold}/metrics.json'
                            for arm in arms for scenario in ids for fold in range(1, config['n_folds'] + 1))
        missing = [str(p.relative_to(self.run_dir)) for p in expected if not p.is_file()]
        return {'complete': len(expected) - len(missing), 'expected': len(expected),
                'ready': not missing, 'missing': missing}

    def run_module(self, module, *options):
        if module not in {'research_v3', 'research_v3.benchmark', 'analysis_v2.explain_v2', 'analysis_v2.preview_candidates',
                          'analysis_v3.analyze_final', 'analysis_v3.explain_final'}:
            raise ValueError('Unknown manual endpoint')
        command = [str(self.python), '-u', '-m', module, *map(str, options)]
        directory = self.run_dir / 'manual_logs'
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        path = directory / f'{stamp}_{self.label}_{module}.log'
        print('RUN MANUAL:', shlex.join(command), '\nLog:', path, flush=True)
        environment = {**os.environ, 'PYTHONUNBUFFERED': '1',
                       'MPLCONFIGDIR': str(self.root / 'tmp/matplotlib')}
        tail = deque(maxlen=30)
        with path.open('w', encoding='utf-8', buffering=1) as log:
            log.write(shlex.join(command) + '\n')
            with subprocess.Popen(command, cwd=self.root, env=environment,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
                try:
                    for line in process.stdout:
                        tail.append(line); print(line, end='', flush=True); log.write(line)
                    code = process.wait()
                except KeyboardInterrupt:
                    print('\nMenghentikan proses; ulang sel yang sama untuk resume.', flush=True)
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill(); process.wait()
                    log.write('\nInterrupted manually; completed epoch checkpoints retained.\n')
                    raise
        if code:
            raise RuntimeError(f'Endpoint gagal (exit {code}). Log: {path}\n' + ''.join(tail))
        print(f'Selesai. Log: {path.name}', flush=True)
        return path
