"""Manual V8 subprocess runner with live logs and actionable error tail."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import os
import shlex
import signal
import subprocess


class V8Runner:
    def __init__(self, root, run_id, device='mps', label='manual_v8'):
        self.root, self.run_id, self.device, self.label = Path(root), run_id, device, label
        self.python = self.root / '.venv/bin/python'
        self.run_dir = self.root / 'results/research_v8' / run_id
        if not self.python.is_file(): raise FileNotFoundError(self.python)
        print(f'Project: {self.root}\nPython: {self.python}\nRun: {run_id}\nDevice: {device}', flush=True)

    def run_stage(self, stage, *options):
        command = [str(self.python), '-u', '-m', 'research_v8', stage,
                   '--run-id', self.run_id, '--device', self.device, *map(str, options)]
        directory = self.run_dir / 'manual_logs'; directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        path = directory / f'{stamp}_{self.label}_{stage}.log'
        print('RUN MANUAL:', shlex.join(command), '\nLog:', path, flush=True)
        environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'MPLCONFIGDIR': str(self.root / 'tmp/matplotlib')}
        tail = deque(maxlen=40)
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
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try: process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill(); process.wait()
                    print('Terhenti. Ulang sel untuk melanjutkan epoch terakhir lengkap pada device yang sama.', flush=True)
                    raise
        if code: raise RuntimeError(f'Endpoint gagal (exit {code}). Log: {path}\n' + ''.join(tail))
        print(f'Selesai. Log: {path.name}', flush=True)
        return path
