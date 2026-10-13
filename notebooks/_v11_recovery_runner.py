"""Manual recovery runner; original V11 runner/source are preserved."""
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
import os
import shlex
import signal
import subprocess


class V11Runner:
    def __init__(self,root,run_id):
        if not isinstance(run_id,str) or not run_id or Path(run_id).name!=run_id or run_id in {'.','..'} or '\\' in run_id:
            raise ValueError('Single run-id required')
        self.root=Path(root);self.run_id=run_id;self.python=self.root/'.venv/bin/python'
        if not self.python.is_file():raise FileNotFoundError(self.python)
        print(f'Project: {self.root}\nRun: {run_id}\nDevice: CPU (fitur dan convex L2 head)',flush=True)

    def run_stage(self,stage,*options):
        command=[str(self.python),'-u','-m','research_v11_recovery',stage,'--run-id',self.run_id,*map(str,options)]
        folder=self.root/'results/research_v11'/self.run_id/'manual_logs';folder.mkdir(parents=True,exist_ok=True)
        log_path=folder/f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")}_{stage}.log'
        print('RUN MANUAL:',shlex.join(command),'\nLog:',log_path,flush=True)
        environment={**os.environ,'PYTHONUNBUFFERED':'1','MPLCONFIGDIR':str(self.root/'tmp/matplotlib')}
        tail=deque(maxlen=40)
        with log_path.open('w',encoding='utf-8',buffering=1) as log:
            log.write(shlex.join(command)+'\n')
            with subprocess.Popen(command,cwd=self.root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,encoding='utf-8',errors='replace',bufsize=1) as process:
                try:
                    for line in process.stdout:tail.append(line);print(line,end='',flush=True);log.write(line)
                    code=process.wait()
                except KeyboardInterrupt:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try:process.wait(timeout=5)
                            except subprocess.TimeoutExpired:process.kill();process.wait()
                    print('Terhenti. Ulang sel; record/fit yang lengkap diverifikasi dan dilewati.',flush=True);raise
        if code:raise RuntimeError(f'Endpoint gagal (exit {code}). Log: {log_path}\n'+''.join(tail))
        print('Selesai:',log_path.name,flush=True);return log_path
