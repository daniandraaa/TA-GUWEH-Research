"""Visible, resumable manual CPU stages; no automatic installation or fitting."""
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
import os
import shlex
import signal
import subprocess


class V12Runner:
    def __init__(self,root,run_id):
        self.root=Path(root).resolve();self.run_id=run_id;self.python=self.root/'.venv/bin/python'
        if not isinstance(run_id,str) or not run_id or Path(run_id).name!=run_id or run_id in {'.','..'} or '\\' in run_id:
            raise ValueError('Satu nama run-id diperlukan')
        if not self.python.is_file():raise FileNotFoundError(self.python)
        print(f'Project: {self.root}\nRun: {run_id}\nCPU: fitur dan classifier jenis beat; tanpa model asal anatomi',flush=True)

    def run_stage(self,stage,*options):
        command=[str(self.python),'-u','-m','research_v12',stage,'--run-id',self.run_id,*map(str,options)]
        folder=self.root/'results/research_v12'/self.run_id/'manual_logs';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")}_{stage}.log'
        print('RUN MANUAL:',shlex.join(command),'\nLog:',path,flush=True)
        tail=deque(maxlen=40)
        env={**os.environ,'PYTHONUNBUFFERED':'1','MPLCONFIGDIR':str(self.root/'tmp/matplotlib'),
            'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        with path.open('w',encoding='utf-8',buffering=1) as log:
            log.write(shlex.join(command)+'\n')
            with subprocess.Popen(command,cwd=self.root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1) as process:
                try:
                    for line in process.stdout:tail.append(line);print(line,end='',flush=True);log.write(line)
                    code=process.wait()
                except KeyboardInterrupt:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                    print('Terhenti. Ulang sel; record/fit lengkap diverifikasi dan dilewati.',flush=True);raise
        if code:raise RuntimeError(f'Endpoint {stage} gagal (exit {code}). Log: {path}\n'+''.join(tail))
        print('Selesai:',path.name,flush=True);return path
