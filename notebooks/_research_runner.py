"""Pengaturan eksekusi notebook: output langsung, log, dan interupsi proses anak."""
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import signal
import subprocess


class ResearchRunner:
    def __init__(self, root, run_id, device="mps", label="manual"):
        self.root = Path(root)
        self.python = self.root / ".venv/bin/python"
        self.run_id, self.device, self.label = run_id, device, label
        self.run_dir = self.root / "results/research_v2" / run_id
        if not self.python.is_file():
            raise FileNotFoundError(f"Python project tidak ditemukan: {self.python}")
        print(f"Project: {self.root}\nPython training: {self.python}\n"
              f"Notebook: {label} | Device: {device} | Run: {run_id}")

    def run_stage(self, stage, *options):
        command = [str(self.python), "-u", "-m", "research_v2",
                   stage, "--run-id", self.run_id, "--device", self.device, *options]
        log_dir = self.run_dir / "manual_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        log_path = log_dir / f"{stamp}_{self.label}_{stage}.log"
        print("RUN MANUAL:", shlex.join(command), flush=True)
        print("Log:", log_path, flush=True)
        environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            log.write(shlex.join(command) + "\n")
            with subprocess.Popen(command, cwd=self.root, env=environment,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace",
                                  bufsize=1) as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    return_code = process.wait()
                except KeyboardInterrupt:
                    print("\nMenghentikan training; checkpoint epoch terakhir tetap tersedia.", flush=True)
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                    log.write("\nDihentikan manual. Lanjutkan sel yang sama untuk melanjutkan otomatis dari epoch terakhir.\n")
                    raise
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        print(f"Selesai: {stage}. Log tersimpan di {log_path.name}", flush=True)
