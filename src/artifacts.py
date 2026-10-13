"""Artefak atomik, identitas eksperimen, dan locking proses."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def cpu_weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Proses lain sedang menggunakan {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def choose_device(name="auto"):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS tidak tersedia dalam proses ini; gunakan runner di luar sandbox atau --device cpu.")
    return torch.device(name)


def load_configs():
    config = yaml.safe_load((ROOT / "configs/finetune_fixed.yaml").read_text())
    grid = yaml.safe_load((ROOT / "configs/scenarios.yaml").read_text())
    import itertools
    scenarios = [
        {"id": f"S{i:02}", "learning_rate": float(lr), "batch_size": int(bs)}
        for i, (lr, bs) in enumerate(itertools.product(grid["learning_rates"], grid["batch_sizes"]), 1)
    ]
    if len(scenarios) != 9 or len({(s["learning_rate"], s["batch_size"]) for s in scenarios}) != 9:
        raise ValueError("Protokol memerlukan sembilan kombinasi unik.")
    return config, scenarios


def code_fingerprint():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    return canonical_hash({str(p.relative_to(ROOT)): sha256_file(p) for p in paths})


def input_fingerprint():
    small = [ROOT / "Diagnosis.xlsx", ROOT / "preprocessed_data/train_ids.npy", ROOT / "preprocessed_data/test_ids.npy"]
    files = sorted((ROOT / "preprocessed_data/dwt_denoised").glob("*.csv"))
    return canonical_hash({
        "metadata": {str(p.relative_to(ROOT)): sha256_file(p) for p in small},
        "signals": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files],
    })


def open_run(run_id):
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run-id harus satu nama folder.")
    config, scenarios = load_configs()
    directory = ROOT / "results/finetune_revision" / run_id
    identity = {"config": config, "scenarios": scenarios, "code_hash": code_fingerprint(), "input_hash": input_fingerprint()}
    identity["experiment_hash"] = canonical_hash(identity)
    path = directory / "protocol.json"
    if path.exists():
        previous = read_json(path)
        if previous["experiment_hash"] != identity["experiment_hash"]:
            raise RuntimeError("Konfigurasi, kode, atau input berubah. Gunakan run-id baru; hasil lama tidak boleh tercampur.")
    else:
        import platform
        import importlib.metadata
        identity["environment"] = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "PyWavelets", "PyYAML", "shap", "matplotlib", "openpyxl"]}}
        identity["input_fingerprint_method"] = "SHA256 metadata + nama, ukuran, mtime_ns CSV DWT; SHA256 sinyal development disimpan saat persiapan."
        write_json(path, identity)
    return directory, config, scenarios
