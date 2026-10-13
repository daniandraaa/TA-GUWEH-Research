"""Metadata pasien, segmentasi, dan dataset berbasis memory map."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .artifacts import ROOT, lock, read_json, sha256_file, write_json

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUBLOCATIONS = {
    "LVOT": ["LCC", "RCC", "AMC", "LCC-RCC Ommisure"],
    "Non-OT": ["LC", "AC", "RC"],
    "RVOT": ["PosteriorSeptal", "AnteriorSeptal", "FreeWall", "RVOTOther"],
}


def patient_id(value):
    return str(int(value)) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)


def metadata(config):
    diagnosis = pd.read_excel(ROOT / "Diagnosis.xlsx")
    diagnosis["hospital_id"] = diagnosis["HospitalID"].map(patient_id)
    if diagnosis["hospital_id"].duplicated().any():
        raise ValueError("HospitalID harus unik pada metadata diagnosis.")
    mapping = {sub: cls for cls, subs in SUBLOCATIONS.items() for sub in subs}
    diagnosis["class_name"] = diagnosis["Sublocation"].map(mapping)
    train = set(map(patient_id, np.load(ROOT / "preprocessed_data/train_ids.npy", allow_pickle=True)))
    test = set(map(patient_id, np.load(ROOT / "preprocessed_data/test_ids.npy", allow_pickle=True)))
    if train & test:
        raise ValueError("Pasien train dan test beririsan.")
    if train | test != set(diagnosis["hospital_id"]):
        raise ValueError("ID split tidak sesuai dengan metadata diagnosis.")
    records = []
    for _, row in diagnosis.iterrows():
        cls = row["class_name"]
        record = {
            "hospital_id": row["hospital_id"],
            "sublocation": None if pd.isna(row["Sublocation"]) else str(row["Sublocation"]),
            "class_name": None if pd.isna(cls) else cls,
            "label": -1 if pd.isna(cls) else config["class_names"].index(cls),
            "historical_split": "development" if row["hospital_id"] in train else "test",
        }
        record["pool"] = record["historical_split"] if record["label"] >= 0 else "summit" if record["sublocation"] == "Summit" else "unknown"
        records.append(record)
    return records


def segment_windows(signal, length, overlap):
    step = int(length * (1 - overlap))
    if not 0 < step <= length:
        raise ValueError("Overlap harus berada dalam [0, 1).")
    if len(signal) < length:
        segment = np.zeros((length, signal.shape[1]), dtype=np.float32)
        segment[:len(signal)] = signal
        yield 0, len(signal), segment.T
    else:
        for start in range(0, len(signal) - length + 1, step):
            yield start, length, signal[start:start + length].T


def prepare_pool(run, config, pool):
    with lock(run / "data" / pool / ".lock"):
        return _prepare_pool(run, config, pool)


def _prepare_pool(run, config, pool):
    directory = run / "data" / pool
    complete = directory / "manifest.json"
    if complete.exists():
        if not (directory / "X.npy").exists() or not (directory / "segments.csv").exists():
            raise RuntimeError(f"Cache data tidak lengkap: {directory}")
        return directory
    all_metadata = run / "data/patients.json"
    if not all_metadata.exists():
        write_json(all_metadata, metadata(config))
    patients = [r for r in read_json(all_metadata) if r["pool"] == pool]
    if not patients:
        raise ValueError(f"Pool kosong: {pool}")
    directory.mkdir(parents=True, exist_ok=True)
    arrays, segments, hashes = [], [], {}
    for record in patients:
        path = ROOT / "preprocessed_data/dwt_denoised" / f"{record['hospital_id']}.csv"
        signal = pd.read_csv(path, usecols=LEADS, dtype=np.float32)[LEADS].to_numpy()
        if not np.isfinite(signal).all():
            raise ValueError(f"Sinyal tidak finite: {path.name}")
        hashes[path.name] = sha256_file(path)
        for start, valid_length, segment in segment_windows(signal, config["segment_length"], config["overlap"]):
            arrays.append(segment.copy())
            segments.append({"hospital_id": record["hospital_id"], "label": record["label"], "start_sample": start, "valid_samples": valid_length})
    target = directory / "X.npy"
    temporary = directory / "X.tmp.npy"
    mmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(arrays), 12, config["segment_length"]))
    for i, value in enumerate(arrays):
        mmap[i] = value
    mmap.flush()
    del mmap, arrays
    temporary.replace(target)
    pd.DataFrame(segments).to_csv(directory / "segments.csv", index=False)
    write_json(complete, {"pool": pool, "n_patients": len(patients), "n_segments": len(segments), "patients": patients, "signal_sha256": hashes})
    print(f"DATA {pool}: {len(patients)} pasien / {len(segments)} segmen", flush=True)
    return directory


def load_pool(run, pool="development"):
    directory = run / "data" / pool
    segments = pd.read_csv(directory / "segments.csv", dtype={"hospital_id": str})
    return np.load(directory / "X.npy", mmap_mode="r"), segments


class ECGDataset(Dataset):
    def __init__(self, X, labels, indices=None, scaler=None):
        self.X, self.labels = X, np.asarray(labels, dtype=np.int64)
        self.indices = np.arange(len(X)) if indices is None else np.asarray(indices)
        self.mean = None if scaler is None else np.asarray(scaler.mean_, dtype=np.float32)[:, None]
        self.scale = None if scaler is None else np.asarray(scaler.scale_, dtype=np.float32)[:, None]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        value = np.array(self.X[index], dtype=np.float32, copy=True)
        if self.mean is not None:
            value -= self.mean
            value /= self.scale
        return torch.from_numpy(value), torch.tensor(int(self.labels[index]), dtype=torch.long)
