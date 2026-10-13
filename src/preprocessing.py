"""Fit scaler hanya dari indeks training yang diberikan."""
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from .artifacts import lock, write_json


def fit_training_scaler(X, train_indices, chunk_size=32):
    if len(train_indices) == 0:
        raise ValueError("Training kosong.")
    scaler = StandardScaler()
    for start in range(0, len(train_indices), chunk_size):
        block = np.asarray(X[train_indices[start:start + chunk_size]])
        scaler.partial_fit(block.transpose(0, 2, 1).reshape(-1, block.shape[1]))
    return scaler


def prepare_scaler(directory, X, train_indices, patient_ids):
    with lock(directory / ".scaler.lock"):
        return _prepare_scaler(directory, X, train_indices, patient_ids)


def _prepare_scaler(directory, X, train_indices, patient_ids):
    path = directory / "scaler.joblib"
    if path.exists() and (directory / "scaler_provenance.json").exists():
        return joblib.load(path)
    directory.mkdir(parents=True, exist_ok=True)
    scaler = fit_training_scaler(X, train_indices)
    temporary = directory / "scaler.tmp.joblib"
    joblib.dump(scaler, temporary)
    temporary.replace(path)
    write_json(directory / "scaler_provenance.json", {"fit_patient_ids": list(patient_ids), "n_segments": len(train_indices), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()})
    return scaler
