"""Manifest lima fold dibuat pada pasien, lalu dipetakan ke segmen."""
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from .artifacts import read_json, write_json


def make_patient_folds(patients, n_folds=5, seed=42):
    ids = np.asarray([r["hospital_id"] for r in patients])
    labels = np.asarray([r["label"] for r in patients])
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for i, (train, val) in enumerate(splitter.split(ids, labels, groups=ids), 1):
        training, validation = ids[train].tolist(), ids[val].tolist()
        if set(training) & set(validation):
            raise ValueError("Pasien training dan validation beririsan.")
        if set(labels[val]) != set(labels):
            raise ValueError(f"Fold {i} tidak mencakup seluruh kelas.")
        folds.append({"fold": i, "train_ids": training, "validation_ids": validation})
    if sorted(x for f in folds for x in f["validation_ids"]) != sorted(ids.tolist()):
        raise ValueError("Setiap pasien harus menjadi validation tepat sekali.")
    return folds


def prepare_folds(run, config):
    path = run / "folds/manifest.json"
    if path.exists():
        return read_json(path)
    patients = read_json(run / "data/patients.json")
    development = [r for r in patients if r["pool"] == "development"]
    folds = make_patient_folds(development, config["n_folds"], config["seed"])
    test_ids = {r["hospital_id"] for r in patients if r["pool"] == "test"}
    for fold in folds:
        if (set(fold["train_ids"]) | set(fold["validation_ids"])) & test_ids:
            raise ValueError("Pasien test masuk fold.")
    write_json(path, folds)
    return folds


def fold_indices(segments, fold):
    train = np.flatnonzero(segments["hospital_id"].isin(fold["train_ids"]).to_numpy())
    val = np.flatnonzero(segments["hospital_id"].isin(fold["validation_ids"]).to_numpy())
    return train, val
