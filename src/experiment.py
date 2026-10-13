"""Tahap eksperimen terpisah; hold-out dibaca hanya pada inferensi."""
import gc
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .artifacts import canonical_hash, lock, read_json, seed_everything, write_json
from .augmentation import train_fold_gan
from .data import ECGDataset, load_pool, prepare_pool
from .evaluation import predict_patient, prediction_table, report
from .models import build_model
from .preprocessing import prepare_scaler
from .splits import fold_indices, prepare_folds
from .training import fit_classifier

METRICS = ["accuracy", "precision", "sensitivity", "specificity", "f1", "auc"]


def prepare_data(run, config):
    return prepare_pool(run, config, "development")


def prepare_fold(run, config, fold_number, device, resume=False, gan=True):
    prepare_data(run, config)
    folds = prepare_folds(run, config)
    fold = folds[fold_number - 1]
    X, segments = load_pool(run)
    train, val = fold_indices(segments, fold)
    scaler = prepare_scaler(run / "folds" / f"fold_{fold_number}", X, train, fold["train_ids"])
    if gan:
        dataset = ECGDataset(X, segments["label"], train, scaler)
        train_fold_gan(dataset, run / "augmentation" / f"fold_{fold_number}", config, device, fold["train_ids"], config["seed"] + 1000 + fold_number, resume)
    return X, segments, train, val, scaler, fold


def training_settings(config, scenario, baseline=False):
    settings = dict(config["training"])
    settings.update(config["baseline"] if baseline else {"learning_rate": scenario["learning_rate"], "batch_size": scenario["batch_size"]})
    return settings


def bind_case(directory, identity):
    path = directory / "config.json"
    signature = canonical_hash(identity)
    if path.exists() and read_json(path)["case_hash"] != signature:
        raise RuntimeError(f"Identitas kasus berubah: {directory}")
    write_json(path, {"case_hash": signature, **identity})


def run_cv_case(run, config, model_name, scenario, fold_number, device, resume=False, baseline=False):
    root = run / "baseline" if baseline else run
    scenario_id = "default" if baseline else scenario["id"]
    directory = root / model_name / scenario_id / f"fold_{fold_number}"
    with lock(directory / ".lock"):
        if (directory / "metrics.json").exists():
            return read_json(directory / "metrics.json")
        X, segments, train, val, scaler, fold = prepare_fold(run, config, fold_number, device, resume, gan=not baseline)
        settings = training_settings(config, scenario, baseline)
        seed = config["seed"] + 100 * fold_number + 17 * config["models"].index(model_name)
        identity = {"model": model_name, "scenario": scenario_id, "fold": fold_number, "seed": seed, "settings": settings, "architecture": config["model"], "train_patient_ids": fold["train_ids"], "validation_patient_ids": fold["validation_ids"], "experiment_hash": read_json(run / "protocol.json")["experiment_hash"]}
        bind_case(directory, identity)
        seed_everything(seed)
        labels = segments["label"].to_numpy()
        real = ECGDataset(X, labels, train, scaler)
        train_dataset = real
        if not baseline:
            aug = run / "augmentation" / f"fold_{fold_number}"
            sy = np.load(aug / "synthetic_y.npy")
            if len(sy):
                synthetic = np.load(aug / "synthetic_X.npy", mmap_mode="r")
                train_dataset = ConcatDataset([real, ECGDataset(synthetic, sy)])
        loader_rng = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True, generator=loader_rng, num_workers=0, drop_last=False)
        val_loader = DataLoader(ECGDataset(X, labels, val, scaler), batch_size=settings["eval_batch_size"], shuffle=False, num_workers=0)
        weights = None
        if settings["class_weighting"] != "none":
            counts = np.bincount(labels[train], minlength=3)
            values = 1.0 / counts
            weights = torch.tensor(values / values.sum() * 3, dtype=torch.float32)
        model = build_model(model_name, config["model"]).to(device)
        print(f"CV {'baseline ' if baseline else ''}{model_name} {scenario_id} fold {fold_number} | {len(real)} real + {len(train_dataset)-len(real)} synthetic", flush=True)
        result = fit_classifier(model, train_loader, val_loader, segments.iloc[val]["hospital_id"].to_numpy(), device, settings, directory, config["class_names"], resume, weights)
        del model, train_loader, val_loader, X
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        return result


def summarize_cv(run, config, scenarios, baseline=False, require_complete=False):
    root = run / "baseline" if baseline else run
    selected = []
    for model in config["models"]:
        rows, summaries = [], []
        for scenario in ([{"id": "default", **config["baseline"]}] if baseline else scenarios):
            current = []
            for fold in range(1, config["n_folds"] + 1):
                path = root / model / scenario["id"] / f"fold_{fold}" / "metrics.json"
                if path.exists():
                    row = {"model": model, "scenario": scenario["id"], "fold": fold, "learning_rate": scenario["learning_rate"], "batch_size": scenario["batch_size"], **read_json(path)}
                    rows.append(row)
                    current.append(row)
            if len(current) == config["n_folds"]:
                summary = {"model": model, "scenario": scenario["id"], "learning_rate": scenario["learning_rate"], "batch_size": scenario["batch_size"], "median_best_epoch": int(np.median([r["best_epoch"] for r in current]))}
                for metric in METRICS:
                    values = [r[metric] for r in current]
                    if any(v is None for v in values):
                        raise ValueError(f"Metrik CV tidak terdefinisi: {model} {metric}")
                    summary[f"{metric}_mean"] = float(np.mean(values))
                    summary[f"{metric}_std"] = float(np.std(values, ddof=1))
                summaries.append(summary)
        (root / model).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(root / model / "cv_fold_results.csv", index=False)
        pd.DataFrame(summaries).to_csv(root / model / "cv_summary.csv", index=False)
        expected = config["n_folds"] * (1 if baseline else len(scenarios))
        if len(rows) != expected:
            if require_complete:
                raise RuntimeError(f"CV {model} belum lengkap: {len(rows)}/{expected}.")
            continue
        best = sorted(summaries, key=lambda r: (-r["f1_mean"], r["f1_std"], r["scenario"]))[0]
        settings = training_settings(config, best, baseline)
        choice = {**best, "settings": settings, "architecture": config["model"], "final_epochs": best["median_best_epoch"], "selection_source": "patient-level development CV only", "primary_seed": config["selection"]["primary_final_seed"]}
        write_json(root / model / "best_config.json", choice)
        selected.append(choice)
    if len(selected) == len(config["models"]):
        chosen = sorted(selected, key=lambda r: (-r["f1_mean"], r["f1_std"], r["model"]))[0]
        write_json(root / "selected_model.json", {**chosen, "test_used_for_selection": False})
    return selected


def require_selection(run, config, scenarios, baseline=False):
    summarize_cv(run, config, scenarios, baseline, require_complete=True)
    return read_json((run / "baseline" if baseline else run) / "selected_model.json")


def train_final(run, config, scenarios, model_name, device, seed=42, resume=False, baseline=False):
    require_selection(run, config, scenarios, baseline)
    root = run / "baseline" if baseline else run
    choice = read_json(root / model_name / "best_config.json")
    directory = root / model_name / "final" / f"seed_{seed}"
    with lock(directory / ".lock"):
        if (directory / "metrics.json").exists():
            return read_json(directory / "metrics.json")
        X, segments = load_pool(run)
        indices = np.arange(len(X))
        ids = segments["hospital_id"].unique().tolist()
        scaler = prepare_scaler(run / "folds/final", X, indices, ids)
        real = ECGDataset(X, segments["label"], scaler=scaler)
        data = real
        if not baseline:
            aug = run / "augmentation/final"
            train_fold_gan(real, aug, config, device, ids, config["seed"] + 2000, resume)
            sy = np.load(aug / "synthetic_y.npy")
            if len(sy):
                data = ConcatDataset([real, ECGDataset(np.load(aug / "synthetic_X.npy", mmap_mode="r"), sy)])
        settings = dict(choice["settings"])
        settings["epochs"] = choice["final_epochs"]
        bind_case(directory, {"model": model_name, "seed": seed, "settings": settings, "selection": choice, "scaler": str((run / "folds/final/scaler.joblib").relative_to(run)), "label_mapping": dict(enumerate(config["class_names"])), "train_patient_ids": ids, "experiment_hash": read_json(run / "protocol.json")["experiment_hash"]})
        seed_everything(seed)
        loader = DataLoader(data, batch_size=settings["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
        weights = None
        if settings["class_weighting"] != "none":
            counts = np.bincount(segments["label"], minlength=3)
            values = 1.0 / counts
            weights = torch.tensor(values / values.sum() * 3, dtype=torch.float32)
        print(f"FINAL {'baseline ' if baseline else ''}{model_name} seed {seed} | epochs {settings['epochs']}", flush=True)
        model = build_model(model_name, config["model"]).to(device)
        result = fit_classifier(model, loader, None, None, device, settings, directory, config["class_names"], resume, weights)
        del model, loader, X
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        return result


def evaluate_test(run, config, scenarios, model_name, device, seed=42, baseline=False):
    require_selection(run, config, scenarios, baseline)
    root = run / "baseline" if baseline else run
    final = root / model_name / "final" / f"seed_{seed}"
    if not (final / "metrics.json").exists():
        raise RuntimeError("Training final belum selesai.")
    directory = root / model_name / "test" / f"seed_{seed}"
    if (directory / "metrics.json").exists():
        return read_json(directory / "metrics.json")
    prepare_pool(run, config, "test")
    X, segments = load_pool(run, "test")
    scaler = joblib.load(run / "folds/final/scaler.joblib")
    loader = DataLoader(ECGDataset(X, segments["label"], scaler=scaler), batch_size=config["training"]["eval_batch_size"], shuffle=False, num_workers=0)
    model = build_model(model_name, config["model"]).to(device)
    model.load_state_dict(torch.load(final / "model.pth", map_location="cpu", weights_only=True))
    ids, truth, probabilities = predict_patient(model, loader, segments["hospital_id"].to_numpy(), device)
    result = report(truth, probabilities, config["class_names"])
    result.update({"model": model_name, "seed": seed, "n_patients": len(ids), "test_status": config["holdout_status"], "test_used_for_selection": False})
    directory.mkdir(parents=True, exist_ok=True)
    table = prediction_table(ids, truth, probabilities, config["class_names"])
    table = table.merge(pd.DataFrame(read_json(run / "data/test/manifest.json")["patients"])[["hospital_id", "sublocation"]], on="hospital_id", validate="one_to_one")
    table.to_csv(directory / "patient_predictions.csv", index=False)
    table.groupby(["true_class", "sublocation"], dropna=False).agg(n_patients=("correct", "size"), n_correct=("correct", "sum"), accuracy=("correct", "mean")).to_csv(directory / "sublocation_summary.csv")
    write_json(directory / "metrics.json", result)
    print(f"TEST {model_name}: macro-F1 {result['metrics']['f1']:.4f}", flush=True)
    del model, loader, X
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result
