"""SHAP untuk model utama yang telah dipilih dari CV."""
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .artifacts import read_json, seed_everything, write_json
from .data import ECGDataset, LEADS, load_pool, prepare_pool
from .evaluation import predict_patient, prediction_table
from .models import build_model
from .reports import plotting


def explain_selected(run, config, device, baseline=False):
    root = run / "baseline" if baseline else run
    choice = read_json(root / "selected_model.json")
    model_name, seed = choice["model"], config["selection"]["primary_final_seed"]
    final = root / model_name / "final" / f"seed_{seed}"
    if not (root / model_name / "test" / f"seed_{seed}" / "metrics.json").exists():
        raise RuntimeError("Inferensi final perlu diselesaikan sebelum SHAP.")
    output = root / model_name / "explain"
    if (output / "manifest.json").exists():
        return read_json(output / "manifest.json")
    import shap
    seed_everything(config["shap"]["seed"])
    X, segments = load_pool(run)
    Xt, test_segments = load_pool(run, "test")
    scaler = joblib.load(run / "folds/final/scaler.joblib")
    rng = np.random.default_rng(config["shap"]["seed"])
    background_indices = rng.choice(len(X), min(len(X), config["shap"]["background_segments"]), replace=False)
    explained_indices = rng.choice(len(Xt), min(len(Xt), config["shap"]["explain_segments"]), replace=False)
    train_dataset = ECGDataset(X, segments["label"], scaler=scaler)
    test_dataset = ECGDataset(Xt, test_segments["label"], scaler=scaler)
    background = torch.stack([train_dataset[int(i)][0] for i in background_indices]).to(device)
    explained = torch.stack([test_dataset[int(i)][0] for i in explained_indices]).to(device)
    model = build_model(model_name, config["model"]).to(device)
    model.load_state_dict(torch.load(final / "model.pth", map_location="cpu", weights_only=True))
    model.eval()
    print(f"SHAP {model_name}: {len(explained)} segmen", flush=True)
    values = shap.GradientExplainer(model, background).shap_values(explained, nsamples=config["shap"]["integration_samples"], rseed=config["shap"]["seed"])
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    expected = (len(explained), 12, config["segment_length"], 3)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"Output SHAP tidak valid: {values.shape}, expected {expected}")
    truth = test_segments.iloc[explained_indices]["label"].to_numpy()
    selected = np.stack([values[i, :, :, label] for i, label in enumerate(truth)])
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "shap_true_class_values.npy", selected.astype(np.float32))
    importance = pd.DataFrame({"lead": LEADS, "importance": np.abs(selected).mean(axis=(0, 2))}).sort_values("importance", ascending=False)
    importance.to_csv(output / "lead_importance.csv", index=False)
    test_segments.iloc[explained_indices].assign(segment_index=explained_indices).to_csv(output / "explained_segments.csv", index=False)
    plt = plotting(run)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance["lead"].iloc[::-1], importance["importance"].iloc[::-1])
    ax.set_xlabel("Mean absolute SHAP pada output kelas benar")
    ax.set_title(f"Kontribusi sadapan: {model_name}")
    fig.tight_layout()
    fig.savefig(output / "lead_importance.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    time = np.arange(config["segment_length"]) / config["sampling_rate"]
    signals = explained.detach().cpu().numpy()
    for ax, i in zip(axes, range(min(3, len(explained)))):
        ax.plot(time, signals[i, 1], color="#333333", linewidth=0.7)
        ax2 = ax.twinx()
        ax2.plot(time, selected[i, 1], color="#cc3333", linewidth=0.6, alpha=0.8)
        ax.set_title(f"Pasien {test_segments.iloc[explained_indices[i]]['hospital_id']} - Lead II")
        ax.set_ylabel("DWT Z-score")
        ax2.set_ylabel("SHAP")
    axes[-1].set_xlabel("Waktu (detik, asumsi 1.000 Hz)")
    fig.tight_layout()
    fig.savefig(output / "temporal_shap.png", dpi=150)
    plt.close(fig)
    result = {"method": "shap.GradientExplainer", "model": model_name, "selection_source": "development CV", "output_explained": "true-class logit", "input_representation": "standardized DWT-denoised signal, not DWT coefficients", "background_pool": "development", "background_indices": background_indices.tolist(), "explained_pool": "historical_test", "explained_indices": explained_indices.tolist(), "settings": config["shap"]}
    write_json(output / "manifest.json", result)
    return result


def supplementary_selected(run, config, device):
    choice = read_json(run / "selected_model.json")
    model_name, seed = choice["model"], config["selection"]["primary_final_seed"]
    scaler = joblib.load(run / "folds/final/scaler.joblib")
    model = build_model(model_name, config["model"]).to(device)
    model.load_state_dict(torch.load(run / model_name / "final" / f"seed_{seed}" / "model.pth", map_location="cpu", weights_only=True))
    output = run / model_name / "supplementary"
    output.mkdir(parents=True, exist_ok=True)
    for pool in ["summit", "unknown"]:
        prepare_pool(run, config, pool)
        X, segments = load_pool(run, pool)
        loader = DataLoader(ECGDataset(X, segments["label"], scaler=scaler), batch_size=config["training"]["eval_batch_size"], shuffle=False)
        ids, truth, probabilities = predict_patient(model, loader, segments["hospital_id"].to_numpy(), device)
        prediction_table(ids, truth, probabilities, config["class_names"]).to_csv(output / f"{pool}_predictions.csv", index=False)
    write_json(output / "manifest.json", {"model": model_name, "selection_source": "development CV", "scored": False, "reason": "Summit berada di luar label tiga kelas; unknown tidak memiliki ground truth sublokasi."})
