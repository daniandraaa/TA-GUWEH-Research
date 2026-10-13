"""Tabel laporan dan visualisasi dari hasil yang tersimpan."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import read_json, write_json
from .models import MODEL_NAMES


def plotting(run):
    os.environ.setdefault("MPLCONFIGDIR", str(run / "plot_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def status(run, config, scenarios):
    gan = []
    for scope in [f"fold_{i}" for i in range(1, 6)] + ["final"]:
        directory = run / "augmentation" / scope
        if (directory / "manifest.json").exists():
            gan.append({"scope": scope, "complete": True, "epochs": config["gan"]["epochs"]})
        elif (directory / "history.csv").exists():
            gan.append({"scope": scope, "complete": False, "epochs": len(pd.read_csv(directory / "history.csv"))})
    result = {"run": str(run), "development_prepared": (run / "data/development/manifest.json").exists(), "folds_prepared": (run / "folds/manifest.json").exists(), "gan": gan, "models": {}}
    for model in config["models"]:
        result["models"][model] = {
            "cv_complete": sum((run / model / scenario["id"] / f"fold_{fold}" / "metrics.json").exists() for scenario in scenarios for fold in range(1, 6)),
            "cv_expected": 45,
            "baseline_cv_complete": sum((run / "baseline" / model / "default" / f"fold_{fold}" / "metrics.json").exists() for fold in range(1, 6)),
            "final_complete": sum((run / model / "final" / f"seed_{seed}" / "metrics.json").exists() for seed in config["selection"]["final_seeds"]),
            "test_complete": sum((run / model / "test" / f"seed_{seed}" / "metrics.json").exists() for seed in config["selection"]["final_seeds"]),
        }
    result["cv_completed"] = sum(r["cv_complete"] for r in result["models"].values())
    result["cv_expected"] = 135
    result["pipeline"] = read_json(run / "progress.json") if (run / "progress.json").exists() else None
    write_json(run / "status.json", result)
    return result


def compare_results(run, config, scenarios, require_complete=True):
    from .experiment import summarize_cv
    summarize_cv(run, config, scenarios, require_complete=require_complete)
    summarize_cv(run, config, scenarios, baseline=True, require_complete=require_complete)
    directory = run / "comparison"
    directory.mkdir(parents=True, exist_ok=True)
    rows, cv = [], []
    for model in config["models"]:
        for scenario_name, root in [("baseline", run / "baseline"), ("fine_tuning", run)]:
            choice = root / model / "best_config.json"
            if choice.exists():
                cv.append({"experiment": scenario_name, **read_json(choice)})
            for seed in config["selection"]["final_seeds"]:
                path = root / model / "test" / f"seed_{seed}" / "metrics.json"
                if not path.exists():
                    if require_complete:
                        raise RuntimeError(f"Hasil test belum tersedia: {path}")
                    continue
                result = read_json(path)
                rows.append({"experiment": scenario_name, "model": model, "seed": seed, "level": "patient", "n_patients": result["n_patients"], **result["metrics"]})
    pd.DataFrame(cv).drop(columns=["settings", "architecture"], errors="ignore").to_csv(directory / "cv_comparison.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "final_test_results.csv", index=False)
    if len(frame):
        frame.groupby(["experiment", "model"])[["accuracy", "sensitivity", "specificity", "f1", "auc"]].agg(["mean", "std"]).to_csv(directory / "seed_summary.csv")
    plt = plotting(run)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, model in zip(axes, config["models"]):
        path = run / model / "cv_summary.csv"
        if path.exists() and path.stat().st_size > 1:
            summary = pd.read_csv(path)
            if len(summary):
                matrix = summary.pivot(index="learning_rate", columns="batch_size", values="f1_mean").sort_index()
                im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
                ax.set_xticks(range(len(matrix.columns)), matrix.columns)
                ax.set_yticks(range(len(matrix.index)), [f"{x:g}" for x in matrix.index])
                for i in range(len(matrix.index)):
                    for j in range(len(matrix.columns)):
                        ax.text(j, i, f"{matrix.iloc[i,j]:.3f}", ha="center", va="center", color="white")
                fig.colorbar(im, ax=ax, shrink=0.75, label="Mean CV macro-F1")
        ax.set_title(MODEL_NAMES[model])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Learning rate")
    fig.tight_layout()
    fig.savefig(directory / "grid_macro_f1.png", dpi=150)
    plt.close(fig)
    if len(frame):
        fig, ax = plt.subplots(figsize=(9, 5))
        table = frame.groupby(["model", "experiment"])["f1"].mean().unstack().reindex(config["models"])
        table.plot.bar(ax=ax, ylim=(0, 1), rot=0)
        ax.set_ylabel("Patient-level macro-F1")
        ax.set_title("Baseline dan fine-tuning pada hold-out historis")
        fig.tight_layout()
        fig.savefig(directory / "baseline_vs_finetune.png", dpi=150)
        plt.close(fig)
    lines = ["# Hasil eksperimen revisi TA", "", "Hold-out ini telah dipakai dalam eksperimen lama. Pemilihan konfigurasi dan model pada revisi dilakukan dari development CV sebelum inferensi.", "", "## Pemilihan dari CV", ""]
    for row in cv:
        lines.append(f"- {row['experiment']} {MODEL_NAMES[row['model']]}: {row['scenario']}; LR {row['learning_rate']:g}; batch {row['batch_size']}; macro-F1 {row['f1_mean']:.4f} ± {row['f1_std']:.4f}; final {row['final_epochs']} epoch.")
    lines.extend(["", "## Evaluasi patient-level", "", "| Skenario | Model | Seed | Accuracy | Sensitivity | Specificity | Macro-F1 | AUC |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(f"| {row['experiment']} | {MODEL_NAMES[row['model']]} | {row['seed']} | {row['accuracy']:.4f} | {row['sensitivity']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} | {row['auc']:.4f} |")
    lines.extend(["", "Metrik CV dan test tidak digunakan untuk tujuan yang sama. CV dipakai untuk pemilihan; test untuk pelaporan final. Sampling rate 1.000 Hz masih mengikuti asumsi kode historis. Fine-tuning mencakup GAN dan optimasi parameter, sehingga selisih hasil tidak mengisolasi pengaruh learning rate saja."])
    (directory / "HASIL_REVISI.md").write_text("\n".join(lines) + "\n")
    return rows
