"""CLI runner, orchestration sequential, dan status eksperimen."""
import argparse
import contextlib
from datetime import datetime, timezone
import json
import time
import traceback
import sys
import torch
from .artifacts import choose_device, lock, open_run, write_json
from .experiment import evaluate_test, prepare_data, prepare_fold, require_selection, run_cv_case, summarize_cv, train_final
from .reports import compare_results, status
from .splits import prepare_folds


class LogStream:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.log.write(value)
        return self.stream.write(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def progress(run, phase, detail, state="running"):
    write_json(run / "progress.json", {"state": state, "phase": phase, "detail": detail, "updated_at": datetime.now(timezone.utc).isoformat()})


def pipeline(run, config, scenarios, device, resume, skip_baseline=False, skip_shap=False):
    with lock(run / ".pipeline.lock"):
        try:
            progress(run, "prepare", "Development data dan manifest fold")
            prepare_data(run, config)
            prepare_folds(run, config)
            for fold in range(1, config["n_folds"] + 1):
                progress(run, "augmentation", f"GAN fold {fold}")
                prepare_fold(run, config, fold, device, resume)
                status(run, config, scenarios)
            for model in config["models"]:
                for scenario in scenarios:
                    for fold in range(1, config["n_folds"] + 1):
                        progress(run, "tuning", f"{model} {scenario['id']} fold {fold}")
                        run_cv_case(run, config, model, scenario, fold, device, resume)
                        summarize_cv(run, config, scenarios)
                        status(run, config, scenarios)
            progress(run, "selection", "Seleksi konfigurasi dan model utama dari CV")
            require_selection(run, config, scenarios)
            if not skip_baseline:
                for model in config["models"]:
                    for fold in range(1, config["n_folds"] + 1):
                        progress(run, "baseline", f"{model} fold {fold}")
                        run_cv_case(run, config, model, None, fold, device, resume, baseline=True)
                        summarize_cv(run, config, scenarios, baseline=True)
                        status(run, config, scenarios)
                require_selection(run, config, scenarios, baseline=True)
            for baseline in ([False] if skip_baseline else [False, True]):
                for model in config["models"]:
                    for seed in config["selection"]["final_seeds"]:
                        progress(run, "final", f"{'baseline ' if baseline else ''}{model} seed {seed}")
                        train_final(run, config, scenarios, model, device, seed, resume, baseline)
                        status(run, config, scenarios)
            # All model/seed choices and all final training are finished before
            # opening the hold-out signal data for the first time in this run.
            for baseline in ([False] if skip_baseline else [False, True]):
                for model in config["models"]:
                    for seed in config["selection"]["final_seeds"]:
                        progress(run, "test", f"{'baseline ' if baseline else ''}{model} seed {seed}")
                        evaluate_test(run, config, scenarios, model, device, seed, baseline)
            from .explain import explain_selected, supplementary_selected
            progress(run, "supplementary", "Prediksi kelompok di luar eksperimen utama")
            supplementary_selected(run, config, device)
            if not skip_shap:
                progress(run, "shap", "SHAP model utama dari CV")
                explain_selected(run, config, device)
                if not skip_baseline:
                    explain_selected(run, config, device, baseline=True)
            progress(run, "report", "Perbandingan dan grafik hasil")
            compare_results(run, config, scenarios, require_complete=not skip_baseline)
            progress(run, "complete", "Semua tahap yang diminta selesai", state="complete")
            status(run, config, scenarios)
        except Exception as error:
            progress(run, "failed", str(error), state="failed")
            (run / "failure.log").write_text(traceback.format_exc())
            status(run, config, scenarios)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pipeline revisi TA dengan grid sembilan skenario dan patient-level CV.")
    parser.add_argument("stage", choices=["prepare", "folds", "augment", "tune", "select", "final", "test", "explain", "compare", "baseline", "pipeline", "status"])
    parser.add_argument("--run-id", default="20260929_grid9_v1")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--model", choices=["encoder", "patchtst", "hierarchical"])
    parser.add_argument("--scenario")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args(argv)
    run, config, scenarios = open_run(args.run_id)
    if args.stage == "status":
        print(json.dumps(status(run, config, scenarios), indent=2))
        return
    device = choose_device(args.device)
    torch.set_num_threads(6)
    print(f"RUN {args.run_id} | device {device} | grid {len(scenarios)} | max CV epochs {config['training']['epochs']}", flush=True)
    models = [args.model] if args.model else config["models"]
    folds = [args.fold] if args.fold else list(range(1, 6))
    selected_scenarios = [s for s in scenarios if s["id"] == args.scenario] if args.scenario else scenarios
    if args.scenario and not selected_scenarios:
        parser.error("Scenario harus S01-S09.")
    if args.stage == "prepare":
        prepare_data(run, config)
    elif args.stage == "folds":
        prepare_data(run, config)
        prepare_folds(run, config)
        for fold in folds:
            prepare_fold(run, config, fold, device, args.resume, gan=False)
    elif args.stage == "augment":
        for fold in folds:
            prepare_fold(run, config, fold, device, args.resume)
    elif args.stage in ["tune", "baseline"]:
        baseline = args.stage == "baseline" or args.baseline
        for model in models:
            for scenario in ([None] if baseline else selected_scenarios):
                for fold in folds:
                    run_cv_case(run, config, model, scenario, fold, device, args.resume, baseline)
                    summarize_cv(run, config, scenarios, baseline)
    elif args.stage == "select":
        require_selection(run, config, scenarios, args.baseline)
    elif args.stage == "final":
        for model in models:
            train_final(run, config, scenarios, model, device, args.seed, args.resume, args.baseline)
    elif args.stage == "test":
        for model in models:
            evaluate_test(run, config, scenarios, model, device, args.seed, args.baseline)
    elif args.stage == "explain":
        from .explain import explain_selected
        explain_selected(run, config, device, args.baseline)
    elif args.stage == "compare":
        compare_results(run, config, scenarios)
    elif args.stage == "pipeline":
        with (run / "pipeline.log").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(LogStream(sys.stdout, log)), contextlib.redirect_stderr(LogStream(sys.stderr, log)):
                pipeline(run, config, scenarios, device, args.resume, args.skip_baseline, args.skip_shap)
    status(run, config, scenarios)
