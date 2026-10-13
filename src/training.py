"""Training dengan best checkpoint, early stopping, dan resume per epoch."""
import time
import numpy as np
import pandas as pd
import torch
from torch import nn

from .artifacts import cpu_weights, read_json, restore_rng, rng_state, save_torch, write_json
from .evaluation import compute_metrics, predict_patient, prediction_table


def optimizer_for(model, settings):
    cls = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[settings["optimizer"]]
    optimizer = cls(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["scheduler_t_max"], eta_min=settings["minimum_lr"])
    return optimizer, scheduler


def train_epoch(model, loader, criterion, optimizer, device, clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss tidak finite.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(labels)
        correct += int((logits.detach().argmax(1) == labels).sum().cpu())
        total += len(labels)
    return total_loss / total, correct / total


def fit_classifier(model, train_loader, val_loader, val_groups, device, settings, directory, class_names, resume=False, class_weights=None):
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "metrics.json"
    if complete.exists():
        return read_json(complete)
    optimizer, scheduler = optimizer_for(model, settings)
    criterion = nn.CrossEntropyLoss(weight=None if class_weights is None else class_weights.to(device), label_smoothing=settings["label_smoothing"])
    last_path = directory / "last.pt"
    start, history, best_f1, best_epoch, patience = 0, [], -1.0, 0, 0
    elapsed_previous = 0.0
    if last_path.exists():
        if not resume:
            raise RuntimeError(f"Training terputus ditemukan di {directory}; gunakan --resume.")
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("completed"):
            raise RuntimeError("Marker completion tidak konsisten dengan metrics.json.")
        if state["device"] != str(device):
            raise RuntimeError("Resume harus memakai jenis perangkat yang sama.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, history = state["epoch"], state["history"]
        best_f1, best_epoch, patience = state["best_f1"], state["best_epoch"], state["patience"]
        elapsed_previous = state["elapsed_seconds"]
        restore_rng(state["rng"])
        train_loader.generator.set_state(state["loader_rng"])
        print(f"RESUME {directory.name}: epoch {start}", flush=True)
    began = time.monotonic()
    for epoch in range(start, settings["epochs"]):
        if val_loader is not None and patience >= settings["patience"]:
            break
        epoch_start = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device, settings["gradient_clip"])
        row = {"epoch": epoch + 1, "learning_rate": lr, "train_loss": loss, "train_accuracy_segment": acc}
        if val_loader is not None:
            _, truth, probabilities = predict_patient(model, val_loader, val_groups, device)
            metrics = compute_metrics(truth, probabilities)
            row.update({f"validation_{k}": v for k, v in metrics.items()})
            if metrics["f1"] > best_f1:
                best_f1, best_epoch, patience = metrics["f1"], epoch + 1, 0
                save_torch(directory / "best.pth", cpu_weights(model))
            else:
                patience += 1
        else:
            best_epoch = epoch + 1
            save_torch(directory / "model.pth", cpu_weights(model))
        scheduler.step()
        row["seconds"] = time.monotonic() - epoch_start
        history.append(row)
        elapsed = elapsed_previous + time.monotonic() - began
        save_torch(last_path, {"epoch": epoch + 1, "device": str(device), "model": cpu_weights(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history, "best_f1": best_f1, "best_epoch": best_epoch, "patience": patience, "elapsed_seconds": elapsed, "rng": rng_state(), "loader_rng": train_loader.generator.get_state()})
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        score = f" | val macro-F1 {row['validation_f1']:.4f}" if val_loader is not None else ""
        print(f"  ep {epoch+1}/{settings['epochs']} | loss {loss:.4f}{score} | {row['seconds']:.1f}s", flush=True)
    elapsed = elapsed_previous + time.monotonic() - began
    result = {"best_epoch": best_epoch, "epochs_ran": len(history), "elapsed_seconds": elapsed}
    if val_loader is not None:
        model.load_state_dict(torch.load(directory / "best.pth", map_location="cpu", weights_only=True))
        ids, truth, probabilities = predict_patient(model, val_loader, val_groups, device)
        result.update(compute_metrics(truth, probabilities))
        prediction_table(ids, truth, probabilities, class_names).to_csv(directory / "validation_predictions.csv", index=False)
        result["n_validation_patients"] = len(ids)
    write_json(complete, result)
    # Completed cases retain the best model and history; optimizer state is only
    # necessary while a case is running. Compacting avoids duplicating gigabytes.
    save_torch(last_path, {"completed": True, "best_epoch": best_epoch})
    return result
