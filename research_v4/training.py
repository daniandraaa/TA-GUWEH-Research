"""Auditable training: patient loss, class diagnostics, and epoch-exact resume."""
from collections import Counter
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from src.artifacts import cpu_weights, read_json, restore_rng, rng_state, save_torch, sha256_file, write_json
from src.evaluation import compute_metrics, prediction_table, report


@torch.no_grad()
def predict(model, dataset, device, chunk):
    model.eval()
    ids, truth, probabilities, contributions = [], [], [], []
    for i in range(len(dataset)):
        values, label, hid = dataset[i]
        probability, weights = model.all_segments(values.to(device), chunk)
        ids.append(hid); truth.append(label); probabilities.append(probability.cpu().numpy())
        contributions.extend({'hospital_id': hid, 'segment_index': int(index),
                              'weight': float(weight), 'n_patient_segments': len(values)}
                             for index, weight in zip(dataset.indices[i], weights.cpu().tolist()))
    return ids, np.asarray(truth), np.asarray(probabilities), contributions


def class_diagnostics(truth, probabilities, prefix='validation'):
    prediction = probabilities.argmax(1)
    output = {}
    for c in range(probabilities.shape[1]):
        mask = truth == c
        output[f'{prefix}_recall_class_{c}'] = float((prediction[mask] == c).mean()) if mask.any() else None
        output[f'{prefix}_predicted_class_{c}'] = int((prediction == c).sum())
    output[f'{prefix}_missing_predicted_classes'] = int(sum(not (prediction == c).any() for c in range(probabilities.shape[1])))
    return output


def batchnorm_diagnostics(model):
    values = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    if not values:
        return {'batchnorm_layers': 0}
    return {'batchnorm_layers': len(values),
            'batchnorm_running_mean_rms': float(torch.cat([m.running_mean.detach().cpu() for m in values]).square().mean().sqrt()),
            'batchnorm_running_variance_mean': float(torch.cat([m.running_var.detach().cpu() for m in values]).mean())}


def train_epoch(model, loader, optimizer, device, class_weights, clip, mode):
    model.train()
    total_loss, correct, total, norm_total, steps = 0., 0, 0, 0., 0
    exposure, class_exposure = Counter(), Counter()
    for batch in loader:
        if mode == 'bag':
            inputs, labels, mask, ids = batch
            mask = mask.to(device)
        else:
            inputs, labels, ids = batch
            mask = None
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logp = model(inputs, mask)
        loss = F.nll_loss(logp, labels, weight=class_weights)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
        count = len(labels)
        total_loss += float(loss.detach().cpu()) * count
        correct += int((logp.detach().argmax(1) == labels).sum().cpu())
        total += count; steps += 1; norm_total += float(norm.detach().cpu())
        exposure.update(ids); class_exposure.update(labels.detach().cpu().tolist())
    return {'train_loss_minibatch_weighted_mean': total_loss / total,
            f'train_accuracy_{"patient_bag" if mode == "bag" else "segment"}': correct / total,
            'optimizer_steps': steps, 'gradient_norm_before_clip_mean': norm_total / steps,
            'unique_training_patients_seen': len(exposure),
            'patient_exposure_min': min(exposure.values()), 'patient_exposure_max': max(exposure.values()),
            'native_patient_exposure_json': json.dumps(dict(sorted(exposure.items()))) if mode == 'segment' else None,
            **{f'train_class_{c}_draws': class_exposure[c] for c in range(model.classes)}}


def fit(model, loader, validation, device, settings, directory, class_names, class_weights, chunk, mode='bag'):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'metrics.json').exists():
        return read_json(directory / 'metrics.json')
    optimizer = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}[settings['optimizer']](
        model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings['scheduler_t_max'],
                                                        eta_min=settings['minimum_lr'])
    weights = class_weights.to(device)
    last = directory / 'last.pt'
    epoch_start, best, best_epoch, patience, history, elapsed_before = 0, -1., 0, 0, [], 0.
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state.get('completed') or state['device'] != str(device):
            raise RuntimeError('Resume marker/device mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); restore_rng(state['rng'])
        loader.generator.set_state(state['loader_rng'])
        if state.get('dataset_rng') is not None:
            loader.dataset.generator.set_state(state['dataset_rng'])
        epoch_start, best, best_epoch, patience = state['epoch'], state['best'], state['best_epoch'], state['patience']
        history, elapsed_before = state['history'], state['elapsed_seconds']
        print(f'RESUME epoch={epoch_start}', flush=True)
    start = time.monotonic()
    for epoch in range(epoch_start, settings['epochs']):
        if validation is not None and patience >= settings['patience']:
            break
        began = time.monotonic()
        row = {'epoch': epoch + 1, 'learning_rate': optimizer.param_groups[0]['lr'],
               **train_epoch(model, loader, optimizer, device, weights, settings['gradient_clip'], mode),
               **batchnorm_diagnostics(model)}
        if validation is not None:
            _, truth, probabilities, _ = predict(model, validation, device, chunk)
            metrics = compute_metrics(truth, probabilities)
            row.update({f'validation_{k}': v for k, v in metrics.items()})
            row.update(class_diagnostics(truth, probabilities))
            row['validation_loss_patient_weighted'] = float(F.nll_loss(
                torch.from_numpy(np.log(np.clip(probabilities, 1e-30, 1))), torch.from_numpy(truth),
                weight=class_weights.cpu()))
            if metrics['f1'] > best:
                best, best_epoch, patience = metrics['f1'], epoch + 1, 0
                save_torch(directory / 'best.pth', cpu_weights(model))
            else:
                patience += 1
            diagnostic = ','.join(str(row[f'validation_predicted_class_{c}']) for c in range(model.classes))
            print(f'epoch {epoch+1}/{settings["epochs"]} loss={row["train_loss_minibatch_weighted_mean"]:.4f} '
                  f'val_F1={metrics["f1"]:.4f} predicted_classes=[{diagnostic}] best_epoch={best_epoch}', flush=True)
        else:
            best_epoch = epoch + 1
            save_torch(directory / 'model.pth', cpu_weights(model))
            print(f'final epoch {epoch+1}/{settings["epochs"]} loss={row["train_loss_minibatch_weighted_mean"]:.4f}', flush=True)
        scheduler.step()
        row['seconds'] = time.monotonic() - began
        history.append(row)
        save_torch(last, {'epoch': epoch + 1, 'device': str(device), 'model': cpu_weights(model),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': rng_state(),
            'loader_rng': loader.generator.get_state(), 'best': best, 'best_epoch': best_epoch,
            'dataset_rng': loader.dataset.generator.get_state() if mode == 'bag' else None,
            'patience': patience, 'history': history, 'elapsed_seconds': elapsed_before + time.monotonic() - start})
        pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
    if validation is not None:
        model.load_state_dict(torch.load(directory / 'best.pth', map_location='cpu', weights_only=True))
        ids, truth, probabilities, contributions = predict(model, validation, device, chunk)
        prediction_table(ids, truth, probabilities, class_names).to_csv(directory / 'validation_predictions.csv', index=False)
        pd.DataFrame(contributions).to_csv(directory / 'validation_segment_weights.csv', index=False)
        result = {**report(truth, probabilities, class_names), **class_diagnostics(truth, probabilities),
                  'n_validation_patients': len(ids), 'selected_checkpoint_sha256': sha256_file(directory / 'best.pth')}
    else:
        result = {'fit_complete': True, 'model_sha256': sha256_file(directory / 'model.pth')}
    result.update({'best_epoch': best_epoch, 'epochs_run': len(history),
                   'elapsed_seconds': elapsed_before + time.monotonic() - start})
    write_json(directory / 'metrics.json', result)
    compact_completed(directory, result)
    return result


def compact_completed(directory, result):
    """CV predictions/history kept; unnecessary optimizer checkpoints not accumulated."""
    save_torch(directory / 'last.pt', {'completed': True, 'best_epoch': result['best_epoch']})
    (directory / 'best.pth').unlink(missing_ok=True)
