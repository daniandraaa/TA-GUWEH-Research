"""Patient-weighted head/tail training with inner selection and exact resume."""
from collections import Counter
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from src.artifacts import (cpu_weights, read_json, save_torch, sha256_file, write_json,
    rng_state, restore_rng)
from src.evaluation import compute_metrics
from research_v6.common import verify_files, write_manifest


class FeatureClassifier(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('feature_mean', torch.as_tensor(mean).clone())
        self.register_buffer('feature_scale', torch.as_tensor(scale).clone())
        self.head = nn.Linear(len(mean), 3)
    def forward(self, values): return self.head((values - self.feature_mean) / self.feature_scale)


class Patients:
    def __init__(self, ids, labels, features=None, tokens=None, groups=None):
        self.ids, self.labels = list(ids), np.asarray(labels, dtype=np.int64)
        self.features, self.tokens, self.groups = features, tokens, groups
        if len(set(ids)) != len(ids) or len(labels) != len(ids): raise ValueError('Unique patient IDs required')
    def __len__(self): return len(self.ids)
    def __getitem__(self, index):
        values = self.features[index] if self.tokens is None else self.tokens[self.groups[index]]
        return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)), int(self.labels[index]), self.ids[index]


@torch.no_grad()
def predict(model, dataset, device):
    model.eval(); probabilities = []
    for i in range(len(dataset)):
        values, _, _ = dataset[i]
        logits = model(values.to(device))
        if logits.shape != (3,) or not torch.isfinite(logits).all(): raise RuntimeError('Invalid patient logits')
        probabilities.append(logits.softmax(-1).cpu().numpy())
    return dataset.ids, dataset.labels.copy(), np.stack(probabilities)


def train_epoch(model, dataset, optimizer, device, weights, batch_size, order_generator, clip):
    model.train(); order = torch.randperm(len(dataset), generator=order_generator).tolist()
    exposure = Counter(); total_loss = correct = steps = 0
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        # Normalize once per effective patient batch. Single-patient weighted CE
        # with reduction='mean' would incorrectly cancel inverse class weights.
        denominator = weights[torch.as_tensor(dataset.labels[batch], device=device)].sum()
        optimizer.zero_grad(set_to_none=True); batch_loss = 0.
        for index in batch:
            values, label, hid = dataset[index]; logits = model(values.to(device))
            loss = -F.log_softmax(logits, -1)[label] * weights[label] / denominator
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite patient training loss')
            loss.backward(); batch_loss += float(loss.detach().cpu())
            correct += int(logits.detach().argmax().cpu()) == label; exposure[hid] += 1
        nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step(); total_loss += batch_loss * len(batch); steps += 1
    if set(exposure) != set(dataset.ids) or set(exposure.values()) != {1}: raise RuntimeError('Each training patient must appear once')
    return {'train_loss': total_loss / len(dataset), 'train_accuracy': correct / len(dataset),
        'optimizer_steps': steps, 'anchor_patients': len(dataset),
        'patient_exposure_json': json.dumps(dict(sorted(exposure.items()))),
        **{f'class_{c}_patients': int((dataset.labels == c).sum()) for c in range(3)}}


def optimizer_for(model, settings):
    if hasattr(model, 'tail'):
        groups = [{'params': model.tail.parameters(), 'lr': settings['learning_rate']},
                  {'params': model.head.parameters(), 'lr': settings['head_learning_rate']}]
    else: groups = [{'params': model.head.parameters(), 'lr': settings['learning_rate']}]
    return torch.optim.AdamW(groups, weight_decay=settings['weight_decay'])


def fit(model, train, validation, device, settings, folder, case_hash, resume_root, stop_after=None):
    folder.mkdir(parents=True, exist_ok=True); resume_root.mkdir(parents=True, exist_ok=True)
    if (folder / 'fit_finished.json').exists():
        done = read_json(folder / 'fit_finished.json'); verify_files(folder, done)
        if done['case_hash'] != case_hash: raise RuntimeError('Finished fit identity changed')
        return read_json(folder / 'fit_result.json')
    counts = np.bincount(train.labels, minlength=3)
    if (counts == 0).any(): raise RuntimeError('Training needs all classes')
    weights = torch.tensor(len(train) / (3 * counts), dtype=torch.float32, device=device)
    optimizer = optimizer_for(model, settings)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings['scheduler_t_max'], eta_min=settings['minimum_lr'])
    generator = torch.Generator().manual_seed(settings['effective_seed'] + 10000)
    last = resume_root / f'{case_hash}.pt'
    history = []; best = -1.; selected = patience = elapsed = start_epoch = 0
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['case_hash'] != case_hash or saved['device'] != str(device): raise RuntimeError('Resume identity/device changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng']); generator.set_state(saved['order_rng'])
        history = saved['history']; start_epoch = saved['epoch']; best = saved['best']
        selected = saved['selected_epoch']; patience = saved['patience']; elapsed = saved['elapsed_seconds']
        print(f'RESUME epoch={start_epoch}', flush=True)
    began = time.monotonic(); maximum = settings['epochs'] if stop_after is None else min(stop_after, settings['epochs'])
    for epoch in range(start_epoch, maximum):
        if validation is not None and patience >= settings['patience']: break
        row = {'epoch': epoch + 1, 'learning_rate': optimizer.param_groups[0]['lr'],
            'head_learning_rate': optimizer.param_groups[-1]['lr'],
            **train_epoch(model, train, optimizer, device, weights, settings['batch_size'], generator, settings['gradient_clip'])}
        if validation is not None:
            _, truth, p = predict(model, validation, device); metrics = compute_metrics(truth, p)
            row.update({f'inner_{key}': value for key, value in metrics.items()})
            if metrics['f1'] > best:
                best = metrics['f1']; selected = epoch + 1; patience = 0
                save_torch(folder / 'selected.pth', cpu_weights(model))
            else: patience += 1
        else: selected = epoch + 1
        scheduler.step(); history.append(row)
        save_torch(last, {'case_hash': case_hash, 'device': str(device), 'epoch': epoch + 1,
            'model': cpu_weights(model), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'rng': rng_state(), 'order_rng': generator.get_state(), 'history': history,
            'best': best, 'selected_epoch': selected, 'patience': patience, 'elapsed_seconds': elapsed + time.monotonic() - began})
        tmp = folder / 'history.csv.tmp'; pd.DataFrame(history).to_csv(tmp, index=False); tmp.replace(folder / 'history.csv')
        suffix = f' inner_F1={row["inner_f1"]:.4f}' if validation is not None else ' fixed refit; outer not evaluated'
        print(f'epoch {epoch+1}/{settings["epochs"]} loss={row["train_loss"]:.4f}' + suffix, flush=True)
    finished = len(history) == settings['epochs'] or validation is not None and patience >= settings['patience']
    if not finished: return {'partial': True, 'epochs_run': len(history)}
    tmp = folder / 'history.csv.tmp'; pd.DataFrame(history).to_csv(tmp, index=False); tmp.replace(folder / 'history.csv')
    if validation is None: save_torch(folder / 'selected.pth', cpu_weights(model))
    result = {'selected_epoch': selected, 'epochs_run': len(history), 'inner_selection': validation is not None,
        'best_inner_f1': best if validation is not None else None,
        'optimizer_steps': int(sum(r['optimizer_steps'] for r in history)),
        'elapsed_seconds': elapsed + time.monotonic() - began,
        'selected_checkpoint_sha256': sha256_file(folder / 'selected.pth')}
    write_json(folder / 'fit_result.json', result)
    write_manifest(folder / 'fit_finished.json', {'case_hash': case_hash, 'device': str(device),
        'files': {n: sha256_file(folder / n) for n in ['selected.pth', 'history.csv', 'fit_result.json']},
        'optimizer_retention': 'local_resume_only_until_predictions_committed'})
    return result


def release_completed_resume(resume_root, case_hash):
    # Only this experiment's temporary optimizer state. Selected learned weights,
    # histories and predictions are retained; prior research artifacts never enter here.
    path = resume_root / f'{case_hash}.pt'
    if path.exists(): path.unlink()
