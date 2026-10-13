"""Unchanged GAN updates/RNG/budget, new fixed class-distribution objective."""
from collections import Counter
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from src.artifacts import cpu_weights, restore_rng, rng_state, save_torch


def optimizers(generator, discriminator, cfg):
    args = dict(lr=cfg['learning_rate'], betas=tuple(cfg['betas']))
    return torch.optim.Adam(generator.parameters(), **args), torch.optim.Adam(discriminator.parameters(), **args)


def step(generator, discriminator, opt_g, opt_d, real, labels, noise_rng, cfg, objective):
    device = real.device
    def noise():
        return torch.randn(len(real), generator.latent, generator=noise_rng).to(device)
    discriminator.train(); generator.train()
    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad(): fake = generator(noise(), labels)
    scores = discriminator(torch.cat((real, fake)), torch.cat((labels, labels)))
    real_score, fake_score = scores.chunk(2)
    d_loss = F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
    if not torch.isfinite(d_loss): raise RuntimeError('Nonfinite discriminator loss')
    d_loss.backward()
    d_norm = nn.utils.clip_grad_norm_(discriminator.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
    opt_d.step(); opt_d.zero_grad(set_to_none=True)
    # Avoid modifying spectral normalization buffers during the generator backward.
    discriminator.eval()
    for parameter in discriminator.parameters(): parameter.requires_grad_(False)
    try:
        opt_g.zero_grad(set_to_none=True)
        fake = generator(noise(), labels)
        adversarial = -discriminator(fake, labels).mean()
        auxiliary, family_losses = objective(fake, labels)
        g_loss = adversarial + auxiliary
        if not torch.isfinite(g_loss): raise RuntimeError('Nonfinite generator loss')
        g_loss.backward()
        g_norm = nn.utils.clip_grad_norm_(generator.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
        opt_g.step()
    finally:
        for parameter in discriminator.parameters(): parameter.requires_grad_(True)
    return {'d_loss': float(d_loss.detach().cpu()), 'g_loss': float(g_loss.detach().cpu()),
        'g_adversarial': float(adversarial.detach().cpu()), 'g_fidelity': float(auxiliary.detach().cpu()),
        **{f'fidelity_{k}': float(v.detach().cpu()) for k, v in family_losses.items()},
        'd_gradient_norm': float(d_norm.detach().cpu()), 'g_gradient_norm': float(g_norm.detach().cpu())}


def fit(generator, discriminator, loader, device, cfg, folder, case_hash, seed, objective, stop_after=None):
    opt_g, opt_d = optimizers(generator, discriminator, cfg)
    noise_rng = torch.Generator().manual_seed(seed + 20000)
    last = folder / 'last.pt'
    history, epoch_start, elapsed_before = [], 0, 0.
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['case_hash'] != case_hash or state['device'] != str(device):
            raise RuntimeError('GAN resume identity/device mismatch')
        generator.load_state_dict(state['generator']); discriminator.load_state_dict(state['discriminator'])
        opt_g.load_state_dict(state['opt_g']); opt_d.load_state_dict(state['opt_d'])
        restore_rng(state['rng'])
        loader.generator.set_state(state['loader_rng'])
        loader.dataset.generator.set_state(state['dataset_rng'])
        noise_rng.set_state(state['noise_rng'])
        history, epoch_start, elapsed_before = state['history'], state['epoch'], state['elapsed_seconds']
        print(f'GAN RESUME completed_epoch={epoch_start}', flush=True)
    began = time.monotonic()
    final_epoch = cfg['epochs'] if stop_after is None else min(cfg['epochs'], stop_after)
    for epoch in range(epoch_start, final_epoch):
        started = time.monotonic()
        aggregate, exposure, classes = Counter(), Counter(), Counter()
        total, updates = 0, 0
        for real, labels, ids in loader:
            output = step(generator, discriminator, opt_g, opt_d, real.to(device), labels.to(device), noise_rng, cfg, objective)
            for key, value in output.items(): aggregate[key] += value * len(labels)
            total += len(labels); updates += 1
            exposure.update(ids); classes.update(labels.tolist())
        row = {'epoch': epoch + 1, **{k: v / total for k, v in aggregate.items()},
            'draws': total, 'd_steps': updates, 'g_steps': updates,
            'unique_fit_patients_drawn': len(exposure),
            'patient_exposure_json': json.dumps(dict(sorted(exposure.items()))),
            **{f'class_{c}_draws': classes[c] for c in range(3)},
            'seconds': time.monotonic() - started}
        history.append(row)
        # Only fully completed epochs are committed. An interrupted epoch is replayed.
        save_torch(last, {'case_hash': case_hash, 'device': str(device), 'epoch': epoch + 1,
            'generator': cpu_weights(generator), 'discriminator': cpu_weights(discriminator),
            'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(), 'rng': rng_state(),
            'loader_rng': loader.generator.get_state(), 'dataset_rng': loader.dataset.generator.get_state(),
            'noise_rng': noise_rng.get_state(), 'history': history,
            'elapsed_seconds': elapsed_before + time.monotonic() - began})
        path = folder / 'history.csv'
        temporary = path.with_suffix('.csv.tmp'); pd.DataFrame(history).to_csv(temporary, index=False); temporary.replace(path)
        print(f'epoch {epoch+1}/{cfg["epochs"]} D={row["d_loss"]:.4f} G={row["g_loss"]:.4f} '
            f'fidelity={row["g_fidelity"]:.4f} spectrum={row["fidelity_spectrum"]:.4f} '
            f'roughness={row["fidelity_roughness"]:.4f} correlation={row["fidelity_correlation"]:.4f} '
            f'amplitude={row["fidelity_amplitude"]:.4f} draws={total} patients={len(exposure)} seconds={row["seconds"]:.1f}', flush=True)
    return history
