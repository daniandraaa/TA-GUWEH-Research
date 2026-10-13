"""Conditional GAN hanya belajar dari pasien training pada scope terkait."""
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import cpu_weights, lock, read_json, restore_rng, rng_state, save_torch, seed_everything, write_json
from .gan_models import Generator1D, Discriminator1D


def train_fold_gan(dataset, directory, config, device, patient_ids, seed, resume=False):
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "manifest.json"
    with lock(directory / ".lock"):
        if complete.exists():
            return read_json(complete)
        seed_everything(seed)
        settings = config["gan"]
        generator = Generator1D(latent_dim=settings["latent_dim"], embed_dim=settings["embed_dim"], num_classes=len(config["class_names"]), seq_length=config["segment_length"]).to(device)
        discriminator = Discriminator1D(num_classes=len(config["class_names"]), embed_dim=settings["embed_dim"], seq_length=config["segment_length"]).to(device)
        opt_g = torch.optim.Adam(generator.parameters(), lr=settings["learning_rate"], betas=tuple(settings["betas"]))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=settings["learning_rate"], betas=tuple(settings["betas"]))
        loader_rng = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True, generator=loader_rng, num_workers=0)
        criterion = nn.BCELoss()
        last = directory / "last.pt"
        start, history, elapsed_before = 0, [], 0.0
        if last.exists():
            if not resume:
                raise RuntimeError(f"GAN terputus di {directory}; gunakan --resume.")
            state = torch.load(last, map_location="cpu", weights_only=False)
            if state.get("completed"):
                raise RuntimeError("Manifest GAN tidak konsisten.")
            if state["device"] != str(device):
                raise RuntimeError("Resume GAN harus memakai jenis perangkat yang sama.")
            generator.load_state_dict(state["generator"])
            discriminator.load_state_dict(state["discriminator"])
            opt_g.load_state_dict(state["opt_g"])
            opt_d.load_state_dict(state["opt_d"])
            start, history, elapsed_before = state["epoch"], state["history"], state["elapsed_seconds"]
            restore_rng(state["rng"])
            loader_rng.set_state(state["loader_rng"])
        began = time.monotonic()
        for epoch in range(start, settings["epochs"]):
            generator.train()
            discriminator.train()
            g_total, d_total, count = 0.0, 0.0, 0
            for real, labels in loader:
                real = (real.to(device) / settings["scale"]).clamp(-1, 1)
                labels = labels.to(device)
                valid = torch.full((len(real), 1), settings["real_target"], device=device)
                fake = torch.full_like(valid, settings["fake_target"])
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                opt_g.zero_grad(set_to_none=True)
                noise = torch.randn(len(real), settings["latent_dim"], device=device)
                generated = generator(noise, labels)
                g_loss = criterion(discriminator(generated, labels), valid)
                if not torch.isfinite(g_loss):
                    raise RuntimeError("GAN generator loss tidak finite.")
                g_loss.backward()
                opt_g.step()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                d_loss = (criterion(discriminator(real, labels), valid) + criterion(discriminator(generated.detach(), labels), fake)) / 2
                if not torch.isfinite(d_loss):
                    raise RuntimeError("GAN discriminator loss tidak finite.")
                d_loss.backward()
                opt_d.step()
                g_total += float(g_loss.detach().cpu()) * len(real)
                d_total += float(d_loss.detach().cpu()) * len(real)
                count += len(real)
            history.append({"epoch": epoch + 1, "generator_loss": g_total/count, "discriminator_loss": d_total/count})
            elapsed = elapsed_before + time.monotonic() - began
            save_torch(last, {"epoch": epoch + 1, "device": str(device), "generator": cpu_weights(generator), "discriminator": cpu_weights(discriminator), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "history": history, "elapsed_seconds": elapsed, "rng": rng_state(), "loader_rng": loader_rng.get_state()})
            pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
            print(f"GAN {directory.name} ep {epoch+1}/{settings['epochs']} | G {g_total/count:.4f} D {d_total/count:.4f} | elapsed {elapsed/60:.1f}m", flush=True)
        save_torch(directory / "generator.pth", cpu_weights(generator))
        save_torch(directory / "discriminator.pth", cpu_weights(discriminator))
        counts = np.bincount(dataset.labels[dataset.indices], minlength=len(config["class_names"]))
        needed = counts.max() - counts
        total = int(needed.sum())
        if total:
            synthetic = np.lib.format.open_memmap(directory / "synthetic_X.tmp.npy", mode="w+", dtype=np.float32, shape=(total, 12, config["segment_length"]))
        else:
            synthetic = np.empty((0, 12, config["segment_length"]), dtype=np.float32)
        synthetic_labels = []
        offset = 0
        seed_everything(seed + 10000)
        generator.eval()
        with torch.no_grad():
            for cls, number in enumerate(needed):
                for begin in range(0, int(number), settings["batch_size"]):
                    batch = min(settings["batch_size"], int(number) - begin)
                    labels = torch.full((batch,), cls, dtype=torch.long, device=device)
                    noise = torch.randn(batch, settings["latent_dim"], device=device)
                    values = (generator(noise, labels) * settings["scale"]).cpu().numpy()
                    synthetic[offset:offset + batch] = values
                    synthetic_labels.extend([cls] * batch)
                    offset += batch
        if total:
            synthetic.flush()
        else:
            np.save(directory / "synthetic_X.tmp.npy", synthetic)
        del synthetic
        (directory / "synthetic_X.tmp.npy").replace(directory / "synthetic_X.npy")
        np.save(directory / "synthetic_y.npy", np.asarray(synthetic_labels, dtype=np.int64))
        result = {"fit_patient_ids": list(patient_ids), "seed": seed, "n_real_segments": len(dataset), "n_synthetic_segments": total, "real_class_counts": counts.tolist(), "augmented_class_counts": (counts + needed).tolist(), "epochs": settings["epochs"], "settings": settings, "elapsed_seconds": elapsed_before + time.monotonic() - began}
        write_json(complete, result)
        save_torch(last, {"completed": True, "epochs": settings["epochs"]})
        return result
