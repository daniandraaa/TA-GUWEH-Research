"""Periksa kompatibilitas arsitektur lama dan forward/backward aktual."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.artifacts import ROOT, choose_device, load_configs, seed_everything
from src.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config, scenarios = load_configs()
    device = choose_device(args.device)
    torch.set_num_threads(6)
    legacy_names = {"encoder":"Transformer_Encoder", "patchtst":"PatchTST", "hierarchical":"Hierarchical_Transformer"}
    result = []
    for name in config["models"]:
        seed_everything(42)
        model = build_model(name, config["model"]).to(device)
        old = torch.load(ROOT / "baseline_results" / f"final_model_{legacy_names[name]}.pth", map_location="cpu", weights_only=True)
        assert model.state_dict().keys() == old.keys(), name
        assert all(model.state_dict()[key].shape == old[key].shape for key in old), name
        model.load_state_dict(old, strict=True)
        model.train()
        batch = torch.randn(16,12,5000,device=device)
        labels = torch.arange(16,device=device) % 3
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        times=[]
        for iteration in range(3):
            began=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            out=model(batch)
            assert out.shape == (16,3)
            loss=torch.nn.functional.cross_entropy(out,labels)
            loss.backward()
            optimizer.step()
            if device.type == "mps": torch.mps.synchronize()
            times.append(time.monotonic()-began)
        result.append({"model":name, "legacy_state_dict_compatible":True, "parameters":sum(p.numel() for p in model.parameters()), "batch16_train_step_seconds":times, "finite_loss":bool(torch.isfinite(loss)), "device":str(device)})
        del model, optimizer, batch, out, old
        if device.type == "mps": torch.mps.empty_cache()
    print(json.dumps({"grid":scenarios,"verification":result},indent=2))


if __name__ == "__main__":
    main()
