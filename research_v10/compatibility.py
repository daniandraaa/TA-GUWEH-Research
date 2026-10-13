"""Safe checkpoint configuration, complete encoder loading, synthetic CPU parity."""
from pathlib import Path
import gc
import json
import yaml
import torch
from src.artifacts import ROOT, read_json, sha256_file, write_json
from .assets import ASSETS, FILENAME, SHA256, REVISION, file_hash, checkpoint_path
from .model import config, load_encoder, canonical_encoder_state, official_unmasked_forward

SPEC = ROOT / 'assets/ecg_fm' / REVISION / 'inspection.json'


def numerical_code():
    package = Path(__file__).parent
    return {str(q.relative_to(package)): sha256_file(q) for q in
        [package / 'model.py', package / 'data.py', *sorted((package / '_official').glob('*'))] if q.is_file()}


def inspect():
    path = checkpoint_path()
    if not path.exists(): raise RuntimeError('Jalankan endpoint download / notebook 51 dahulu')
    compact = read_json(ASSETS / 'model_only.json')
    if compact['source_sha256'] != SHA256 or file_hash(path) != compact['sha256']:
        raise RuntimeError('Official model-only digest mismatch')
    if SPEC.exists():
        saved = read_json(SPEC)
        if saved['numerical_code'] != numerical_code(): raise RuntimeError('CPU compatibility code changed; use a new inspected experiment')
        print('CPU compatibility cached and verified', flush=True); return saved
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(checkpoint.get('cfg'), dict): raise RuntimeError('Need embedded official model configuration')
    embedded = checkpoint['cfg']['model']; cfg = config(embedded)
    state, unused = canonical_encoder_state(checkpoint['model'])
    if embedded.get('quantize_input', False): raise RuntimeError('Input quantizer is unsupported')
    if not all(torch.isfinite(t).all() for t in state.values()): raise RuntimeError('Nonfinite official encoder weights')
    observed_layers = len({int(k.split('.')[2]) for k in state if k.startswith('encoder.layers.')})
    if observed_layers != cfg.encoder_layers or state['encoder.layers.0.self_attn.q_proj.weight'].shape != (cfg.encoder_embed_dim,) * 2:
        raise RuntimeError('Embedded architecture differs from tensors')
    all_params = sum(t.numel() for t in checkpoint['model'].values())
    encoder_params = sum(t.numel() for t in state.values())
    del checkpoint, state; gc.collect()
    model = load_encoder(path, embedded); comparisons = []
    with torch.no_grad():
        for length in [1250, 2500]:
            x = torch.randn(1, 12, length, generator=torch.Generator().manual_seed(103 + length))
            actual = model(x); reference = official_unmasked_forward(model, x)
            split = model.finish(model.until_last(x))
            error = float((actual - reference).abs().max()); split_error = float((actual - split).abs().max())
            if not torch.allclose(actual, reference, rtol=1e-6, atol=1e-7) or not torch.equal(actual, split):
                raise RuntimeError('CPU unmasked-forward/tail-split parity failed')
            comparisons.append({'samples': length, 'shape': list(actual.shape), 'reference_max_abs_error': error,
                'tail_split_max_abs_error': split_error, 'finite': bool(torch.isfinite(actual).all())})
            print(f'CPU PARITY {length}: official forward error={error:g}; tail split error={split_error:g}', flush=True)
    public_cfg = yaml.safe_load((ASSETS / 'official_config.yaml').read_text())['model']
    result = {'checkpoint_sha256': SHA256, 'checkpoint_revision': REVISION,
        'embedded_model_config': embedded, 'encoder_model_config': vars(cfg), 'all_pretraining_parameters': all_params,
        'encoder_parameters': encoder_params, 'pretraining_only_keys_omitted': unused,
        'public_yaml_disagreements': {k: {'public_yaml': public_cfg.get(k), 'embedded': embedded.get(k)}
            for k in ('encoder_layers', 'encoder_embed_dim', 'encoder_ffn_embed_dim', 'encoder_attention_heads')
            if public_cfg.get(k) != embedded.get(k)},
        'source_revision': 'f8f0ff1c788a82c2059cb452cd5462898867489e', 'numerical_code': numerical_code(),
        'model_only_sha256': compact['sha256'], 'optimizer_removed_model_tensors_unchanged': True,
        'strict_encoder_loading': True, 'unmasked_cpu_parity': comparisons,
        'no_environment_packages_installed': True, 'genuine_waveform_inference': False, 'training_performed': False,
        'test_signals_read': False}
    write_json(SPEC, result)
    del model; gc.collect()
    return result


def require_inspection():
    result = read_json(SPEC)
    if result['checkpoint_sha256'] != SHA256 or result['numerical_code'] != numerical_code():
        raise RuntimeError('Compatibility identity changed')
    if not checkpoint_path().exists() or file_hash(checkpoint_path()) != result['model_only_sha256']:
        raise RuntimeError('Official checkpoint missing/changed; restore pinned download')
    return result


if __name__ == '__main__': inspect()
