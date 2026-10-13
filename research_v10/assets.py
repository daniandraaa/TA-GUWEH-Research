"""Pinned official download; retain exact model/config without its optimizer."""
from pathlib import Path
import gc
import hashlib
import io
import json
import shutil
import psutil
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
REVISION = '584219ea492cdeef2e19ffbdf9c6ecc874ba427e'
FILENAME = 'mimic_iv_ecg_physionet_pretrained.pt'
SHA256 = '4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e'
ASSETS = Path.home() / 'Library/Caches/TA-GUWEH-ECGFM' / REVISION
MODEL_ONLY = 'ecg_fm_model_only.pt'


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def checkpoint_path(): return ASSETS / MODEL_ONLY


def download():
    ASSETS.mkdir(parents=True, exist_ok=True); target = checkpoint_path()
    if target.exists():
        marker = json.loads((ASSETS / 'model_only.json').read_text())
        if marker['source_sha256'] != SHA256 or marker['sha256'] != file_hash(target):
            raise RuntimeError('Compact official weights changed; preserve and investigate')
        print('Exact official model/config cached and verified', flush=True); return target
    response = requests.get(f'https://huggingface.co/api/models/wanglab/ecg-fm/tree/{REVISION}', timeout=30)
    response.raise_for_status(); meta = next(f for f in response.json() if f['path'] == FILENAME)
    size = int(meta['size'])
    if meta['lfs']['oid'] != SHA256: raise RuntimeError('Official checkpoint digest changed')
    # Preserve the 3-GiB disk reserve by buffering the verified upstream download
    # in RAM and retaining only its exact model/config, excluding optimizer moments.
    if shutil.disk_usage(ROOT).free < 3 * 1024**3 + 420 * 1024**2:
        raise RuntimeError('Need reserve 3 GiB plus 420 MiB for exact official model-only storage')
    if psutil.virtual_memory().available < 3 * size + 256 * 1024**2:
        raise RuntimeError('Need about 3.3 GiB available RAM to verify/convert the upstream checkpoint; close heavy applications and retry')
    url = f'https://huggingface.co/wanglab/ecg-fm/resolve/{REVISION}/{FILENAME}'
    buffer = io.BytesIO(); digest = hashlib.sha256(); total = announced = 0
    with requests.get(url, stream=True, timeout=(30, 90)) as source:
        source.raise_for_status()
        for block in source.iter_content(chunk_size=1024 * 1024):
            buffer.write(block); digest.update(block); total += len(block)
            if total // (128 * 1024**2) > announced:
                announced = total // (128 * 1024**2); print(f'Verify upstream {total / 1024**2:.0f} / {size / 1024**2:.0f} MiB', flush=True)
    if total != size or digest.hexdigest() != SHA256: raise RuntimeError('Official download size/hash mismatch')
    buffer.seek(0); checkpoint = torch.load(buffer, map_location='cpu', weights_only=True)
    cfg, state = checkpoint['cfg'], checkpoint['model']
    del checkpoint; buffer.close(); gc.collect()
    temporary = target.with_suffix('.tmp')
    torch.save({'cfg': cfg, 'model': state, 'source_sha256': SHA256}, temporary)
    copied = torch.load(temporary, map_location='cpu', weights_only=True, mmap=True)
    if copied['cfg'] != cfg or set(copied['model']) != set(state) or any(not torch.equal(t, copied['model'][k]) for k, t in state.items()):
        raise RuntimeError('Exact model/config conversion verification failed')
    converted_sha = file_hash(temporary)
    config_url = f'https://huggingface.co/wanglab/ecg-fm/raw/{REVISION}/mimic_iv_ecg_physionet_pretrained.yaml'
    config = requests.get(config_url, timeout=30); config.raise_for_status()
    (ASSETS / 'official_config.yaml').write_text(config.text)
    (ASSETS / 'download.json').write_text(json.dumps({'repo': 'wanglab/ecg-fm', 'revision': REVISION,
        'filename': FILENAME, 'sha256': SHA256, 'bytes': size, 'url': url, 'license': 'MIT'}, indent=2) + '\n')
    (ASSETS / 'model_only.json').write_text(json.dumps({'source_sha256': SHA256,
        'sha256': converted_sha, 'bytes': temporary.stat().st_size, 'all_model_tensors_bitwise_preserved': True,
        'embedded_config_preserved': True, 'removed': 'upstream optimizer/history only; no prior experiment files'}, indent=2) + '\n')
    temporary.replace(target)
    del cfg, state, copied; gc.collect()
    print(f'Exact official model/config verified in local cache: {target}', flush=True)
    return target


if __name__ == '__main__': download()
