"""Compact candidate cache, unchanged cohort, deferred historical test access."""
import numpy as np
import pandas as pd
from src.artifacts import ROOT, lock, read_json, sha256_file, write_json
from src.data import LEADS, load_pool as original_load_pool
from research_v2.data import denoise
from .candidates import detect_candidates, extract_windows
from .common import disk_check, require_final_complete
from .sampling import contribution_table


def representation(arm):
    return 'r_candidate' if arm.startswith('r_candidate_') else 'window'


def load_pool(run, source, arm, pool='development'):
    if pool not in {'development', 'test'}:
        raise ValueError(pool)
    cfg = read_json(run / 'protocol.json').get('protocol')
    if pool == 'test' and cfg is not None:
        require_final_complete(run, cfg)
    X, segments = original_load_pool(source if representation(arm) == 'window' else run / 'representations/r_candidate', pool)
    records_path = run / 'data/patients.json'
    if records_path.exists():
        expected = {r['hospital_id']: r['label'] for r in read_json(records_path) if r['pool'] == pool}
        if set(segments.hospital_id) != set(expected) or len(X) != len(segments):
            raise ValueError('Pool does not cover the unchanged patient cohort')
        if any(int(row.label) != expected[row.hospital_id] for row in segments.itertuples()):
            raise ValueError('Pool patient labels changed')
    return X, segments


def prepare(run, source, cfg, config, v2cfg, pool='development'):
    if pool not in {'development', 'test'}:
        raise ValueError(pool)
    if pool == 'test':
        require_final_complete(run, cfg)
    records = [r for r in read_json(run / 'data/patients.json') if r['pool'] == pool]
    directory = run / 'representations/r_candidate/data' / pool
    with lock(directory / '.lock'):
        if (directory / 'manifest.json').exists():
            if not (directory / 'X.npy').exists() or not (directory / 'segments.csv').exists():
                raise RuntimeError('Incomplete candidate cache')
            return directory
        cache = directory / 'patients'
        cache.mkdir(parents=True, exist_ok=True)
        rows, qa_rows, assets, shards = [], [], {}, []
        for i, r in enumerate(records, 1):
            hid = r['hospital_id']
            raw_path = ROOT / v2cfg['preprocessing']['input'] / f'{hid}.csv'
            digest = sha256_file(raw_path)
            path, marker = cache / f'{hid}.npy', cache / f'{hid}.json'
            if path.exists() and marker.exists():
                item = read_json(marker)
                if item['raw_sha256'] != digest or item['array_sha256'] != sha256_file(path):
                    raise RuntimeError(f'Candidate shard changed: {hid}')
            else:
                disk_check(run, cfg, 16 * 1024**2)
                raw = pd.read_csv(raw_path, usecols=LEADS)[LEADS].to_numpy()
                centers, qa = detect_candidates(raw, cfg['candidate'])
                clean = denoise(raw, config['sampling_rate'], v2cfg['preprocessing'])
                values, crops = extract_windows(clean, centers, cfg['candidate'])
                temporary = cache / f'{hid}.tmp.npy'
                np.save(temporary, values)
                temporary.replace(path)
                item = {'raw_sha256': digest, 'array_sha256': sha256_file(path),
                        'qa': qa, 'segments': crops}
                write_json(marker, item)
            rows.extend({'hospital_id': hid, 'label': r['label'], **row} for row in item['segments'])
            qa_rows.append({'hospital_id': hid, 'sublocation': r['sublocation'],
                            **item['qa'], 'used_fallback': bool(item['segments'][0]['is_fallback'])})
            assets[hid] = digest
            shards.append(path)
            print(f'CANDIDATES {pool} {i}/{len(records)}: {len(item["segments"])} crops', flush=True)
        length = round(cfg['candidate']['fs'] * (cfg['candidate']['before_seconds'] + cfg['candidate']['after_seconds']))
        disk_check(run, cfg, len(rows) * 12 * length * 4)
        temporary = directory / 'X.tmp.npy'
        X = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32', shape=(len(rows), 12, length))
        offset = 0
        for p in shards:
            values = np.load(p, mmap_mode='r')
            X[offset:offset + len(values)] = values
            offset += len(values)
        X.flush()
        del X
        temporary.replace(directory / 'X.npy')
        table = pd.DataFrame(rows)
        table.to_csv(directory / 'segments.csv', index=False)
        pd.DataFrame(qa_rows).to_csv(directory / 'candidate_quality.csv', index=False)
        contribution_table(table).to_csv(directory / 'patient_contributions.csv', index=False)
        write_json(directory / 'manifest.json', {'n_patients': len(records), 'n_segments': len(rows),
                   'sampling_rate': config['sampling_rate'], 'candidate_spec': cfg['candidate'],
                   'signal_sha256': assets, 'pvc_verified': False,
                   'input': 'HPF+DWT-denoised waveform; centers from bandpassed raw signal'})
        # Shards are retained for restart/integrity; no user artifacts removed.
    return directory
