"""GAN final untuk model fine-tuning final: resep V7 identik, fit hanya pada development.

Pasien development dibagi fit/reference (80/20, stratified) hanya untuk QC mutu,
sama seperti GAN per fold V7. Data test tidak pernah dibaca.
"""
from collections import Counter
import gc
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.artifacts import canonical_hash, cpu_weights, lock, read_json, save_torch, seed_everything, sha256_file, write_json
from research_v4.common import disk_check
from research_v6.common import verify_files, write_manifest
from research_v6.data import BalancedPatients, global_scale, reference_rows
from research_v6.models import Discriminator, Generator
from research_v6.quality import collect_features, compare_features, features, native_nearest, numerical_screen
from research_v7.fidelity import FidelityLoss, build_bank
from research_v7.training import fit
from .data import load_pool


def folder(study):
    return study.run / 'gan_final'


def split(study):
    labels = {r['hospital_id']: r['label'] for r in study.records if r['pool'] == 'development'}
    ids = sorted(labels)
    cfg = study.protocol['final_gan']
    fit_ids, reference = train_test_split(ids, test_size=cfg['inner_reference_fraction'],
        stratify=[labels[x] for x in ids], random_state=cfg['inner_split_seed'])
    fit_ids, reference = sorted(fit_ids), sorted(reference)
    if set(fit_ids) & set(reference) or set(fit_ids + reference) != set(ids):
        raise RuntimeError('Split fit/reference GAN final tidak valid')
    return {'gan_fit_ids': fit_ids, 'gan_reference_ids': reference,
            'fit_class_patients': dict(Counter(labels[x] for x in fit_ids)),
            'reference_class_patients': dict(Counter(labels[x] for x in reference))}


def prepare(study):
    out = folder(study); g = study.gan.protocol
    with lock(out / '.prepare.lock'):
        if (out / 'preparation.json').exists():
            saved = read_json(out / 'preparation.json'); verify_files(out, saved)
            if saved['experiment_hash'] != study.signature:
                raise RuntimeError('Persiapan GAN final berasal dari run lain')
            print('GAN FINAL prepare cached', flush=True); return
        disk_check(study.run, study.protocol)
        X, table = load_pool(study)
        parts = split(study)
        write_json(out / 'split.json', parts)
        rows = {name: reference_rows(table, parts[key], g['reference_windows_per_patient'])
                for name, key in (('fit', 'gan_fit_ids'), ('reference', 'gan_reference_ids'))}
        scale = global_scale(X, rows['fit'])
        files = ['split.json']
        for name in ('fit', 'reference'):
            rows[name].to_csv(out / f'{name}_rows.csv', index=False)
            frame, families = collect_features(X, rows[name], scale)
            frame.to_csv(out / f'{name}_features.csv', index=False)
            files += [f'{name}_rows.csv', f'{name}_features.csv']
        write_json(out / 'feature_families.json', families)
        write_json(out / 'amplitude_scale.json', {'scale': scale, 'method': g['amplitude_scale'],
                                                   'fit_patient_ids': parts['gan_fit_ids']})
        bank = build_bank(X, rows['fit'], scale, g['fidelity'])
        with (out / 'fit_target_bank.npz').open('wb') as stream:
            np.savez(stream, **bank)
        files += ['feature_families.json', 'amplitude_scale.json', 'fit_target_bank.npz']
        write_manifest(out / 'preparation.json', {'experiment_hash': study.signature,
            'fit_patients': len(parts['gan_fit_ids']), 'reference_patients': len(parts['gan_reference_ids']),
            'files': {n: sha256_file(out / n) for n in files}, 'test_signals_read': False})
    print(f'GAN FINAL prepared: fit {len(parts["gan_fit_ids"])} / reference {len(parts["gan_reference_ids"])} pasien development', flush=True)


def _prepared(study):
    out = folder(study)
    saved = read_json(out / 'preparation.json'); verify_files(out, saved)
    if saved['experiment_hash'] != study.signature:
        raise RuntimeError('Jalankan prepare GAN final pada run ini')
    return out, read_json(out / 'split.json'), read_json(out / 'amplitude_scale.json')['scale']


def networks(study, device):
    p = study.gan.protocol['generator']
    return Generator(p['latent_dim'], p['embedding_dim'], p['base_channels']).to(device), Discriminator().to(device)


def train(study, device):
    out, parts, scale = _prepared(study)
    g = study.gan.protocol; seed = study.protocol['final_gan']['seed']
    case = {'experiment_hash': study.signature, 'device': str(device), 'seed': seed,
            'fit_patient_ids': parts['gan_fit_ids'], 'recipe': 'v7_identical',
            'v7_protocol_hash': canonical_hash(g), 'fit_target_bank_sha256': sha256_file(out / 'fit_target_bank.npz'),
            'checkpoint_selection': f'fixed_epoch{g["training"]["epochs"]}'}
    case_hash = canonical_hash(case)
    with lock(out / '.training.lock'):
        if (out / 'case.json').exists() and read_json(out / 'case.json') != {**case, 'case_hash': case_hash}:
            raise RuntimeError('Identitas/device GAN final berubah; lanjutkan pada device yang sama')
        write_json(out / 'case.json', {**case, 'case_hash': case_hash})
        if (out / 'completed.json').exists():
            saved = read_json(out / 'completed.json'); verify_files(out, saved)
            print('GAN FINAL train cached', flush=True); return
        disk_check(study.run, study.protocol)
        X, table = load_pool(study)
        seed_everything(seed)
        generator, discriminator = networks(study, device)
        cfg = g['training']
        dataset = BalancedPatients(X, table, parts['gan_fit_ids'], scale, cfg['draws_per_fit_patient_per_epoch'], seed + 10000)
        loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(seed + 11000))
        with np.load(out / 'fit_target_bank.npz', allow_pickle=False) as data:
            bank = {k: data[k].copy() for k in data.files}
        print(f'GAN FINAL train {cfg["epochs"]} epoch pada {len(parts["gan_fit_ids"])} pasien fit', flush=True)
        fit(generator, discriminator, loader, device, cfg, out, case_hash, seed, FidelityLoss(bank, g['fidelity']))
        save_torch(out / 'generator.pth', cpu_weights(generator))
        save_torch(out / 'discriminator.pth', cpu_weights(discriminator))
        names = ('case.json', 'generator.pth', 'discriminator.pth', 'history.csv', 'last.pt')
        write_manifest(out / 'completed.json', {'experiment_hash': study.signature, 'case_hash': case_hash,
            'files': {n: sha256_file(out / n) for n in names}, 'test_signals_read': False})
        del generator, discriminator, loader; gc.collect()
        if device.type == 'mps': torch.mps.empty_cache()


def generate(study, device):
    out, _, scale = _prepared(study)
    saved = read_json(out / 'completed.json'); verify_files(out, saved)
    if read_json(out / 'case.json')['device'] != str(device):
        raise RuntimeError('Generate harus memakai device training GAN')
    with lock(out / '.training.lock'):
        if (out / 'generated.json').exists():
            verify_files(out, read_json(out / 'generated.json')); print('GAN FINAL generate cached', flush=True); return
        generator, _ = networks(study, device)
        generator.load_state_dict(torch.load(out / 'generator.pth', map_location='cpu', weights_only=True)); generator.eval()
        cfg = study.gan.protocol['generation']
        labels = np.repeat(np.arange(3), cfg['per_class'])
        rng = torch.Generator().manual_seed(study.protocol['final_gan']['seed'] + cfg['seed_offset'])
        samples = np.empty((len(labels), 12, study.protocol['segment_length']), dtype=np.float32)
        with torch.no_grad():
            for o in range(0, len(labels), cfg['batch_size']):
                batch = labels[o:o + cfg['batch_size']]
                noise = torch.randn(len(batch), generator.latent, generator=rng).to(device)
                samples[o:o + len(batch)] = generator(noise, torch.as_tensor(batch, device=device)).cpu().numpy() * scale
        with (out / 'synthetic.npy').open('wb') as stream:
            np.save(stream, samples)
        pd.DataFrame({'sample_index': np.arange(len(labels)), 'label': labels}).to_csv(out / 'synthetic_labels.csv', index=False)
        write_manifest(out / 'generated.json', {'experiment_hash': study.signature,
            'generator_sha256': sha256_file(out / 'generator.pth'), 'shape': list(samples.shape),
            'files': {n: sha256_file(out / n) for n in ('synthetic.npy', 'synthetic_labels.csv')}})
    print(f'GAN FINAL generate: {len(labels)} window sintetis', flush=True)


def qc(study):
    out, parts, scale = _prepared(study)
    generated = read_json(out / 'generated.json'); verify_files(out, generated)
    with lock(out / '.analysis.lock'):
        if (out / 'quality.json').exists():
            saved = read_json(out / 'quality.json'); verify_files(out, saved)
            print(f'GAN FINAL QC cached: passed={saved["passed"]}', flush=True); return saved
        X, table = load_pool(study)
        samples = np.load(out / 'synthetic.npy', mmap_mode='r')
        labels = pd.read_csv(out / 'synthetic_labels.csv').label.to_numpy()
        settings = study.gan.protocol['quality_gate']
        numeric, neighbours = numerical_screen(samples, labels, X, table, parts['gan_fit_ids'], scale, settings)
        failed, files = [], []
        if neighbours is not None:
            neighbours.to_csv(out / 'synthetic_nearest_fit.csv', index=False); files.append('synthetic_nearest_fit.csv')
        if numeric['finite']:
            frames = [features(np.asarray(samples[o:o + 16], dtype=np.float64) / scale)[0] for o in range(0, len(samples), 16)]
            fake = pd.concat(frames, ignore_index=True); fake.insert(0, 'label', labels); fake.insert(1, 'weight', 1.)
            fake.to_csv(out / 'synthetic_features.csv', index=False); files.append('synthetic_features.csv')
            distances, failed = compare_features(pd.read_csv(out / 'fit_features.csv'), pd.read_csv(out / 'reference_features.csv'),
                                                 fake, read_json(out / 'feature_families.json'), settings)
            distances.to_csv(out / 'feature_distances.csv', index=False); files.append('feature_distances.csv')
            rows = pd.read_csv(out / 'reference_rows.csv', dtype={'hospital_id': str})
            reference = np.asarray(X[rows.segment_index.to_numpy()], dtype=np.float64) / scale
            nearest = native_nearest(reference, X, table, parts['gan_fit_ids'], scale, rows.label)
            nearest.to_csv(out / 'reference_nearest_fit.csv', index=False); files.append('reference_nearest_fit.csv')
        passed = bool(numeric['passed'] and numeric['finite'] and not failed)
        result = {'experiment_hash': study.signature, 'passed': passed, 'numerical_screen': numeric,
                  'failed_families': failed, 'role': settings['role'], 'clinical_validity_established': False,
                  'test_signals_read': False, 'files': {n: sha256_file(out / n) for n in files}}
        write_manifest(out / 'quality.json', result)
    print(f'GAN FINAL QC: passed={passed}; failed_families={failed}; errors={numeric["errors"]}', flush=True)
    if not passed:
        print('QC GAGAL: final fine-tuning terkunci. Jangan ubah ambang; laporkan dan diskusikan dengan pembimbing.', flush=True)
    return result


def require_quality(study):
    out = folder(study)
    saved = read_json(out / 'quality.json'); verify_files(out, saved)
    generated = read_json(out / 'generated.json'); verify_files(out, generated)
    if saved['experiment_hash'] != study.signature or not saved['passed']:
        raise RuntimeError('GAN final belum lolos QC; final fine-tuning tidak boleh dijalankan')
    return out
