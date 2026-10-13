"""Manual endpoints: prepare, smoke, fit, generate, QC. No classifier or test path."""
import gc
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.artifacts import canonical_hash, cpu_weights, lock, read_json, save_torch, seed_everything, sha256_file, write_json
from research_v4.common import disk_check
from .common import require_prepared, require_quality, verify_files, write_manifest
from .data import BalancedPatients, global_scale, load_development, reference_rows
from .models import Discriminator, Generator
from .quality import collect_features, compare_features, features, native_nearest, numerical_screen
from .training import fit, optimizers, step


def save_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, index=False); temp.replace(path)


def prepare(study):
    X, table = load_development(study)
    with lock(study.run / '.prepare.lock'):
        for fold in study.folds:
            number = fold['fold']; folder = study.folder(number)
            if (folder / 'preparation.json').exists():
                require_prepared(study, number)
                print(f'PREPARE cached fold{number}', flush=True)
                continue
            disk_check(study.run, study.protocol)
            print(f'PREPARE fold{number}: inner fit/reference only', flush=True)
            rows = {name: reference_rows(table, fold[key], study.protocol['reference_windows_per_patient'])
                for name, key in (('fit', 'gan_fit_ids'), ('reference', 'gan_reference_ids'))}
            scale = global_scale(X, rows['fit'])
            files = []
            for name in ('fit', 'reference'):
                save_csv(folder / f'{name}_rows.csv', rows[name])
                frame, families = collect_features(X, rows[name], scale)
                save_csv(folder / f'{name}_features.csv', frame)
                files.extend((f'{name}_rows.csv', f'{name}_features.csv'))
            write_json(folder / 'feature_families.json', families)
            write_json(folder / 'amplitude_scale.json', {'scale': scale, 'units': 'unchanged native cache units',
                'fit_patient_ids': fold['gan_fit_ids'], 'reference_rows_hash': canonical_hash(rows['fit'].to_dict('records')),
                'method': study.protocol['amplitude_scale'], 'centering': False, 'clipping': False})
            files.extend(('feature_families.json', 'amplitude_scale.json'))
            write_manifest(folder / 'preparation.json', {'experiment_hash': study.signature, 'fold': number,
                'fit_patients': len(fold['gan_fit_ids']), 'reference_patients': len(fold['gan_reference_ids']),
                'outer_validation_patients': len(fold['outer_validation_ids']),
                'files': {name: sha256_file(folder / name) for name in files},
                'training_started': False, 'test_signals_read': False, 'outer_validation_signals_read': False})
    print(f'Prepared {len(study.folds)} folds. GAN optimizer steps=0. Run notebook33 smoke manually.', flush=True)


def networks(study, device):
    p = study.protocol['generator']
    return (Generator(p['latent_dim'], p['embedding_dim'], p['base_channels']).to(device), Discriminator().to(device))


def make_loader(study, fold, device):
    require_prepared(study, fold)
    X, table = load_development(study)
    scale = read_json(study.folder(fold) / 'amplitude_scale.json')['scale']
    cfg = study.protocol['training']; seed = study.protocol['gan_seed'] + 100 * fold
    dataset = BalancedPatients(X, table, study.folds[fold - 1]['gan_fit_ids'], scale,
        cfg['draws_per_fit_patient_per_epoch'], seed + 10000)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=0, shuffle=False,
        generator=torch.Generator().manual_seed(seed + 11000))
    return loader


def smoke(study, device):
    require_prepared(study, 1)
    path = study.run / 'smoke.json'
    with lock(study.run / '.training.lock'):
        if path.exists():
            require_smoke(study, device)
            print('SMOKE cached; no new disposable step', flush=True); return
        disk_check(study.run, study.protocol)
        seed_everything(142)
        loader = make_loader(study, 1, device)
        generator, discriminator = networks(study, device)
        opt_g, opt_d = optimizers(generator, discriminator, study.protocol['training'])
        real, labels, _ = next(iter(loader))
        started = time.monotonic()
        record = step(generator, discriminator, opt_g, opt_d, real.to(device), labels.to(device),
            torch.Generator().manual_seed(30142), study.protocol['training'])
        with torch.no_grad():
            fake = generator(torch.randn(len(real), generator.latent).to(device), labels.to(device))
        if fake.shape != real.shape or not torch.isfinite(fake).all(): raise RuntimeError('Smoke output shape/finite failure')
        write_manifest(path, {'experiment_hash': study.signature, 'device': str(device),
            'batch_shape': list(real.shape), 'd_disposable_steps': 1, 'g_disposable_steps': 1,
            'passed': True, 'seconds': time.monotonic() - started, 'diagnostics': record, 'files': {}})
        print(f'SMOKE passed on {device}; batch={list(real.shape)}; disposable G+D, no training checkpoint saved.', flush=True)
        del generator, discriminator, opt_g, opt_d, real, labels, fake, loader
        gc.collect()
        if device.type == 'mps': torch.mps.empty_cache()


def require_smoke(study, device):
    saved = read_json(study.run / 'smoke.json')
    verify_files(study.run, saved)
    if (saved['experiment_hash'] != study.signature or saved['device'] != str(device)
            or not saved['passed'] or saved['batch_shape'] != [study.protocol['training']['batch_size'], 12, 5000]):
        raise RuntimeError('Jalankan smoke notebook33 pada device yang sama')


def train(study, fold, device):
    require_smoke(study, device); require_prepared(study, fold)
    if fold != 1: require_quality(study, 1)
    folder = study.folder(fold)
    seed = study.protocol['gan_seed'] + 100 * fold
    case = {'experiment_hash': study.signature, 'fold': fold, 'device': str(device), 'effective_seed': seed,
        'fit_patient_ids': study.folds[fold - 1]['gan_fit_ids'],
        'reference_patient_ids': study.folds[fold - 1]['gan_reference_ids'],
        'checkpoint_selection': f'fixed_epoch{study.protocol["training"]["epochs"]}; no classifier/outer validation/test scores'}
    case_hash = canonical_hash(case)
    with lock(study.run / '.training.lock'):
        if (folder / 'case.json').exists():
            if read_json(folder / 'case.json') != {**case, 'case_hash': case_hash}: raise RuntimeError('GAN case/device changed')
        else: write_json(folder / 'case.json', {**case, 'case_hash': case_hash})
        if (folder / 'completed.json').exists():
            completed = read_json(folder / 'completed.json'); verify_files(folder, completed)
            if completed['experiment_hash'] != study.signature or completed['case_hash'] != case_hash:
                raise RuntimeError('Completed GAN identity changed')
            print(f'TRAIN cached complete fold{fold}', flush=True); return
        disk_check(study.run, study.protocol)
        seed_everything(seed)
        generator, discriminator = networks(study, device)
        loader = make_loader(study, fold, device)
        fit(generator, discriminator, loader, device, study.protocol['training'], folder, case_hash, seed)
        save_torch(folder / 'generator.pth', cpu_weights(generator))
        save_torch(folder / 'discriminator.pth', cpu_weights(discriminator))
        files = ('case.json', 'generator.pth', 'discriminator.pth', 'history.csv', 'last.pt')
        write_manifest(folder / 'completed.json', {'experiment_hash': study.signature, 'case_hash': case_hash,
            'fold': fold, 'epochs': study.protocol['training']['epochs'],
            'files': {n: sha256_file(folder / n) for n in files}, 'test_signals_read': False})
        print(f'TRAIN complete fold{fold}; checkpoint is fixed last epoch, not a selected classifier score.', flush=True)


def generate(study, fold, device):
    require_prepared(study, fold)
    folder = study.folder(fold)
    completed = read_json(folder / 'completed.json'); verify_files(folder, completed)
    if completed['experiment_hash'] != study.signature: raise RuntimeError('Wrong completed run')
    if read_json(folder / 'case.json')['device'] != str(device): raise RuntimeError('Generation must use GAN training device')
    with lock(study.run / '.training.lock'):
        marker = folder / 'generated.json'
        if marker.exists():
            previous = read_json(marker); verify_files(folder, previous)
            if previous['experiment_hash'] != study.signature or previous['generator_sha256'] != sha256_file(folder / 'generator.pth'):
                raise RuntimeError('Generated source changed')
            print(f'GENERATE cached fold{fold}', flush=True); return
        disk_check(study.run, study.protocol)
        generator, _ = networks(study, device)
        generator.load_state_dict(torch.load(folder / 'generator.pth', map_location='cpu', weights_only=True)); generator.eval()
        cfg = study.protocol['generation']; per_class = cfg['per_class']
        labels = np.repeat(np.arange(3), per_class)
        scale = read_json(folder / 'amplitude_scale.json')['scale']
        seed = study.protocol['gan_seed'] + 100 * fold + cfg['seed_offset']
        rng = torch.Generator().manual_seed(seed)
        samples = np.empty((len(labels), 12, 5000), dtype=np.float32)
        with torch.no_grad():
            for offset in range(0, len(labels), cfg['batch_size']):
                batch = labels[offset:offset + cfg['batch_size']]
                noise = torch.randn(len(batch), generator.latent, generator=rng).to(device)
                value = generator(noise, torch.as_tensor(batch, device=device)).cpu().numpy()
                samples[offset:offset + len(batch)] = value * scale
        # Keep even bad samples for diagnosis; numerical QC subsequently rejects them.
        target = folder / 'synthetic.npy'; temp = target.with_suffix('.npy.tmp')
        with temp.open('wb') as stream: np.save(stream, samples)
        temp.replace(target)
        save_csv(folder / 'synthetic_labels.csv', pd.DataFrame({'sample_index': np.arange(len(labels)), 'label': labels}))
        write_manifest(marker, {'experiment_hash': study.signature, 'fold': fold, 'device': str(device), 'seed': seed,
            'generator_sha256': sha256_file(folder / 'generator.pth'), 'shape': list(samples.shape),
            'native_sampling_rate': 2000, 'units': 'unchanged native cache units; no clipping',
            'files': {n: sha256_file(folder / n) for n in ('synthetic.npy', 'synthetic_labels.csv')}})
        print(f'GENERATE fold{fold}: {len(labels)} windows, {samples.nbytes / 1024**2:.1f} MiB', flush=True)


def qc(study, fold):
    require_prepared(study, fold)
    folder = study.folder(fold)
    with lock(study.run / '.analysis.lock'):
        complete = read_json(folder / 'completed.json'); verify_files(folder, complete)
        generated = read_json(folder / 'generated.json'); verify_files(folder, generated)
        if (complete['experiment_hash'] != study.signature or generated['experiment_hash'] != study.signature
                or generated['generator_sha256'] != sha256_file(folder / 'generator.pth')):
            raise RuntimeError('GAN provenance mismatch')
        path = folder / 'quality.json'
        if path.exists():
            previous = read_json(path); verify_files(folder, previous)
            if previous['experiment_hash'] != study.signature or previous['generated_manifest_sha256'] != sha256_file(folder / 'generated.json'):
                raise RuntimeError('Quality source changed')
            print(f'QC cached fold{fold}: passed={previous["passed"]}', flush=True); return
        X, table = load_development(study)
        scale = read_json(folder / 'amplitude_scale.json')['scale']
        samples = np.load(folder / 'synthetic.npy', mmap_mode='r')
        labels = pd.read_csv(folder / 'synthetic_labels.csv').label.to_numpy()
        settings = study.protocol['quality_gate']
        print(f'QC fold{fold}: native copy/diversity checks against inner fit only...', flush=True)
        numeric, neighbours = numerical_screen(samples, labels, X, table,
            study.folds[fold - 1]['gan_fit_ids'], scale, settings)
        failed, files = [], []
        if neighbours is not None:
            save_csv(folder / 'synthetic_nearest_fit.csv', neighbours); files.append('synthetic_nearest_fit.csv')
        if numeric['finite']:
            frames = [features(np.asarray(samples[o:o+16], dtype=np.float64) / scale)[0] for o in range(0, len(samples), 16)]
            fake = pd.concat(frames, ignore_index=True)
            fake.insert(0, 'label', labels); fake.insert(1, 'weight', 1.)
            save_csv(folder / 'synthetic_features.csv', fake); files.append('synthetic_features.csv')
            a = pd.read_csv(folder / 'fit_features.csv'); b = pd.read_csv(folder / 'reference_features.csv')
            distances, failed = compare_features(a, b, fake, read_json(folder / 'feature_families.json'), settings)
            save_csv(folder / 'feature_distances.csv', distances); files.append('feature_distances.csv')
            rows = pd.read_csv(folder / 'reference_rows.csv', dtype={'hospital_id': str})
            reference_values = np.asarray(X[rows.segment_index.to_numpy()], dtype=np.float64) / scale
            print(f'QC fold{fold}: inner reference-to-fit copy-distance baseline...', flush=True)
            nearest_reference = native_nearest(reference_values, X, table, study.folds[fold-1]['gan_fit_ids'], scale, rows.label)
            nearest_reference['reference_hospital_id'] = rows.hospital_id.to_numpy()
            nearest_reference['weight'] = rows.weight.to_numpy()
            save_csv(folder / 'reference_nearest_fit.csv', nearest_reference); files.append('reference_nearest_fit.csv')
        passed = bool(numeric['passed'] and numeric['finite'] and not failed)
        write_manifest(path, {'experiment_hash': study.signature, 'fold': fold,
            'generated_manifest_sha256': sha256_file(folder / 'generated.json'),
            'passed': passed, 'role': settings['role'], 'numerical_screen': numeric, 'failed_families': failed,
            'synthetic_condition_is_patient_diagnosis_not_verified_PVC': True,
            'clinical_validity_established': False, 'classifier_benefit_established': False,
            'test_signals_read': False, 'outer_validation_signals_read': False,
            'files': {n: sha256_file(folder / n) for n in files}})
        print(f'QC fold{fold}: passed={passed}; numeric_errors={numeric["errors"]}; failed_families={failed}', flush=True)
        print('Next: notebook35 visual review; continue other folds only if numerical gate passes.', flush=True)
