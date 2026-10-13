"""Tahap final, test dan analisis untuk PatchTST biner V14 (lihat PRAREGISTRASI_V14_FINAL_TEST).

Model, skenario dan jumlah epoch ditetapkan dari CV development V14 sebelum test dibaca:
PatchTST, fine-tuning G02 (LR 1e-4, batch 32, GAN final V13 yang lolos QC), epoch = median best epoch CV.
"""
import argparse
import gc
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from src.artifacts import ROOT, canonical_hash, lock, read_json, seed_everything, sha256_file, write_json
from src.evaluation import prediction_table
from src.experiment import bind_case
from research_v3.sampling import patient_class_weights
from research_v4.data import scaler_for
from research_v4.models import build_model
from research_v4.training import fit, predict
from research_v13 import gan_final
from research_v13.analysis import bootstrap_difference, mcnemar_exact
from research_v13.common import test_directory as v13_test_directory
from research_v13.data import PatientBags, load_pool as v13_load_pool
from research_v14.common import context as v14_context
from research_v14.experiment import binary_table, load_development, make_loader, report, settings_for
from analysis_v14.calibration import calibration, net_benefit, THRESHOLDS
from analysis_v14.clinical import paired_auc_bootstrap

MODEL, ARM = 'patchtst', 'finetune'
SEEDS = [42, 43, 44, 45, 46]
PREREGISTRATION = ROOT / 'docs/revisi/PRAREGISTRASI_V14_FINAL_TEST_20261008.md'
OUT = ROOT / 'results/research_v14/20261008_binary_patchtst_final_test_v14'


def require_preregistered():
    text = PREREGISTRATION.read_text()
    for p in sorted((ROOT / 'research_v14_final').glob('*.py')):
        if f'{p.name}: {sha256_file(p)}' not in text:
            raise RuntimeError(f'Kode {p.name} berubah setelah pra-registrasi')


def frozen_choice(study):
    decision = read_json(study.run / 'reports/decision.json')
    if decision['experiment_hash'] != study.signature or decision['test_read'] or decision['choices'][MODEL]['approach'] != 'v14_binary':
        raise RuntimeError('Keputusan CV V14 tidak memilih PatchTST biner')
    cases = pd.read_csv(study.run / 'reports/cv_paired_cases.csv')
    epochs = int(round(cases[(cases.model == MODEL) & (cases.arm == ARM)].best_epoch_v14.median()))
    return study.scenarios[MODEL][ARM], epochs, canonical_hash(decision)


def final(study, device):
    scenario, epochs, decision_hash = frozen_choice(study)
    X, table = load_development(study)
    ids = table.hospital_id.unique().tolist(); train = np.arange(len(X))
    synthetic = gan_final.require_quality(study.v13)
    s = {**settings_for(study, scenario), 'epochs': epochs}
    identity_run = {'v14_cv_hash': study.signature, 'decision_hash': decision_hash, 'model': MODEL, 'scenario': scenario,
                    'epochs': epochs, 'seeds': SEEDS, 'development_cache': study.identity['development_cache'],
                    'code': {q.name: sha256_file(q) for q in sorted((ROOT / 'research_v14_final').glob('*.py'))}}
    if (OUT / 'protocol.json').exists() and read_json(OUT / 'protocol.json') != json.loads(json.dumps(identity_run)):
        raise RuntimeError('Identitas run final V14 berubah; gunakan folder baru')
    write_json(OUT / 'protocol.json', identity_run)
    for seed in SEEDS:
        directory = OUT / 'final' / f'seed_{seed}'
        identity = {'v14_cv_hash': study.signature, 'decision_hash': decision_hash, 'device': str(device), 'model': MODEL,
                    'scenario': scenario, 'seed': seed, 'settings': s, 'train_patient_ids': sorted(ids),
                    'synthetic_generated_sha256': sha256_file(synthetic / 'generated.json')}
        with lock(directory / '.lock'):
            bind_case(directory, identity)
            if (directory / 'metrics.json').exists():
                continue
            scaler = scaler_for(OUT, 'final', X, table, train, ids)
            seed_everything(seed)
            loader = make_loader(study, X, table, ids, scaler, scenario, seed, synthetic)
            network = build_model(MODEL, 'mean', study.model_config, {}).to(device)
            seed_everything(seed + 20000)
            print(f'V14 FINAL {MODEL}/{scenario["id"]} seed={seed} epoch={epochs} LR={s["learning_rate"]} batch={s["batch_size"]}', flush=True)
            try:
                fit(network, loader, None, device, s, directory, study.class_names, patient_class_weights(table, 2),
                    study.protocol['evaluation_segment_chunk'])
            finally:
                del network, loader; gc.collect()
                if device.type == 'mps': torch.mps.empty_cache()


def test(study, device):
    require_preregistered()
    if not all((OUT / 'final' / f'seed_{s}' / 'metrics.json').is_file() for s in SEEDS):
        raise RuntimeError('Test terkunci: 5 model final belum lengkap')
    X, table = v13_load_pool(study.v13, 'test'); table = binary_table(study, table)
    scaler_path = OUT / 'scalers/final/scaler.joblib'
    dataset = PatientBags(X, table, table.hospital_id.unique(), joblib.load(scaler_path))
    for seed in SEEDS:
        trained = OUT / 'final' / f'seed_{seed}'; directory = OUT / 'test' / f'seed_{seed}'
        if sha256_file(trained / 'model.pth') != read_json(trained / 'metrics.json')['model_sha256']:
            raise RuntimeError('Bobot final berubah')
        with lock(directory / '.lock'):
            bind_case(directory, {'final_model_sha256': sha256_file(trained / 'model.pth'), 'scaler_sha256': sha256_file(scaler_path)})
            if (directory / 'metrics.json').exists():
                continue
            network = build_model(MODEL, 'mean', study.model_config, {}).to(device)
            network.load_state_dict(torch.load(trained / 'model.pth', map_location='cpu', weights_only=True))
            ids, truth, probability, _ = predict(network, dataset, device, study.protocol['evaluation_segment_chunk'])
            prediction_table(ids, truth, probability, study.class_names).to_csv(directory / 'patient_predictions.csv', index=False)
            result = {**report(truth, probability, study.class_names), 'n_patients': len(ids),
                      'test_role': 'historical_holdout_previously_observed'}
            write_json(directory / 'metrics.json', result)
            print(f'TEST V14 {MODEL} seed={seed} F1={result["metrics"]["f1"]:.4f} AUC={result["metrics"]["auc"]:.4f}', flush=True)
            del network; gc.collect()


def analyze(study):
    require_preregistered()
    frames = [pd.read_csv(OUT / 'test' / f'seed_{s}' / 'patient_predictions.csv', dtype={'hospital_id': str})
              .sort_values('hospital_id').set_index('hospital_id') for s in SEEDS]
    y = frames[0].true_label.to_numpy()
    p14 = np.mean([f.probability_RVOT.to_numpy() for f in frames], axis=0)
    old = [pd.read_csv(v13_test_directory(study.v13, MODEL, ARM, s) / 'patient_predictions.csv', dtype={'hospital_id': str})
           .sort_values('hospital_id').set_index('hospital_id') for s in SEEDS]
    if any(not o.index.equals(frames[0].index) for o in old):
        raise RuntimeError('Pasien test V13/V14 tidak identik')
    p13 = np.mean([(o.probability_RVOT_cusp + o.probability_RVOT_septal_freewall_other).to_numpy() for o in old], axis=0)
    rows = []
    for name, per_seed, ens in (('v13_patchtst_3kelas_digabung', [(o.probability_RVOT_cusp + o.probability_RVOT_septal_freewall_other).to_numpy() for o in old], p13),
                                ('v14_patchtst_biner', [f.probability_RVOT.to_numpy() for f in frames], p14)):
        seeds = [report(y, np.column_stack([1 - p, p]), study.class_names)['metrics'] for p in per_seed]
        ens_m = report(y, np.column_stack([1 - ens, ens]), study.class_names)
        lvot = ens_m['classification_report']['LVOT']['recall']
        rows.append({'model': name, 'f1_mean_5seed': np.mean([m['f1'] for m in seeds]), 'f1_sd_5seed': np.std([m['f1'] for m in seeds], ddof=1),
                     'auc_mean_5seed': np.mean([m['auc'] for m in seeds]), 'f1_ensemble': ens_m['metrics']['f1'],
                     'auc_ensemble': ens_m['metrics']['auc'], 'accuracy_ensemble': ens_m['metrics']['accuracy'],
                     'sensitivity_LVOT_ensemble': lvot, 'sensitivity_RVOT_ensemble': ens_m['classification_report']['RVOT']['recall'],
                     **{f'calib_{k}': v for k, v in calibration((y == 0).astype(int), 1 - ens).items()}})
    table = pd.DataFrame(rows)
    pair = {**mcnemar_exact((p13 > .5).astype(int) == y, (p14 > .5).astype(int) == y),
            **bootstrap_difference(y, np.column_stack([1 - p13, p13]), np.column_stack([1 - p14, p14]), 5000, 20261008)}
    rules = pd.read_csv(ROOT / 'results/research_v14/clinical_criteria_v14/patient_measures_and_calls.csv',
                        dtype={'hospital_id': str}).set_index('hospital_id').reindex(frames[0].index)
    clinical = pd.DataFrame([{'rule': r, **paired_auc_bootstrap(y, rules[f'{r}_score'].to_numpy(), p14)} for r in ('tz', 'v2s_v3r', 'v2tr')])
    lv = (y == 0).astype(int)
    dca = pd.DataFrame([{'threshold': t, 'v14_patchtst': net_benefit(lv, ((1 - p14) >= t).astype(int), t),
                         'v13_patchtst': net_benefit(lv, ((1 - p13) >= t).astype(int), t),
                         'kriteria_V2S_V3R': net_benefit(lv, (rules.v2s_v3r_call == 0).astype(int).to_numpy(), t),
                         'semua_LVOT': net_benefit(lv, np.ones(len(y), int), t)} for t in THRESHOLDS])
    out = OUT / 'reports'; out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'test_v13_vs_v14_patchtst.csv', index=False); clinical.to_csv(out / 'auc_v14_minus_rule.csv', index=False)
    dca.to_csv(out / 'decision_curve.csv', index=False); write_json(out / 'paired_test_v13_vs_v14.json', pair)
    pd.set_option('display.width', 220)
    print(table.round(3).to_string(index=False)); print(pair); print(clinical.round(3).to_string(index=False))
    print(dca[dca.threshold.isin([.1, .2, .3, .4])].round(3).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['final', 'test', 'analyze'])
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = parser.parse_args()
    study = v14_context()
    from src.artifacts import choose_device
    if args.stage == 'final': final(study, choose_device(args.device))
    elif args.stage == 'test': test(study, choose_device(args.device))
    else: analyze(study)
