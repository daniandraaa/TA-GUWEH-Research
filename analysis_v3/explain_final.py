"""Manual expected-gradients SHAP for the CV-selected V3 primary checkpoint.

Explains one true-class segment logit per patient. No model training and no
claim of clinical interpretation or explanation of aggregated probabilities.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
import torch

from analysis_v2.explain_v2 import ClassLogit, middle_segments, single_output_values
from src.artifacts import ROOT, canonical_hash, lock, read_json, seed_everything, sha256_file, write_json
from src.data import ECGDataset, LEADS
from src.reports import plotting
from research_v3.common import DEFAULT_RUN, context, require_final_complete
from research_v3.data import load_pool, representation
from research_v3.models import build_model


def explain(run_id=DEFAULT_RUN, device_name='cpu', background_count=30,
            patient_count=15, nsamples=100, seed=20261002):
    import shap
    run, source, cfg, config, _, _ = context(run_id)
    require_final_complete(run, cfg)
    decision = read_json(run / 'selection.json')
    choice, primary_seed = decision['primary'], decision['primary_seed']
    name, arm = choice['model'], choice['arm']
    summaries = pd.read_csv(run / 'reports/cv_summary.csv')
    winners = [summaries[summaries.model == model].sort_values(
        ['f1_mean', 'f1_std', 'arm', 'scenario'], ascending=[False, True, True, True]).iloc[0]
        for model in cfg['models']]
    winner = sorted(winners, key=lambda r: (-r.f1_mean, r.f1_std, r.model))[0]
    if (name, arm, choice['scenario'], primary_seed) != (winner.model, winner.arm, winner.scenario, cfg['primary_seed']):
        raise RuntimeError('Primary selection differs from CV; run notebook 18 audit before SHAP.')
    if nsamples < 1 or background_count < 1 or patient_count < 3:
        raise ValueError('Invalid explanation counts')
    if device_name not in {'cpu', 'mps'}:
        raise ValueError('Use CPU or MPS')
    checkpoint = run / 'final' / name / f'seed_{primary_seed}/model.pth'
    scaler_path = run / 'scalers' / representation(arm) / 'final/scaler.joblib'
    final_case = read_json(checkpoint.parent / 'config.json')
    if final_case['model'] != name or final_case['arm'] != arm or final_case['scenario']['id'] != choice['scenario']:
        raise RuntimeError('SHAP selection does not match final checkpoint identity')
    data_owner = source if representation(arm) == 'window' else run / 'representations/r_candidate'
    X, development = load_pool(run, source, arm)
    Xt, historical = load_pool(run, source, arm, 'test')
    bg = middle_segments(development, background_count, seed)
    ex = middle_segments(historical, patient_count, seed, stratified=True)
    spec = {'run_id': run_id, 'experiment_hash': read_json(run / 'protocol.json')['experiment_hash'],
            'model': name, 'arm': arm, 'primary_seed': primary_seed,
            'checkpoint_sha256': sha256_file(checkpoint), 'scaler_sha256': sha256_file(scaler_path),
            'development_waveform_sha256': sha256_file(data_owner / 'data/development/X.npy'),
            'test_waveform_sha256': sha256_file(data_owner / 'data/test/X.npy'),
            'development_segments_sha256': sha256_file(data_owner / 'data/development/segments.csv'),
            'test_segments_sha256': sha256_file(data_owner / 'data/test/segments.csv'),
            'code_sha256': sha256_file(__file__), 'helper_sha256': sha256_file(ROOT / 'analysis_v2/explain_v2.py'),
            'background': bg.segment_index.astype(int).tolist(), 'explained': ex.segment_index.astype(int).tolist(),
            'nsamples': nsamples, 'seed': seed, 'device': device_name,
            'method': 'shap.GradientExplainer expected gradients', 'output': 'true-class segment logit',
            'shap_version': shap.__version__, 'torch_version': torch.__version__,
            'sampling_rate_hz': config['sampling_rate'], 'segment_samples': int(X.shape[-1])}
    signature = canonical_hash(spec)
    out = run / 'explain_primary_v3' / signature[:12]
    with lock(out / '.lock'):
        scaler = joblib.load(scaler_path)
        dev = ECGDataset(X, development.label, scaler=scaler)
        test = ECGDataset(Xt, historical.label, scaler=scaler)
        device = torch.device(device_name)
        background = torch.stack([dev[int(i)][0] for i in bg.segment_index]).to(device)
        seed_everything(seed)
        model = build_model(name, config['model']).to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        values = []
        try:
            for i, row in ex.iterrows():
                path = out / f'patient_{row.hospital_id}.npy'
                example = test[int(row.segment_index)][0][None].to(device)
                if path.exists():
                    value = single_output_values(np.load(path), tuple(example.shape))
                else:
                    wrapper = ClassLogit(model, int(row.label)).eval()
                    explainer = shap.GradientExplainer(wrapper, background, batch_size=8)
                    raw = explainer.shap_values(example, nsamples=nsamples, rseed=seed + i)
                    value = single_output_values(raw, tuple(example.shape)).astype('float32')
                    temporary = path.with_name(path.stem + '.tmp.npy')
                    np.save(temporary, value); temporary.replace(path)
                    del explainer
                values.append(value[0])
                print(f'SHAP V3 {name} seed {primary_seed}: {i + 1}/{len(ex)} patient {row.hospital_id}', flush=True)
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
        values = np.stack(values)
        np.save(out / 'shap_true_class_values.npy', values)
        bg.to_csv(out / 'background_patients.csv', index=False)
        ex.to_csv(out / 'explained_patients.csv', index=False)
        importance = pd.DataFrame({'lead': LEADS, 'mean_absolute_shap': np.abs(values).mean(axis=(0, 2))})
        importance.sort_values('mean_absolute_shap', ascending=False).to_csv(out / 'lead_importance.csv', index=False)
        plt = plotting(run)
        fig, ax = plt.subplots(figsize=(8, 5))
        importance.sort_values('mean_absolute_shap').plot.barh(x='lead', y='mean_absolute_shap', ax=ax, legend=False)
        ax.set_xlabel('Mean absolute SHAP: true-class segment logit')
        ax.set_title(f'CV-selected V3 {name}, seed {primary_seed}; {len(ex)} patients')
        fig.tight_layout(); fig.savefig(out / 'lead_importance.png', dpi=180); plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        time = np.arange(X.shape[-1]) / config['sampling_rate']
        for ax, (k, group) in zip(axes, ex.groupby('label')):
            i = int(group.index[0]); row = ex.iloc[i]
            ecg = test[int(row.segment_index)][0].numpy()
            ax.plot(time, ecg[1], lw=.7, color='black')
            twin = ax.twinx(); twin.plot(time, values[i, 1], lw=.7, color='tab:red')
            ax.set_title(f'{name}, patient {row.hospital_id}, {config["class_names"][int(k)]}, Lead II')
            ax.set_ylabel('Standardized denoised ECG'); twin.set_ylabel('SHAP logit')
        axes[-1].set_xlabel('Time (seconds; 2000 Hz)')
        fig.tight_layout(); fig.savefig(out / 'temporal_shap.png', dpi=180); plt.close(fig)
        write_json(out / 'manifest.json', {**spec, 'signature': signature,
                   'patient_probability_explained': False, 'clinical_validation': False,
                   'feature_dependence_caveat': 'ECG leads/time points correlated; background changes attributions',
                   'input': 'standardized HPF+DWT-denoised waveform'})
    print('SHAP V3 saved:', out, flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--background', type=int, default=30)
    parser.add_argument('--patients', type=int, default=15)
    parser.add_argument('--nsamples', type=int, default=100)
    args = parser.parse_args()
    explain(args.run_id, args.device, args.background, args.patients, args.nsamples)
