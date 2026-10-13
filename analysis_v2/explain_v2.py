"""Manual patient-balanced SHAP for completed V2, without modifying V2 code.

Explains one deterministic segment per selected patient and a single class
logit. This is a segment explanation, not an explanation of the aggregated
patient probability or a clinically validated ECG interpretation.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
from src.artifacts import canonical_hash, lock, read_json, seed_everything, sha256_file, write_json
from src.data import ECGDataset, LEADS, load_pool
from src.models import build_model
from src.reports import plotting
from research_v2.common import context, DEFAULT_RUN


class ClassLogit(torch.nn.Module):
    def __init__(self, model, target):
        super().__init__()
        self.model, self.target = model, target

    def forward(self, inputs):
        return self.model(inputs)[:, self.target:self.target + 1]


def middle_segments(table, count, seed, stratified=False):
    rng = np.random.default_rng(seed)
    records = table.reset_index(names='segment_index').groupby('hospital_id', sort=True)
    representatives = pd.DataFrame([g.iloc[len(g) // 2].to_dict() for _, g in records])
    if stratified:
        if count < representatives.label.nunique():
            raise ValueError('At least one explained patient per class is required')
        allocations = {int(k): count // 3 + (int(k) < count % 3) for k in range(3)}
        parts = [g.iloc[rng.choice(len(g), min(len(g), allocations[int(k)]), replace=False)]
                 for k, g in representatives.groupby('label', sort=True)]
        chosen = pd.concat(parts)
    else:
        chosen = representatives.iloc[rng.choice(len(representatives), min(len(representatives), count), replace=False)]
    return chosen.sort_values('hospital_id').reset_index(drop=True)


def single_output_values(values, shape):
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError('Expected one class output')
        values = values[0]
    values = np.asarray(values)
    if values.shape == (*shape, 1):
        values = values[..., 0]
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f'Invalid SHAP shape/values: {values.shape}, expected {shape}')
    return values


def explain(run_id=DEFAULT_RUN, device_name='cpu', background_count=30, patient_count=15,
            nsamples=100, seed=20261001):
    import shap
    run, _, cfg, config, _ = context(run_id)
    choice = read_json(run / 'selection.json')
    name, primary_seed = choice['primary']['model'], choice['primary_seed']
    checkpoint = run / 'final' / name / f'seed_{primary_seed}/model.pth'
    if not (run / 'test' / name / f'seed_{primary_seed}/metrics.json').exists():
        raise RuntimeError('V2 primary final/test must be complete')
    if nsamples < 1 or background_count < 1 or patient_count < 3:
        raise ValueError('Invalid explanation sample counts')
    if device_name not in {'cpu', 'mps'}:
        raise ValueError('Use cpu or mps')
    X, development = load_pool(run)
    Xt, test = load_pool(run, 'test')
    background_rows = middle_segments(development, background_count, seed)
    explained_rows = middle_segments(test, patient_count, seed, stratified=True)
    spec = {'model': name, 'primary_seed': primary_seed, 'checkpoint_sha256': sha256_file(checkpoint),
            'code_sha256': sha256_file(__file__),
            'background': background_rows.segment_index.astype(int).tolist(),
            'explained': explained_rows.segment_index.astype(int).tolist(),
            'nsamples': nsamples, 'seed': seed, 'device': device_name,
            'method': 'shap.GradientExplainer expected gradients', 'output': 'true-class segment logit',
            'shap_version': shap.__version__, 'torch_version': torch.__version__}
    signature = canonical_hash(spec)
    out = run / 'explain_patient_v2' / signature[:12]
    with lock(out / '.lock'):
        scaler = joblib.load(run / 'folds/final/scaler.joblib')
        train_dataset, test_dataset = ECGDataset(X, development.label, scaler=scaler), ECGDataset(Xt, test.label, scaler=scaler)
        device = torch.device(device_name)
        background = torch.stack([train_dataset[int(i)][0] for i in background_rows.segment_index]).to(device)
        seed_everything(seed)
        model = build_model(name, config['model']).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            values = []
            for i, row in explained_rows.iterrows():
                path = out / f'patient_{row.hospital_id}.npy'
                example = test_dataset[int(row.segment_index)][0][None].to(device)
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
                print(f'SHAP {i+1}/{len(explained_rows)} patient {row.hospital_id}', flush=True)
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
        values = np.stack(values)
        np.save(out / 'shap_true_class_values.npy', values)
        explained_rows.to_csv(out / 'explained_patients.csv', index=False)
        background_rows.to_csv(out / 'background_patients.csv', index=False)
        importance = pd.DataFrame({'lead': LEADS, 'mean_absolute_shap': np.abs(values).mean(axis=(0, 2))})
        importance.sort_values('mean_absolute_shap', ascending=False).to_csv(out / 'lead_importance.csv', index=False)
        plt = plotting(run)
        fig, ax = plt.subplots(figsize=(8, 5))
        importance.sort_values('mean_absolute_shap').plot.barh(x='lead', y='mean_absolute_shap', ax=ax, legend=False)
        ax.set_xlabel('Mean absolute SHAP, true-class segment logit')
        fig.tight_layout(); fig.savefig(out / 'lead_importance.png', dpi=180); plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        time = np.arange(config['segment_length']) / config['sampling_rate']
        for ax, (k, group) in zip(axes, explained_rows.groupby('label')):
            i = int(group.index[0]); row = explained_rows.iloc[i]
            ecg = test_dataset[int(row.segment_index)][0].numpy()
            ax.plot(time, ecg[1], lw=.7, color='black')
            twin = ax.twinx(); twin.plot(time, values[i, 1], lw=.7, color='tab:red')
            ax.set_title(f'Patient {row.hospital_id}, {config["class_names"][int(k)]}, Lead II')
            ax.set_ylabel('Standardized denoised ECG'); twin.set_ylabel('SHAP logit')
        axes[-1].set_xlabel('Time (seconds; 2000 Hz)')
        fig.tight_layout(); fig.savefig(out / 'temporal_shap.png', dpi=180); plt.close(fig)
        write_json(out / 'manifest.json', {**spec, 'signature': signature,
                   'patient_probability_explained': False, 'clinical_validation': False,
                   'feature_dependence_caveat': 'ECG leads/time points correlated; background changes attributions',
                   'input': 'standardized HPF+DWT-denoised waveform, not DWT coefficients'})
    print(f'SHAP saved: {out}', flush=True)
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
