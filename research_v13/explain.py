"""SHAP (expected gradients) untuk 6 model final seed primer: 3 arsitektur x {baseline, fine-tuning}.

Menjelaskan logit kelas benar pada satu window tengah per pasien test. Ini deskripsi
model, bukan validasi klinis; atribusi bergantung pada background dan korelasi sadapan.
"""
import joblib
import numpy as np
import pandas as pd
import torch
from analysis_v2.explain_v2 import middle_segments, single_output_values
from src.artifacts import lock, read_json, seed_everything, sha256_file, write_json
from src.data import ECGDataset, LEADS
from src.reports import plotting
from research_v4.models import build_model
from .common import final_directory
from .data import load_pool
from .experiment import require_test_unlocked


class SegmentLogit(torch.nn.Module):
    def __init__(self, network, target):
        super().__init__()
        self.network, self.target = network, target

    def forward(self, inputs):
        return self.network.backbone(inputs)[1][:, self.target:self.target + 1]


def explain(study, device_name='cpu'):
    import shap
    require_test_unlocked(study)
    cfg, seed = study.protocol['shap'], study.protocol['primary_seed']
    X, development = load_pool(study)
    Xt, test = load_pool(study, 'test')
    scaler = joblib.load(study.run / 'scalers/final/scaler.joblib')
    background_rows = middle_segments(development, cfg['background_segments'], cfg['seed'])
    explained = middle_segments(test, cfg['explained_patients'], cfg['seed'], stratified=True)
    dev, held = ECGDataset(X, development.label, scaler=scaler), ECGDataset(Xt, test.label, scaler=scaler)
    device = torch.device(device_name)
    background = torch.stack([dev[int(i)][0] for i in background_rows.segment_index]).to(device)
    importance = []
    plt = plotting(study.run)
    for model in study.protocol['models']:
        for arm in ('baseline', 'finetune'):
            out = study.run / 'shap' / model / arm
            checkpoint = final_directory(study, model, arm, seed) / 'model.pth'
            with lock(out / '.lock'):
                network = build_model(model, 'mean', study.config['model'], {}).to(device).eval()
                network.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
                fastpath = torch.backends.mha.get_fastpath_enabled(); torch.backends.mha.set_fastpath_enabled(False)
                values = []
                try:
                    for i, row in explained.iterrows():
                        path = out / f'patient_{row.hospital_id}.npy'
                        example = held[int(row.segment_index)][0][None].to(device)
                        if path.exists():
                            value = np.load(path)
                        else:
                            seed_everything(cfg['seed'] + i)
                            explainer = shap.GradientExplainer(SegmentLogit(network, int(row.label)).eval(), background, batch_size=8)
                            raw = explainer.shap_values(example, nsamples=cfg['nsamples'], rseed=cfg['seed'] + i)
                            value = single_output_values(raw, tuple(example.shape)).astype('float32')
                            np.save(path, value)
                        values.append(value[0])
                        print(f'SHAP {model}/{arm}: {i + 1}/{len(explained)} pasien {row.hospital_id}', flush=True)
                finally:
                    torch.backends.mha.set_fastpath_enabled(fastpath)
                values = np.stack(values)
                np.save(out / 'shap_values.npy', values)
                lead = np.abs(values).mean(axis=(0, 2))
                importance += [{'model': model, 'arm': arm, 'lead': l, 'mean_abs_shap': float(v),
                                'share': float(v / lead.sum())} for l, v in zip(LEADS, lead)]
                for k, cls in enumerate(study.class_names):
                    mask = explained.label.to_numpy() == k
                    importance += [{'model': model, 'arm': arm, 'class': cls, 'lead': l, 'mean_abs_shap': float(v)}
                                   for l, v in zip(LEADS, np.abs(values[mask]).mean(axis=(0, 2)))]
                write_json(out / 'manifest.json', {'experiment_hash': study.signature, 'checkpoint_sha256': sha256_file(checkpoint),
                    'background_segment_index': background_rows.segment_index.astype(int).tolist(),
                    'explained': explained[['hospital_id', 'label', 'segment_index']].astype({'segment_index': int}).to_dict('records'),
                    'method': 'shap.GradientExplainer', 'output': 'true-class segment logit', 'clinical_validation': False})
                del network
    frame = pd.DataFrame(importance)
    frame.to_csv(study.run / 'shap/lead_importance.csv', index=False)
    overall = frame[frame['class'].isna()] if 'class' in frame else frame
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for row, arm in enumerate(('baseline', 'finetune')):
        for col, model in enumerate(study.protocol['models']):
            part = overall[(overall.model == model) & (overall.arm == arm)].set_index('lead').loc[LEADS[::-1]]
            axes[row, col].barh(part.index, part.share)
            axes[row, col].set_title(f'{model} — {arm}'); axes[row, col].set_xlabel('Proporsi mean |SHAP| per sadapan')
    fig.tight_layout(); fig.savefig(study.run / 'shap/lead_importance.png', dpi=180); plt.close(fig)
    print('SHAP selesai:', study.run / 'shap', flush=True)
