"""Scoped V4 factory patch and separate provenance, never rewrite frozen cases."""
from datetime import datetime, timezone
from functools import wraps
import inspect
from pathlib import Path
from torch import nn
from src.artifacts import canonical_hash, lock, read_json, sha256_file, write_json
from research_v4 import experiment
from research_v4.common import cv_directory, require_final_complete
from research_v4.models import build_model as original_build_model
from . import VERSION
from .pooling import ExactAdaptiveAvgPool1d

ORIGINAL_FIT = experiment.fit_case
ORIGINAL_EVALUATE = experiment.evaluate
_INSTALLED = False
RECORD = 'backend_compatibility.json'


def build_model(name, pooling, inherited_config, new_config):
    network = original_build_model(name, pooling, inherited_config, new_config)
    if name == 'lead_time':
        layer = network.backbone.cnn[-1]
        if not isinstance(layer, nn.AdaptiveAvgPool1d) or layer.output_size != network.backbone.bins:
            raise RuntimeError('Unexpected frozen LeadTime pooling architecture')
        network.backbone.cnn[-1] = ExactAdaptiveAvgPool1d(layer.output_size)
    return network


def patch_identity(run):
    package = Path(__file__).resolve().parent
    return {
        'version': VERSION,
        'base_experiment_hash': read_json(Path(run) / 'protocol.json')['experiment_hash'],
        'code_sha256': {p.name: sha256_file(p) for p in sorted(package.glob('*.py'))},
        'scope': 'lead_time; CPU and MPS; training and inference',
        'operation': 'adaptive-average floor/ceil bins implemented with slice/mean/stack',
        'parameters_and_state_dict_unchanged': True,
        'input_padding_cropping_resampling': False,
    }


def _validate_record(path, identity):
    record = read_json(path)
    if record.get('identity') != identity or record.get('compatibility_hash') != canonical_hash(identity):
        raise RuntimeError(f'Backend compatibility changed: {path}. Preserve this patch for this run.')
    return record


def audit_run(run, create=False):
    run = Path(run)
    identity = patch_identity(run)
    path = run / RECORD
    with lock(run / '.backend_compatibility.lock'):
        if path.exists():
            return _validate_record(path, identity)
        if not create:
            raise RuntimeError(f'Missing backend compatibility provenance: {path}')
        # Existing LeadTime progress must never be silently mixed with this patch.
        for phase in ('diagnostic', 'cv', 'stability', 'final', 'test'):
            for p in (run / phase).rglob('*'):
                if 'lead_time' in p.parts and p.name in {'metrics.json', 'last.pt', 'model.pth', 'best.pth'}:
                    raise RuntimeError(f'Unbound LeadTime result already exists: {p}; use a separate run')
        preserved = {
            str(p.relative_to(run)): sha256_file(p)
            for phase in ('diagnostic', 'cv')
            for p in (run / phase).rglob('*')
            if p.is_file() and p.name in {'config.json', 'metrics.json'}
        }
        record = {
            'identity': identity, 'compatibility_hash': canonical_hash(identity),
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'reason': 'MPS adaptive pooling rejects non-divisible length 157 to 16 bins',
            'preserved_artifact_sha256_at_activation': preserved,
            'test_used_to_design_patch': False,
        }
        write_json(path, record)
        print(f'Backend compatibility activated: {record["compatibility_hash"]}', flush=True)
        return record


def case_directory(arguments):
    run, model, phase = Path(arguments['run']), arguments['model'], arguments['phase']
    fold, seed = arguments['fold'], arguments['seed']
    if phase == 'cv':
        return cv_directory(run, model, arguments['pooling'], arguments['scenario']['id'], fold, seed)
    if phase == 'final':
        return run / 'final' / model / f'seed_{seed}'
    if phase == 'stability':
        return run / phase / model / f'fold_{fold}' / f'seed_{seed}'
    raise ValueError('Compatibility patch supports LeadTime CV, stability and final only')


def bind_case(run, directory):
    record = audit_run(run, create=True)
    identity = {
        'run_compatibility_hash': record['compatibility_hash'],
        'case_path': str(directory.relative_to(run)),
    }
    with lock(directory / '.backend_compatibility.lock'):
        path = directory / RECORD
        if path.exists():
            _validate_record(path, identity)
        else:
            if any((directory / name).exists() for name in ('metrics.json', 'last.pt', 'model.pth', 'best.pth')):
                raise RuntimeError(f'Unbound LeadTime progress: {directory}')
            write_json(path, {'identity': identity, 'compatibility_hash': canonical_hash(identity)})


@wraps(ORIGINAL_FIT)
def fit_case(*args, **kwargs):
    bound = inspect.signature(ORIGINAL_FIT).bind(*args, **kwargs)
    bound.apply_defaults()
    arguments = bound.arguments
    if arguments['model'] == 'lead_time':
        run = Path(arguments['run'])
        bind_case(run, case_directory(arguments))
    return ORIGINAL_FIT(*args, **kwargs)


@wraps(ORIGINAL_EVALUATE)
def evaluate(run, source, cfg, config, device):
    # Maintain the original readiness gate before any access to test signals.
    require_final_complete(run, cfg)
    run = Path(run)
    record = audit_run(run)
    for seed in cfg['final_seeds']:
        directory = run / 'final/lead_time' / f'seed_{seed}'
        identity = {'run_compatibility_hash': record['compatibility_hash'],
                    'case_path': str(directory.relative_to(run))}
        _validate_record(directory / RECORD, identity)
    return ORIGINAL_EVALUATE(run, source, cfg, config, device)


def install():
    global _INSTALLED
    if not _INSTALLED:
        # Only these V4 aliases are replaced. No global torch or site-package patch.
        experiment.build_model = build_model
        experiment.fit_case = fit_case
        experiment.evaluate = evaluate
        _INSTALLED = True
