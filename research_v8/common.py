"""Freeze V7 generators and a new paired classifier protocol."""
from dataclasses import dataclass
from pathlib import Path
import importlib.metadata
import json
import platform
import yaml
from src.artifacts import ROOT, canonical_hash, lock, read_json, sha256_file, write_json
from research_v4.common import disk_check
from research_v5.common import context as v5_context
from research_v6.common import verify_files, write_manifest, require_quality
from research_v7.common import context as v7_context

PACKAGE=Path(__file__).resolve().parent
DEFAULT_RUN='20261005_paired_classifier_augmentation_v8'


@dataclass
class Study:
    run: Path
    source: Path
    protocol: dict
    records: list
    folds: list
    choices: dict
    inherited: dict
    training: dict
    gan: object
    identity: dict

    @property
    def signature(self): return canonical_hash(self.identity)


def validate_protocol(p):
    expected={'models':['hierarchical','encoder'],'arms':['real_only','real_aug','gan_aug'],
        'folds':[1,2,3,4,5],'seeds':[42,43,44],'patient_batch_size':8,'segments_per_patient':4,
        'sampling_rate':2000,'segment_length':5000,'evaluation_segment_chunk':16,
        'augmentation':{'replacements_per_eligible_bag':1,'eligible_bag_windows':4,'short_bags':'unchanged',
            'real_source':'same_class_other_patient_uniform_gan_fit_patient_then_window',
            'gan_source':'unchanged_qc_passed_v7_64_windows_per_class_per_fold'},
        'epoch_selection':'real_only_gan_fit_vs_gan_reference_inner_macro_f1',
        'outer_budget':'same_inner_selected_epoch_for_all_three_arms',
        'outer_checkpoint':'fixed_last_epoch_no_outer_selection',
        'hyperparameters':'inherited_v4_selected_lr_pooling_no_new_grid',
        'model_initialization':'from_scratch_same_seed_across_arms',
        'final_training_enabled':False,'test_signals_enabled':False,
        'interpretation':'exploratory_development_ablation_prior_hyperparameter_selection_not_fully_nested'}
    if any(p.get(key)!=value for key,value in expected.items()) or p['minimum_free_gib']<0:
        raise ValueError('Only fixed development augmentation comparison is supported')


def context(run_id=DEFAULT_RUN,initialize=False):
    if not run_id or Path(run_id).name!=run_id or run_id in {'.','..'}: raise ValueError('One run-id required')
    p=yaml.safe_load((PACKAGE/'protocol.yaml').read_text());validate_protocol(p)
    gan=v7_context(p['source_v7_run'])
    source_files={}
    for split in gan.folds:
        f=split['fold']; require_quality(gan,f)
        for name in ('preparation.json','completed.json','generated.json','quality.json'):
            marker=gan.folder(f)/name;saved=read_json(marker)
            source_files[str(marker.relative_to(ROOT))]=sha256_file(marker)
            source_files.update({str((gan.folder(f)/n).relative_to(ROOT)):digest for n,digest in saved['files'].items()})
    native_audit=gan.run/'analysis_20261005_complete/audit.json'
    audit=read_json(native_audit)
    if (audit['experiment_hash']!=gan.signature or audit['folds_quality_passed']!=5
            or not audit['quality_native_recomputed'] or audit['failed_families']
            or audit['audit_code_sha256']!=sha256_file(ROOT/'analysis_v7/audit_completed.py')):
        raise RuntimeError('Complete verified native GAN V7 audit first')
    v5run=read_json(gan.baseline/'protocol.json')['protocol']['source_v5_run']
    old=v5_context(v5run)
    choices={model:old.choices[model] for model in p['models']}
    training={**old.training_cfg['training'],'scheduler_t_max':100}
    identity={'protocol':p,'source_v7_hash':gan.signature,'source_v7_files':source_files,
        'source_v7_native_audit_sha256':sha256_file(native_audit),'source_v5_hash':old.signature,
        'development_cache':gan.identity['development_cache'],'folds':gan.folds,
        'records_hash':canonical_hash(gan.records),'choices':choices,'training':training,
        'inherited':old.inherited,'new_model':old.training_cfg['new_model'],
        'outer_scalers':{str(q.relative_to(ROOT)):sha256_file(q) for split in gan.folds
            for q in [old.baseline/f'scalers/fold_{split["fold"]}/scaler.joblib',old.baseline/f'scalers/fold_{split["fold"]}/provenance.json']},
        'code':{q.name:sha256_file(q) for q in sorted(PACKAGE.glob('*.py'))},
        'runner_sha256':sha256_file(ROOT/'notebooks/_v8_runner.py'),
        'runtime':{'python':platform.python_version(),'platform':platform.platform(),
            **{n:importlib.metadata.version(n) for n in ('torch','numpy','pandas','scikit-learn','scipy','PyYAML','joblib','matplotlib')}},
        'historical_test_used':False,'prior_development_informed_hyperparameters':True,
        'fully_nested_estimate':False}
    study=Study(ROOT/'results/research_v8'/run_id,gan.source,p,gan.records,gan.folds,choices,old.inherited,training,gan,identity)
    expected=json.loads(json.dumps({**identity,'experiment_hash':study.signature}))
    path=study.run/'protocol.json'
    if not path.exists():
        if not initialize: raise RuntimeError('Jalankan notebook 41 / prepare dahulu')
        disk_check(study.run,p)
        with lock(study.run/'.initialize.lock'):
            if not path.exists():
                write_json(study.run/'folds.json',gan.folds);write_json(path,expected)
    if read_json(path)!=expected or read_json(study.run/'folds.json')!=json.loads(json.dumps(gan.folds)):
        raise RuntimeError('Kode/data/runtime/protokol V8 berubah. Pertahankan run; gunakan run-id baru.')
    return study


def directory(study,model,fold,seed,arm='inner_selection'):
    return study.run/'cases'/model/f'fold_{fold}'/f'seed_{seed}'/arm


def require_prepared(study):
    saved=read_json(study.run/'preparation.json');verify_files(study.run,saved)
    if saved['experiment_hash']!=study.signature: raise RuntimeError('Preparation changed')
    return saved


def checked_complete(study,model,fold,seed,arm='inner_selection'):
    folder=directory(study,model,fold,seed,arm)
    if not (folder/'completed.json').exists():
        raise RuntimeError(f'Lengkapi {model}/fold{fold}/seed{seed}/{arm} dahulu')
    saved=read_json(folder/'completed.json');verify_files(folder,saved)
    case=read_json(folder/'case.json')
    if (saved['experiment_hash']!=study.signature or case['experiment_hash']!=study.signature
            or (case['model'],case['fold'],case['seed'],case['arm'])!=(model,fold,seed,arm)
            or saved['case_hash']!=case['case_hash']
            or canonical_hash({k:v for k,v in case.items() if k!='case_hash'})!=case['case_hash']):
        raise RuntimeError('Completed case identity changed')
    return read_json(folder/'metrics.json')


def progress(study):
    inner=outer=0;missing=[]
    for m in study.protocol['models']:
        for f in study.protocol['folds']:
            for seed in study.protocol['seeds']:
                for arm in ('inner_selection',*study.protocol['arms']):
                    if (directory(study,m,f,seed,arm)/'completed.json').exists():
                        checked_complete(study,m,f,seed,arm)
                        if arm=='inner_selection':inner+=1
                        else:outer+=1
                    else:missing.append(f'{m}/fold{f}/seed{seed}/{arm}')
    expected=len(study.protocol['models'])*len(study.protocol['folds'])*len(study.protocol['seeds'])
    return {'experiment_hash':study.signature,'inner_selection_complete':inner,'inner_selection_expected':expected,
        'outer_comparison_complete':outer,'outer_comparison_expected':expected*3,
        'ready_for_comparison':inner==expected and outer==expected*3,'missing':missing,
        'final_enabled':False,'test_enabled':False}
