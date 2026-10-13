"""Manual inner-only selection, fresh fixed-epoch outer refits and tail pilot."""
import gc
import numpy as np
import pandas as pd
import torch
from src.artifacts import (canonical_hash, choose_device, read_json, seed_everything,
    sha256_file, write_json)
from src.evaluation import prediction_table, report
from research_v6.common import verify_files, write_manifest
from .assets import checkpoint_path
from .common import case_directory, grid, select_best, split_for, storage_check
from .data import fit_feature_scale
from .features import open_features
from .model import load_encoder, TailClassifier
from .training import FeatureClassifier, Patients, fit, predict, release_completed_resume


def settings_for(study, arm, row, seed, fold, epochs=None, head_lr=None):
    p = study.protocol; tail = arm == 'pretrained_tail'
    maximum = p['tail_epochs' if tail else 'head_epochs']
    return {'learning_rate': row['learning_rate'], 'head_learning_rate': head_lr or row['learning_rate'],
        'batch_size': row['batch_size'], 'epochs': maximum if epochs is None else epochs,
        'scheduler_t_max': maximum, 'minimum_lr':0., 'patience':p['tail_patience' if tail else 'head_patience'],
        'weight_decay':p['weight_decay'], 'gradient_clip':p['gradient_clip'], 'effective_seed': seed+100*fold}


def make_datasets(study, arm, train_ids, evaluation_ids, patients, windows, arrays):
    ids = patients.hospital_id.tolist(); lookup = {hid:i for i,hid in enumerate(ids)}
    train_indices = [lookup[x] for x in sorted(train_ids)]
    features = arrays['pretrained_probe' if arm == 'pretrained_tail' else arm]
    mean, scale = fit_feature_scale(features, train_indices)
    def dataset(selected):
        selected = sorted(selected); indices = [lookup[x] for x in selected]
        labels = patients.iloc[indices].label.to_numpy()
        if arm == 'pretrained_tail':
            groups = [np.flatnonzero(windows.hospital_id.to_numpy() == x) for x in selected]
            return Patients(selected, labels, tokens=arrays['tokens'], groups=groups)
        return Patients(selected, labels, features=features[indices])
    return dataset(train_ids), dataset(evaluation_ids), mean, scale


def completed_case(folder, case):
    if not (folder/'completed.json').exists(): return None
    saved = read_json(folder/'completed.json'); verify_files(folder,saved)
    fit_done = read_json(folder/'fit_finished.json'); verify_files(folder,fit_done)
    if saved['case_hash'] != canonical_hash(case) or read_json(folder/'case.json') != case:
        raise RuntimeError('Completed case identity changed')
    result = read_json(folder/'metrics.json')
    if result['inner_selection'] != case['inner_selection']: raise RuntimeError('Inner/outer role changed')
    return result


def run_case(study, arm, fold, seed, row, device, bundle, base_encoder=None, choice=None, head_lr=None):
    split = split_for(study,fold); inner = choice is None
    train_ids = split['gan_fit_ids'] if inner else sorted(split['gan_fit_ids']+split['gan_reference_ids'])
    evaluation_ids = split['gan_reference_ids'] if inner else split['outer_validation_ids']
    patients, windows, arrays, cache_hash = bundle
    train, evaluation, mean, scale = make_datasets(study,arm,train_ids,evaluation_ids,patients,windows,arrays)
    settings = settings_for(study,arm,row,seed,fold,None if inner else choice['selected_epoch'],head_lr)
    case = {'experiment_hash':study.signature,'arm':arm,'fold':fold,'seed':seed,'scenario':row,
        'settings':settings,'device':str(device),'train_ids':sorted(train_ids),'evaluation_ids':sorted(evaluation_ids),
        'inner_selection':inner,'scaler_fit_ids':sorted(train_ids),
        'scaler_hash':canonical_hash({'mean':mean.tolist(),'scale':scale.tolist()}),'feature_cache_manifest':cache_hash,
        'selection_hash':None if inner else canonical_hash(choice),'test_signals_read':False}
    folder = case_directory(study,arm,fold,seed,row['scenario'] if inner else None)
    result = completed_case(folder,case)
    if result is not None:
        print(f'SKIP verified {arm}/fold{fold}/{folder.name}',flush=True); return result
    storage_check(study,220*1024**2 if arm == 'pretrained_tail' else 16*1024**2)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'case.json').exists() and read_json(folder/'case.json') != case: raise RuntimeError('Resume case identity changed')
    write_json(folder/'case.json',case); case_hash = canonical_hash(case)
    seed_everything(settings['effective_seed'])
    model = (TailClassifier(base_encoder,mean,scale) if arm == 'pretrained_tail' else FeatureClassifier(mean,scale)).to(device)
    resume_root = study.cache/'resume'
    result = fit(model,train,evaluation if inner else None,device,settings,folder,case_hash,resume_root)
    model.load_state_dict(torch.load(folder/'selected.pth',map_location='cpu',weights_only=True))
    ids, truth, probabilities = predict(model,evaluation,device)
    prediction_table(ids,truth,probabilities,study.classes).to_csv(folder/'predictions.csv',index=False)
    statistics = report(truth,probabilities,study.classes)
    result = {**result,**statistics['metrics'],'patients':len(ids),'arm':arm,'fold':fold,'seed':seed,
        'scenario':row['scenario'],'learning_rate':row['learning_rate'],'batch_size':row['batch_size'],
        'confusion_matrix':statistics['confusion_matrix'],'classification_report':statistics['classification_report'],
        'outer_scores_used_for_selection':False,'test_signals_read':False}
    if inner and not np.isclose(result['f1'],result['best_inner_f1'],rtol=0,atol=1e-12):
        raise RuntimeError('Selected inner checkpoint does not reproduce its selection score')
    write_json(folder/'metrics.json',result)
    files = ['case.json','fit_finished.json','selected.pth','history.csv','fit_result.json','predictions.csv','metrics.json']
    write_manifest(folder/'completed.json',{'case_hash':case_hash,'experiment_hash':study.signature,
        'files':{n:sha256_file(folder/n) for n in files},'role':'inner_selection' if inner else 'fixed_refit_outer_evaluation'})
    release_completed_resume(resume_root,case_hash)
    del model; gc.collect()
    if device.type == 'mps': torch.mps.empty_cache()
    return result


def choice_path(study,arm,fold): return study.run/'choices'/arm/f'fold_{fold}.json'


def select_inner(study,arm,fold,device,bundle,base_encoder=None,head_lr=None):
    seed = study.protocol['cv_seed']; rows = []; sources = {}
    for row in grid(study.protocol,arm):
        print(f'INNER {arm} fold={fold} {row}',flush=True)
        result = run_case(study,arm,fold,seed,row,device,bundle,base_encoder,head_lr=head_lr)
        rows.append(result); folder = case_directory(study,arm,fold,seed,row['scenario'])
        sources[str(folder.relative_to(study.run)/'completed.json')] = sha256_file(folder/'completed.json')
    best = select_best(rows)
    choice = {'experiment_hash':study.signature,'arm':arm,'fold':fold,'seed':seed,
        'scenario':best['scenario'],'learning_rate':best['learning_rate'],'batch_size':best['batch_size'],
        'selected_epoch':best['selected_epoch'],'inner_f1':best['f1'],'source_completed_sha256':sources,
        'rule':study.protocol['selection'],'head_learning_rate':head_lr or best['learning_rate'],
        'outer_validation_scores_used':False}
    path = choice_path(study,arm,fold)
    if path.exists() and read_json(path) != choice: raise RuntimeError('Frozen inner choice changed')
    write_json(path,choice); print(f'SELECTED {arm} fold{fold}: {choice["scenario"]}, epoch {choice["selected_epoch"]}',flush=True)
    return choice


def benchmark(study,device,folds,arms,seeds):
    if any(f not in study.protocol['folds'] for f in folds) or any(s not in study.protocol['outer_seeds'] for s in seeds):
        raise ValueError('Unsupported folds/seeds')
    if any(a not in [*study.protocol['probe_arms'],'pretrained_tail'] for a in arms): raise ValueError('Unsupported arm')
    if 'pretrained_tail' in arms and (folds != [1] or seeds != [42]):
        raise ValueError('Partial fine-tuning is only fold1 seed42 pilot; no full-CV tail claim')
    bundle = open_features(study); encoder = None
    try:
        for arm in arms:
            if arm == 'pretrained_tail':
                path = choice_path(study,'pretrained_probe',1)
                if not path.exists(): raise RuntimeError('Lengkapi pretrained_probe fold1 dahulu')
                head_lr = read_json(path)['learning_rate']
                encoder = load_encoder(checkpoint_path(),study.inspection['embedded_model_config'])
            else: head_lr = None
            for fold in folds:
                choice = select_inner(study,arm,fold,device,bundle,encoder,head_lr)
                row = next(r for r in grid(study.protocol,arm) if r['scenario'] == choice['scenario'])
                for seed in seeds:
                    print(f'OUTER fixed refit {arm} fold={fold} seed={seed}; epochs={choice["selected_epoch"]}',flush=True)
                    run_case(study,arm,fold,seed,row,device,bundle,encoder,choice,head_lr)
    finally:
        del encoder,bundle; gc.collect()


def smoke(study,device):
    """User-triggered synthetic device forward/backward; no genuine ECG read."""
    seed_everything(999); encoder = load_encoder(checkpoint_path(),study.inspection['embedded_model_config']).to(device)
    with torch.no_grad():
        before = encoder.until_last(torch.randn(4,12,1250,device=device)); out = encoder.finish(before)
        if out.shape != (4,78,768) or not torch.isfinite(out).all(): raise RuntimeError('Synthetic encoder smoke failed')
    head = FeatureClassifier(np.zeros(768,dtype='float32'),np.ones(768,dtype='float32')).to(device)
    tail = TailClassifier(encoder,np.zeros(768,dtype='float32'),np.ones(768,dtype='float32')).to(device)
    optimizer = torch.optim.AdamW(tail.parameters(),lr=1e-6)
    loss = torch.nn.functional.cross_entropy(tail(before.detach()).unsqueeze(0),torch.tensor([1],device=device))
    loss.backward()
    if not any(p.grad is not None and p.grad.abs().max()>0 for p in tail.tail.parameters()): raise RuntimeError('Tail has no gradient')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in tail.parameters()): raise RuntimeError('Nonfinite gradient')
    optimizer.step()
    with torch.no_grad():
        if not torch.isfinite(head(out.mean((0,1)))).all(): raise RuntimeError('Probe logits nonfinite')
    write_json(study.run/f'synthetic_smoke_{device.type}.json',{'experiment_hash':study.signature,'device':str(device),
        'encoder_shape':list(out.shape),'tail_backward_finite':True,'synthetic_only':True,
        'genuine_waveform_inference':False,'genuine_classifier_training':False,'test_signals_read':False})
    print(f'Synthetic {device} forward/backward PASS',flush=True)
    del encoder,head,tail,optimizer,before,out,loss; gc.collect()
    if device.type == 'mps': torch.mps.empty_cache()
