"""Prepare without classifier steps; manual smoke, selector, refit and outer score."""
from collections import Counter
import gc
import shutil
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.artifacts import ROOT,canonical_hash,lock,read_json,seed_everything,sha256_file,write_json
from src.evaluation import prediction_table,report
from research_v3.sampling import patient_class_weights
from research_v4.common import disk_check
from research_v4.data import PatientBags,scaler_for
from research_v4.models import build_model
from research_v4.training import predict,class_diagnostics
from .common import directory,require_prepared,checked_complete,verify_files,write_manifest
from .data import AugmentedBags,collate,load_development
from .training import fit


def prepare(study):
    X,table=load_development(study)
    with lock(study.run/'.prepare.lock'):
        if (study.run/'preparation.json').exists():
            require_prepared(study);print('PREPARE cached; no classifier steps',flush=True);return
        files={};inventory=[]
        for split in study.folds:
            f=split['fold'];ids=split['gan_fit_ids']
            indices=np.flatnonzero(table.hospital_id.isin(ids).to_numpy())
            scaler_for(study.run,f'inner_{f}',X,table,indices,ids)
            for name in ('scaler.joblib','provenance.json'):
                source=[PathName for PathName in study.identity['outer_scalers'] if f'/fold_{f}/' in PathName and PathName.endswith('/'+name)]
                if len(source)!=1:raise RuntimeError('Missing frozen outer-training scaler')
                origin=ROOT/source[0];target=study.run/f'scalers/outer_{f}'/name;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and sha256_file(target)!=sha256_file(origin):raise RuntimeError('Copied scaler changed')
                if not target.exists():shutil.copy2(origin,target)
            outer=sorted(split['gan_fit_ids']+split['gan_reference_ids'])
            for stage in ('inner','outer'):
                for name in ('scaler.joblib','provenance.json'):
                    path=study.run/f'scalers/{stage}_{f}'/name;files[str(path.relative_to(study.run))]=sha256_file(path)
            counts=table[table.hospital_id.isin(outer)].groupby('hospital_id').size()
            inventory.append({'fold':f,'inner_fit_patients':len(ids),'inner_validation_patients':len(split['gan_reference_ids']),
                'outer_training_patients':len(outer),'outer_validation_patients':len(split['outer_validation_ids']),
                'outer_eligible_k4_patients':int(counts.ge(4).sum()),'outer_short_bag_patients':int(counts.lt(4).sum())})
        pd.DataFrame(inventory).to_csv(study.run/'fold_inventory.csv',index=False)
        files['fold_inventory.csv']=sha256_file(study.run/'fold_inventory.csv')
        expected=len(study.protocol['models'])*len(study.protocol['folds'])*len(study.protocol['seeds'])
        write_manifest(study.run/'preparation.json',{'experiment_hash':study.signature,'files':files,
            'inner_fits_expected':expected,'outer_fits_expected':expected*3,'classifier_optimizer_steps':0,
            'test_signals_read':False,'outer_validation_signals_read':False,'new_waveform_cache':False})
    print(f'Prepared {len(study.folds)} inner/outer-training scalers; {expected} inner selectors + {expected*3} paired fits. Classifier steps=0.',flush=True)


def setup(study,model,fold,seed,arm,inner=False):
    X,table=load_development(study);split=study.folds[fold-1]
    train_ids=split['gan_fit_ids'] if inner else sorted(split['gan_fit_ids']+split['gan_reference_ids'])
    scope='inner' if inner else 'outer';scaler=joblib.load(study.run/f'scalers/{scope}_{fold}/scaler.joblib')
    fake=np.load(study.gan.folder(fold)/'synthetic.npy',mmap_mode='r')
    labels=pd.read_csv(study.gan.folder(fold)/'synthetic_labels.csv').label.to_numpy()
    effective=seed+100*fold+(500000 if inner else 0)
    dataset=AugmentedBags(X,table,train_ids,scaler,arm,split['gan_fit_ids'],fake,labels,effective)
    loader=DataLoader(dataset,batch_size=study.protocol['patient_batch_size'],shuffle=True,num_workers=0,
        collate_fn=collate,generator=torch.Generator().manual_seed(effective))
    network=build_model(model,study.choices[model]['pooling'],study.inherited['model'],study.identity['new_model'])
    validation_ids=split['gan_reference_ids'] if inner else split['outer_validation_ids']
    validation=PatientBags(X,table,validation_ids,scaler)
    weights=patient_class_weights(table[table.hospital_id.isin(train_ids)],3)
    return network,loader,validation,weights,effective


def require_smoke(study,model,device):
    if not (study.run/f'smoke/{model}.json').exists():
        raise RuntimeError(f'Jalankan smoke notebook 41 untuk {model} dahulu')
    saved=read_json(study.run/f'smoke/{model}.json');verify_files(study.run,saved)
    if saved['experiment_hash']!=study.signature or saved['device']!=str(device) or not saved['passed']:
        raise RuntimeError('Jalankan smoke notebook 41 pada device yang sama')


def smoke(study,models,device):
    require_prepared(study)
    with lock(study.run/'.training.lock'):
        for model in models:
            path=study.run/f'smoke/{model}.json'
            if path.exists():require_smoke(study,model,device);print(f'SMOKE cached {model}',flush=True);continue
            records=[]
            for arm in study.protocol['arms']:
                disk_check(study.run,study.protocol);seed_everything(142)
                network,loader,_,weights,_=setup(study,model,1,42,arm,True)
                network=network.to(device);network.train()
                # Exercise full 8x4 budget from eligible GAN-fit patients only.
                eligible=[i for i,indices in enumerate(loader.dataset.indices) if len(indices)>=4][:8]
                if len(eligible)!=8:raise RuntimeError('Need eight eligible training patients for full smoke')
                values,labels,mask,ids,metadata=collate([loader.dataset[i] for i in eligible])
                optimizer=torch.optim.AdamW(network.parameters(),lr=study.choices[model]['learning_rate'],weight_decay=study.training['weight_decay'])
                started=time.monotonic();optimizer.zero_grad(set_to_none=True)
                loss=F.nll_loss(network(values.to(device),mask.to(device)),labels.to(device),weight=weights.to(device))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite smoke loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(network.parameters(),study.training['gradient_clip'],error_if_nonfinite=True)
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in network.parameters()):raise RuntimeError('Nonfinite smoke parameters')
                records.append({'arm':arm,'batch_shape':list(values.shape),'replacements':sum(r['replacements'] for r in metadata),
                    'disposable_optimizer_steps':1,'loss':float(loss.detach().cpu()),'gradient_norm':float(norm.detach().cpu()),
                    'seconds':time.monotonic()-started,'weights_discarded':True})
                print(f'SMOKE PASS {model}/{arm}: {list(values.shape)}',flush=True)
                del network,loader,optimizer,values,labels,mask,loss;gc.collect()
                if device.type=='mps':torch.mps.empty_cache()
            write_manifest(path,{'experiment_hash':study.signature,'device':str(device),'model':model,'passed':True,'arms':records,'files':{}})


def fit_case(study,model,fold,seed,device,arm='inner_selection'):
    if (model not in study.protocol['models'] or fold not in study.protocol['folds']
            or seed not in study.protocol['seeds'] or arm not in ('inner_selection',*study.protocol['arms'])):
        raise ValueError('Unsupported case')
    require_prepared(study);require_smoke(study,model,device)
    inner=arm=='inner_selection'
    folder=directory(study,model,fold,seed,arm);settings={**study.training,'learning_rate':study.choices[model]['learning_rate']}
    selection_hash=None
    if not inner:
        selector=checked_complete(study,model,fold,seed)
        settings['epochs']=selector['selected_epoch']
        selection_hash=sha256_file(directory(study,model,fold,seed)/'completed.json')
    split=study.folds[fold-1]
    actual_arm='real_only' if inner else arm
    case={'experiment_hash':study.signature,'model':model,'fold':fold,'seed':seed,'arm':arm,'device':str(device),
        'effective_seed':seed+100*fold+(500000 if inner else 0),'settings':settings,'pooling':study.choices[model]['pooling'],
        'train_ids':sorted(split['gan_fit_ids'] if inner else split['gan_fit_ids']+split['gan_reference_ids']),
        'evaluation_ids':sorted(split['gan_reference_ids'] if inner else split['outer_validation_ids']),
        'epoch_selection_manifest_sha256':selection_hash,'source_gan_generated_sha256':sha256_file(study.gan.folder(fold)/'generated.json'),
        'scaler_sha256':sha256_file(study.run/f'scalers/{"inner" if inner else "outer"}_{fold}/scaler.joblib'),
        'test_signals_read':False,'outer_epoch_selection':False}
    case={**case,'case_hash':canonical_hash(case)}
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'case.json').exists() and read_json(folder/'case.json')!=case:raise RuntimeError('Case/device changed')
    if (folder/'completed.json').exists():
        checked_complete(study,model,fold,seed,arm);print(f'REUSE {model}/fold{fold}/seed{seed}/{arm}',flush=True);return
    write_json(folder/'case.json',case);disk_check(study.run,study.protocol)
    seed_everything(case['effective_seed'])
    network,loader,validation,weights,effective=setup(study,model,fold,seed,actual_arm,inner)
    network=network.to(device);seed_everything(effective+20000)
    try:
        print(f'TRAIN {model}/fold{fold}/seed{seed}/{arm}; epochs={settings["epochs"]}; '+('inner selection' if inner else 'shared fixed budget'),flush=True)
        fitted=fit(network,loader,validation if inner else None,device,settings,folder,case['case_hash'],weights,study.protocol['evaluation_segment_chunk'])
        name='best.pth' if inner else 'model.pth'
        network.load_state_dict(torch.load(folder/name,map_location='cpu',weights_only=True))
        ids,truth,probability,contributions=predict(network,validation,device,study.protocol['evaluation_segment_chunk'])
        prediction_table(ids,truth,probability,study.inherited['class_names']).to_csv(folder/'predictions.csv',index=False)
        pd.DataFrame(contributions).to_csv(folder/'segment_weights.csv',index=False)
        result={**report(truth,probability,study.inherited['class_names']),**class_diagnostics(truth,probability),**fitted,
            'evaluation_role':'inner_epoch_selection' if inner else 'outer_real_patient_scoring',
            'test_signals_read':False}
        write_json(folder/'metrics.json',result)
        names=('case.json','metrics.json','predictions.csv','segment_weights.csv','history.csv','fit_result.json','fit_finished.json',name,'last.pt')
        write_manifest(folder/'completed.json',{'experiment_hash':study.signature,'case_hash':case['case_hash'],
            'files':{n:sha256_file(folder/n) for n in names}})
    finally:
        del network,loader,validation;gc.collect()
        if device.type=='mps':torch.mps.empty_cache()


def train(study,models,folds,seeds,device,pilot=False):
    require_prepared(study)
    if not pilot:
        for model in models:
            for arm in ('inner_selection',*study.protocol['arms']):checked_complete(study,model,1,42,arm)
    with lock(study.run/'.training.lock'):
        for model in models:
            for fold in ([1] if pilot else folds):
                for seed in ([42] if pilot else seeds):
                    for arm in ('inner_selection',*study.protocol['arms']):fit_case(study,model,fold,seed,device,arm)
