"""Inner-only epoch selection; matched fixed outer budgets and exact resume."""
from collections import Counter
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from src.artifacts import cpu_weights, read_json, restore_rng, rng_state, save_torch, sha256_file, write_json
from src.evaluation import compute_metrics
from research_v4.training import predict, class_diagnostics, batchnorm_diagnostics
from .common import verify_files, write_manifest


def train_epoch(model,loader,optimizer,device,weights,clip):
    model.train();loss_total=correct=total=steps=norm_total=windows=eligible=replacements=0
    anchors,donors,fakes,classes,aug_classes=Counter(),Counter(),Counter(),Counter(),Counter()
    for values,labels,mask,ids,metadata in loader:
        optimizer.zero_grad(set_to_none=True)
        logp=model(values.to(device),mask.to(device));targets=labels.to(device)
        loss=F.nll_loss(logp,targets,weight=weights)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite classifier loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
        optimizer.step()
        count=len(labels);total+=count;steps+=1;loss_total+=float(loss.detach().cpu())*count
        correct+=int((logp.detach().argmax(1)==targets).sum().cpu());norm_total+=float(norm.detach().cpu())
        anchors.update(ids);classes.update(labels.tolist());windows+=int(mask.sum())
        for meta in metadata:
            eligible+=int(meta['eligible']);replacements+=meta['replacements']
            if meta['replacements']:aug_classes[meta['label']]+=1
            if meta['donor_hospital_id'] is not None:donors.update([meta['donor_hospital_id']])
            if meta['synthetic_index'] is not None:fakes.update([str(meta['synthetic_index'])])
    if len(anchors)!=len(loader.dataset) or set(anchors.values())!={1}:
        raise RuntimeError('Every anchor patient must be visited exactly once per epoch')
    return {'train_loss':loss_total/total,'train_patient_bag_accuracy':correct/total,
        'optimizer_steps':steps,'gradient_norm_before_clip_mean':norm_total/steps,
        'anchor_patients':total,'bag_windows':windows,'eligible_bags':eligible,
        'replacements':replacements,'anchor_windows_retained':windows-replacements,
        'real_donor_windows':sum(donors.values()),'synthetic_windows':sum(fakes.values()),
        'anchor_exposure_json':json.dumps(dict(sorted(anchors.items()))),
        'real_donor_exposure_json':json.dumps(dict(sorted(donors.items()))),
        'synthetic_exposure_json':json.dumps(dict(sorted(fakes.items()))),
        **{f'class_{c}_anchors':classes[c] for c in range(3)},
        **{f'class_{c}_replacements':aug_classes[c] for c in range(3)}}


def fit(model,loader,validation,device,settings,folder,case_hash,class_weights,chunk,stop_after=None):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit_finished.json').exists():
        done=read_json(folder/'fit_finished.json');verify_files(folder,done)
        if done['case_hash']!=case_hash or done['device']!=str(device):raise RuntimeError('Fit cache changed')
        return read_json(folder/'fit_result.json')
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings['scheduler_t_max'],eta_min=settings['minimum_lr'])
    weights=class_weights.to(device)
    history=[];start_epoch=0;best=-1.;best_epoch=0;patience=0;elapsed_before=0.
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if state['case_hash']!=case_hash or state['device']!=str(device):raise RuntimeError('Resume identity/device changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng']);loader.generator.set_state(state['loader_rng']);loader.dataset.restore_rng_states(state['dataset_rng'])
        history=state['history'];start_epoch=state['epoch'];best=state['best'];best_epoch=state['best_epoch']
        patience=state['patience'];elapsed_before=state['elapsed_seconds']
        print(f'RESUME epoch={start_epoch}',flush=True)
    began=time.monotonic();maximum=settings['epochs'] if stop_after is None else min(settings['epochs'],stop_after)
    for epoch in range(start_epoch,maximum):
        if validation is not None and patience>=settings['patience']:break
        started=time.monotonic()
        row={'epoch':epoch+1,'learning_rate':optimizer.param_groups[0]['lr'],
            **train_epoch(model,loader,optimizer,device,weights,settings['gradient_clip']),**batchnorm_diagnostics(model)}
        if validation is not None:
            _,truth,probability,_=predict(model,validation,device,chunk)
            metrics=compute_metrics(truth,probability)
            row.update({f'inner_{k}':v for k,v in metrics.items()});row.update(class_diagnostics(truth,probability,'inner'))
            if metrics['f1']>best:
                best=metrics['f1'];best_epoch=epoch+1;patience=0;save_torch(folder/'best.pth',cpu_weights(model))
            else:patience+=1
        else:best_epoch=epoch+1
        scheduler.step();row['seconds']=time.monotonic()-started;history.append(row)
        save_torch(folder/'last.pt',{'case_hash':case_hash,'device':str(device),'epoch':epoch+1,
            'model':cpu_weights(model),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'rng':rng_state(),'loader_rng':loader.generator.get_state(),'dataset_rng':loader.dataset.rng_states(),
            'history':history,'best':best,'best_epoch':best_epoch,'patience':patience,
            'elapsed_seconds':elapsed_before+time.monotonic()-began})
        tmp=folder/'history.csv.tmp';pd.DataFrame(history).to_csv(tmp,index=False);tmp.replace(folder/'history.csv')
        suffix=f' inner_F1={row["inner_f1"]:.4f} selected_epoch={best_epoch}' if validation is not None else ' fixed budget; outer not evaluated'
        print(f'epoch {epoch+1}/{settings["epochs"]} loss={row["train_loss"]:.4f} replacements={row["replacements"]}'+suffix,flush=True)
    finished=len(history)==settings['epochs'] or (validation is not None and patience>=settings['patience'])
    if not finished:return {'partial':True,'epochs_run':len(history)}
    # A stop between last.pt and the CSV commit must not lose the last epoch's history.
    tmp=folder/'history.csv.tmp';pd.DataFrame(history).to_csv(tmp,index=False);tmp.replace(folder/'history.csv')
    if validation is None:save_torch(folder/'model.pth',cpu_weights(model))
    selected='best.pth' if validation is not None else 'model.pth'
    result={'selected_epoch':best_epoch,'epochs_run':len(history),'optimizer_steps':sum(h['optimizer_steps'] for h in history),
        'inner_selection':validation is not None,'elapsed_seconds':elapsed_before+time.monotonic()-began,
        'selected_checkpoint_sha256':sha256_file(folder/selected)}
    write_json(folder/'fit_result.json',result)
    names=('last.pt','history.csv','fit_result.json',selected)
    write_manifest(folder/'fit_finished.json',{'case_hash':case_hash,'device':str(device),'files':{n:sha256_file(folder/n) for n in names}})
    return result
