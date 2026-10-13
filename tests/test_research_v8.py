"""Meaningful augmentation, selection, isolation and resume checks on CPU toys."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
import joblib
import nbformat
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml
from sklearn.preprocessing import StandardScaler
from src.artifacts import ROOT,canonical_hash,cpu_weights,read_json,seed_everything,sha256_file,write_json
from research_v4.models import build_model as real_model
from research_v8.common import Study,validate_protocol,directory,progress,checked_complete
from research_v8.data import AugmentedBags,collate
from research_v8.training import fit
from research_v8 import experiment,analysis

P=yaml.safe_load((ROOT/'research_v8/protocol.yaml').read_text())


class Toy(torch.nn.Module):
    classes=3
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(12,3);self.dropout=torch.nn.Dropout(.1)
    def segment(self,x):return F.log_softmax(self.linear(self.dropout(x.mean(-1))),-1)
    def forward(self,x,mask=None):
        logp=self.segment(x.flatten(0,1)).reshape(len(x),x.shape[1],3)
        return torch.logsumexp(logp.masked_fill(~mask[...,None],-torch.inf),1)-mask.sum(1).log()[:,None]
    @torch.no_grad()
    def all_segments(self,x,chunk):
        return self.segment(x).exp().mean(0),torch.full((len(x),),1/len(x))


def fixture(folder,windows=6):
    ids=[f'p{i:02}' for i in range(15)]
    table=pd.DataFrame({'hospital_id':np.repeat(ids,windows),'label':np.repeat(np.arange(15)%3,windows)})
    X=np.random.default_rng(13).normal(size=(len(table),12,64)).astype('float32')
    fake=np.random.default_rng(14).normal(size=(6,12,64)).astype('float32');labels=np.repeat(range(3),2)
    records=[{'hospital_id':h,'label':i%3,'class_name':f'c{i%3}','pool':'development'} for i,h in enumerate(ids)]
    split={'fold':1,'gan_fit_ids':ids[:9],'gan_reference_ids':ids[9:12],'outer_validation_ids':ids[12:]}
    protocol=deepcopy(P);protocol.update(models=['hierarchical'],folds=[1],seeds=[42],minimum_free_gib=0)
    run=folder/'run';run.mkdir()
    train={'epochs':2,'patience':99,'scheduler_t_max':100,'minimum_lr':1e-6,'weight_decay':1e-4,'gradient_clip':1.,'optimizer':'AdamW'}
    source=folder/'gan/fold_1';source.mkdir(parents=True)
    np.save(source/'synthetic.npy',fake);pd.DataFrame({'label':labels}).to_csv(source/'synthetic_labels.csv',index=False)
    write_json(source/'generated.json',{'toy':True})
    baseline=folder/'baseline/fold_1';baseline.mkdir(parents=True)
    scaler=StandardScaler().fit(X[:12*windows].transpose(0,2,1).reshape(-1,12))
    joblib.dump(scaler,baseline/'scaler.joblib');write_json(baseline/'provenance.json',{'toy':True})
    identity={'development_cache':{'toy':'hash'},'new_model':{},'outer_scalers':
        {str((baseline/n).relative_to(ROOT)):sha256_file(baseline/n) for n in ('scaler.joblib','provenance.json')}}
    gan=SimpleNamespace(folder=lambda f:folder/f'gan/fold_{f}')
    study=Study(run,folder,protocol,records,[split],{'hierarchical':{'pooling':'mean','learning_rate':1e-3}},
        {'model':{},'class_names':['c0','c1','c2']},train,gan,identity)
    write_json(run/'protocol.json',{'development_cache':identity['development_cache']})
    return study,X,table,fake,labels


class V8Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_protocol_rejects_test_dose_and_unmatched_budget(self):
        validate_protocol(P)
        for key,value in [('test_signals_enabled',True),('segments_per_patient',8),('outer_budget','best_outer_f1')]:
            p=deepcopy(P);p[key]=value
            with self.assertRaises(ValueError):validate_protocol(p)
        p=deepcopy(P);p['augmentation']['replacements_per_eligible_bag']=2
        with self.assertRaises(ValueError):validate_protocol(p)

    def test_matched_crops_slots_fit_only_same_class_and_short_bags(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            study,X,table,fake,labels=fixture(Path(temp))
            ids=study.folds[0]['gan_fit_ids']
            scaler=StandardScaler().fit(X[:54].transpose(0,2,1).reshape(-1,12))
            sets={a:AugmentedBags(X,table,ids,scaler,a,ids,fake,labels,142) for a in P['arms']}
            for i in range(len(ids)):
                base,y,h,_=sets['real_only'][i];r,_,_,rm=sets['real_aug'][i];g,_,_,gm=sets['gan_aug'][i]
                self.assertEqual(rm['slot'],gm['slot']);slot=rm['slot']
                keep=[k for k in range(4) if k!=slot]
                torch.testing.assert_close(base[keep],r[keep],atol=0,rtol=0);torch.testing.assert_close(base[keep],g[keep],atol=0,rtol=0)
                self.assertIn(rm['donor_hospital_id'],ids);self.assertNotEqual(rm['donor_hospital_id'],h)
                self.assertEqual(int(table.iloc[rm['donor_segment_index']].label),y)
                self.assertEqual(labels[gm['synthetic_index']],y)
            short=table.drop(table[table.hospital_id.eq(ids[0])].index[1:]).reset_index(drop=True)
            short_X=np.random.default_rng(3).normal(size=(len(short),12,64)).astype('float32')
            for a in P['arms']:
                dataset=AugmentedBags(short_X,short,ids,scaler,a,ids,fake,labels,142)
                value,_,_,meta=dataset[0];self.assertEqual(len(value),1);self.assertFalse(meta['eligible']);self.assertEqual(meta['replacements'],0)
            with self.assertRaises(ValueError):AugmentedBags(X,table,ids,scaler,'gan_aug',table.hospital_id.unique(),fake,labels,142)

    def test_resume_restores_crop_slot_source_and_dropout_exactly(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            root=Path(temp);study,X,table,fake,labels=fixture(root)
            ids=study.folds[0]['gan_fit_ids'];scaler=StandardScaler().fit(X[:54].transpose(0,2,1).reshape(-1,12))
            def setup():
                seed_everything(142);net=Toy()
                data=AugmentedBags(X,table,ids,scaler,'gan_aug',ids,fake,labels,142)
                loader=DataLoader(data,batch_size=4,shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(142))
                return net,loader
            cfg={**study.training,'learning_rate':1e-3};a=root/'full';b=root/'resume'
            net,loader=setup();fit(net,loader,None,torch.device('cpu'),cfg,a,'case',torch.ones(3),16);expected=cpu_weights(net)
            net,loader=setup();fit(net,loader,None,torch.device('cpu'),cfg,b,'case',torch.ones(3),16,stop_after=1)
            net,loader=setup();fit(net,loader,None,torch.device('cpu'),cfg,b,'case',torch.ones(3),16)
            for key,v in net.state_dict().items():torch.testing.assert_close(v,expected[key],rtol=0,atol=0)
            pd.testing.assert_frame_equal(pd.read_csv(a/'history.csv').drop(columns='seconds'),pd.read_csv(b/'history.csv').drop(columns='seconds'))
            # Simulate interruption after the final last.pt commit but before CSV/final files.
            c=root/'final_commit_gap';c.mkdir();shutil.copy2(a/'last.pt',c/'last.pt')
            pd.read_csv(a/'history.csv').iloc[:1].to_csv(c/'history.csv',index=False)
            net,loader=setup();fit(net,loader,None,torch.device('cpu'),cfg,c,'case',torch.ones(3),16)
            self.assertEqual(len(pd.read_csv(c/'history.csv')),2)
            for key,v in net.state_dict().items():torch.testing.assert_close(v,expected[key],rtol=0,atol=0)

    def test_prepare_and_inner_selector_do_not_read_outer_and_outer_arms_share_budget(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            root=Path(temp);study,X,table,fake,labels=fixture(root)
            original=X.copy();X[54:]=np.nan  # includes reference + outer for preparation
            with patch('research_v8.experiment.load_development',return_value=(X,table)):
                experiment.prepare(study)
            inner=joblib.load(study.run/'scalers/inner_1/scaler.joblib')
            expected=StandardScaler().fit(original[:54].transpose(0,2,1).reshape(-1,12))
            np.testing.assert_allclose(inner.mean_,expected.mean_);np.testing.assert_allclose(inner.scale_,expected.scale_)
            X[54:72]=original[54:72]  # inner-reference available; outer remains NaN
            with patch('research_v8.experiment.load_development',return_value=(X,table)), \
                 patch('research_v8.experiment.build_model',side_effect=lambda *args:Toy()):
                experiment.smoke(study,['hierarchical'],torch.device('cpu'))
                experiment.fit_case(study,'hierarchical',1,42,torch.device('cpu'))
                selector=checked_complete(study,'hierarchical',1,42)
                self.assertEqual(selector['evaluation_role'],'inner_epoch_selection')
                self.assertTrue((study.run/'scalers/inner_1/scaler.joblib').exists())
                X[72:]=original[72:]
                for arm in P['arms']:experiment.fit_case(study,'hierarchical',1,42,torch.device('cpu'),arm)
                before=sha256_file(directory(study,'hierarchical',1,42,'gan_aug')/'model.pth')
                experiment.fit_case(study,'hierarchical',1,42,torch.device('cpu'),'gan_aug')
                self.assertEqual(before,sha256_file(directory(study,'hierarchical',1,42,'gan_aug')/'model.pth'))
            for arm in P['arms']:
                r=checked_complete(study,'hierarchical',1,42,arm)
                self.assertEqual(r['epochs_run'],selector['selected_epoch']);self.assertEqual(r['selected_epoch'],selector['selected_epoch'])
            with patch('research_v8.analysis.load_development',return_value=(X,table)):
                analysis.analyze(study,pilot=True)
            self.assertTrue((study.run/'analysis_pilot/comparison_f1.png').exists())
            self.assertEqual(progress(study)['outer_comparison_complete'],3)
            with self.assertRaises(RuntimeError):checked_complete(study,'encoder',1,42,'gan_aug')
            target=directory(study,'hierarchical',1,42,'gan_aug')/'predictions.csv'
            target.write_text(target.read_text()+'\n')
            with self.assertRaises(RuntimeError):checked_complete(study,'hierarchical',1,42,'gan_aug')

    def test_real_backbones_accept_native_k4_and_backpropagate_cpu_synthetic(self):
        # Synthetic only; no genuine patient or MPS forward is used by this test.
        inherited=read_json(ROOT/'results/research_v4/20261002_patient_mil_hybrids_v4/protocol.json')['inherited_config']['model']
        for name,pooling in [('encoder','attention'),('hierarchical','mean')]:
            seed_everything(17);net=real_model(name,pooling,inherited,{})
            values=torch.randn(1,4,12,5000);mask=torch.ones(1,4,dtype=torch.bool)
            logp=net(values,mask);self.assertEqual(tuple(logp.shape),(1,3))
            F.nll_loss(logp,torch.tensor([1])).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_full_five_fold_analysis_reconstructs_unique_oof_and_paired_effects(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            root=Path(temp);study,X,table,fake,labels=fixture(root)
            ids=[r['hospital_id'] for r in study.records];splits=[]
            study.protocol['folds']=[1,2,3,4,5]
            for f in range(1,6):
                outer=ids[(f-1)*3:f*3];reference=ids[(f%5)*3:(f%5+1)*3]
                fit_ids=sorted(set(ids)-set(outer)-set(reference))
                splits.append({'fold':f,'gan_fit_ids':fit_ids,'gan_reference_ids':reference,'outer_validation_ids':outer})
                folder=root/f'gan/fold_{f}';folder.mkdir(exist_ok=True)
                np.save(folder/'synthetic.npy',fake);pd.DataFrame({'label':labels}).to_csv(folder/'synthetic_labels.csv',index=False)
                write_json(folder/'generated.json',{'toy':True,'fold':f})
                baseline=root/f'baseline/fold_{f}';baseline.mkdir(exist_ok=True)
                indices=np.flatnonzero(table.hospital_id.isin(fit_ids+reference).to_numpy())
                scaler=StandardScaler().fit(X[indices].transpose(0,2,1).reshape(-1,12))
                joblib.dump(scaler,baseline/'scaler.joblib');write_json(baseline/'provenance.json',{'toy':True,'fold':f})
                study.identity['outer_scalers'].update({str((baseline/n).relative_to(ROOT)):sha256_file(baseline/n) for n in ('scaler.joblib','provenance.json')})
            study.folds=splits
            with patch('research_v8.experiment.load_development',return_value=(X,table)), \
                    patch('research_v8.experiment.build_model',side_effect=lambda *args:Toy()):
                experiment.prepare(study);experiment.smoke(study,['hierarchical'],torch.device('cpu'))
                experiment.train(study,['hierarchical'],[1],[42],torch.device('cpu'),pilot=True)
                experiment.train(study,['hierarchical'],[1,2,3,4,5],[42],torch.device('cpu'))
            with patch('research_v8.analysis.load_development',return_value=(X,table)):analysis.analyze(study)
            self.assertTrue(progress(study)['ready_for_comparison'])
            out=study.run/'analysis_complete'
            frames=pd.read_csv(out/'patient_predictions.csv',dtype={'hospital_id':str})
            for _,group in frames.groupby(['model','seed','arm']):
                self.assertEqual(len(group),15);self.assertTrue(group.hospital_id.is_unique)
            self.assertEqual(len(pd.read_csv(out/'pooled_oof.csv')),3)

    def test_new_notebooks_are_unexecuted_valid_and_endpoint_names_exist(self):
        files=[p for p in (ROOT/'notebooks').glob('*V8*.ipynb')]
        self.assertEqual(len(files),5)
        for path in files:
            nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
            for c in nb.cells:
                if c.cell_type=='code':
                    self.assertIsNone(c.execution_count);self.assertFalse(c.outputs);ast.parse(c.source)


if __name__=='__main__':unittest.main()
