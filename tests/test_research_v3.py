"""Synthetic fixtures: sampling, annotated matching, no-test selection and resume."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from src.artifacts import read_json, write_json, seed_everything
from src.training import fit_classifier
from research_v3.sampling import patient_weights, patient_class_weights, training_loader
from research_v3.candidates import detect_candidates, extract_windows, consensus_matches
from research_v3.benchmark import score_peaks
from research_v3.common import require_final_complete
from research_v3.experiment import fit, summarize, final, evaluate
from analysis_v2.explain_v2 import middle_segments, single_output_values, ClassLogit

ROOT = Path(__file__).resolve().parents[1]


class ResearchV3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = yaml.safe_load((ROOT / 'research_v3/protocol.yaml').read_text())

    def test_equal_patient_mass_and_fold_class_weights(self):
        table = pd.DataFrame({'hospital_id': ['a','b','b','b','c','d'], 'label': [0,1,1,1,2,0]})
        weights = patient_weights(table.hospital_id)
        for group in table.hospital_id.unique():
            self.assertAlmostEqual(weights[table.hospital_id == group].sum(), 1.0)
        self.assertTrue(torch.allclose(patient_class_weights(table, 3), torch.tensor([.6,1.2,1.2])))
        bad = table.copy(); bad.loc[2, 'label'] = 2
        with self.assertRaises(ValueError):
            patient_class_weights(bad, 3)
        dataset = torch.utils.data.TensorDataset(torch.arange(6)[:,None], torch.zeros(6, dtype=torch.long))
        loader = training_loader(dataset, table, 8, 42, True, num_samples=20000)
        sampled = torch.cat([x[:,0] for x,y in loader]).numpy()
        for group in table.hospital_id.unique():
            self.assertLess(abs(np.mean(table.hospital_id.to_numpy()[sampled] == group)-.25), .025)

    def test_detector_polarity_and_no_signal_fallback(self):
        spec = self.cfg['candidate']
        time = np.arange(20000)/2000
        centers = np.arange(1,10)*2000
        waveform = sum(np.exp(-((time-c/2000)/.015)**2) for c in centers)
        raw = np.tile(waveform[:,None], (1,12))
        raw[:,7] *= -1
        peaks,qa = detect_candidates(raw,spec)
        negative,_ = detect_candidates(-raw,spec)
        self.assertTrue(np.array_equal(peaks,negative))
        self.assertEqual(len(peaks),len(centers))
        self.assertLess(np.max(np.abs(peaks-centers)),80)
        values,rows=extract_windows(raw,peaks,spec)
        self.assertEqual(values.shape,(9,12,1000))
        self.assertFalse(rows[0]['is_fallback'])
        peaks,qa=detect_candidates(np.zeros_like(raw),spec)
        self.assertTrue(qa['engineering_review_flag'])
        values,rows=extract_windows(raw,peaks,spec)
        self.assertTrue(rows[0]['is_fallback'])
        self.assertFalse(rows[0]['pvc_verified'])

    def test_annotation_matching_is_one_to_one(self):
        pairs=consensus_matches([100,110],[105],20)
        self.assertEqual(len(pairs),1)
        metrics,_=score_peaks([100,200,300],[100,205,600],1000,.01)
        self.assertEqual((metrics['tp'],metrics['fp'],metrics['fn']),(2,1,1))
        self.assertAlmostEqual(metrics['f1'],2/3)

    def test_partial_patch_keeps_tail_and_gradient(self):
        from research_v3.models import complete_patches, build_model
        features=torch.arange(62*8).reshape(1,62,8).float().requires_grad_()
        patches,weights=complete_patches(features,16)
        self.assertEqual(patches.shape,(1,4,128))
        self.assertAlmostEqual(weights[-1].item(),14/16)
        self.assertTrue(torch.equal(patches.reshape(1,64,8)[:,:62],features))
        patches.sum().backward()
        self.assertTrue(torch.equal(features.grad,torch.ones_like(features)))
        config=yaml.safe_load((ROOT/'configs/finetune_fixed.yaml').read_text())['model']
        model=build_model('patchtst',config).eval()
        x=torch.randn(2,12,1000,requires_grad=True)
        model(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_explanation_balances_patients_and_shape(self):
        table=pd.DataFrame({'hospital_id': ['a']*20+['b','c','d','e','f'],
                            'label':[0]*20+[0,1,1,2,2]})
        reps=middle_segments(table,6,42,True)
        self.assertEqual(reps.hospital_id.nunique(),6)
        self.assertEqual(reps.label.value_counts().to_dict(),{0:2,1:2,2:2})
        self.assertTrue(np.array_equal(single_output_values(np.ones((1,12,4,1)),(1,12,4)),np.ones((1,12,4))))
        with self.assertRaises(ValueError):
            single_output_values(np.zeros((1,12,4,3)),(1,12,4))
        import shap
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(48,3))
        wrapper=ClassLogit(model,1)
        values=shap.GradientExplainer(wrapper,torch.zeros(2,12,4)).shap_values(torch.ones(1,12,4),nsamples=4,rseed=42)
        self.assertEqual(single_output_values(values,(1,12,4)).shape,(1,12,4))

    def test_patient_sampler_resume_matches_uninterrupted(self):
        import src.training as training
        cfg=yaml.safe_load((ROOT / 'configs/finetune_fixed.yaml').read_text())
        settings={**cfg['training'], 'epochs':2,'learning_rate':.001,'batch_size':4}
        table=pd.DataFrame({'hospital_id':['a','b','b','c','c','c'], 'label':[0,1,1,2,2,2]})
        x=torch.arange(24).reshape(6,4).float()/24
        dataset=torch.utils.data.TensorDataset(x,torch.tensor(table.label.to_numpy()))
        def run(path):
            seed_everything(7)
            model=torch.nn.Linear(4,3)
            loader=training_loader(dataset,table,4,7,True,num_samples=12)
            return fit_classifier(model,loader,None,None,torch.device('cpu'),settings,path,['a','b','c'],True)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            run(path/'whole')
            original=training.train_epoch
            calls=0
            def stop(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:
                    raise KeyboardInterrupt()
                return original(*args,**kwargs)
            with patch.object(training,'train_epoch',stop),self.assertRaises(KeyboardInterrupt):
                run(path/'resumed')
            run(path/'resumed')
            a=torch.load(path/'whole/model.pth',weights_only=True)
            b=torch.load(path/'resumed/model.pth',weights_only=True)
            for key in a:
                self.assertTrue(torch.equal(a[key],b[key]))

    def test_full_tiny_workflow_and_test_lock(self):
        import research_v3.experiment as exp
        cfg=copy.deepcopy(self.cfg)
        cfg.update(models=['encoder'],minimum_free_gib=0,final_seeds=[42])
        cfg['classical'].update(n_estimators=5,max_depth=3,max_features=[.7],min_samples_leaf=[1])
        config=yaml.safe_load((ROOT/'configs/finetune_fixed.yaml').read_text())
        config.update(models=['encoder'],class_names=['LVOT','RVOT_cusp','RVOT_septal_freewall_other'])
        config['training'].update(epochs=1,patience=1,eval_batch_size=6)
        grid=[{'id':'S01','learning_rate':.001,'batch_size':6}]
        with tempfile.TemporaryDirectory() as temp,patch.object(exp,'build_model',lambda *a:torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(384,3))):
            run,source=Path(temp)/'run',Path(temp)/'source'
            ids=[str(i) for i in range(30)]
            folds=[{'train_ids':[s for s in ids if s not in ids[f*6:(f+1)*6]],'validation_ids':ids[f*6:(f+1)*6]} for f in range(5)]
            write_json(run/'folds/manifest.json',folds)
            write_json(run/'protocol.json',{'experiment_hash':'fixture'})
            rng=np.random.default_rng(42)
            for directory in [source/'data/development',run/'representations/r_candidate/data/development']:
                directory.mkdir(parents=True)
                np.save(directory/'X.npy',rng.normal(size=(30,12,32)).astype('float32'))
                pd.DataFrame({'hospital_id':ids,'label':np.tile([0,1,2],10)}).to_csv(directory/'segments.csv',index=False)
            with self.assertRaises(RuntimeError):
                require_final_complete(run,cfg)
            with self.assertRaises(RuntimeError):
                summarize(run,cfg,config,grid,select=True)
            device=torch.device('cpu')
            for arm in cfg['arms']:
                for f in range(1,6):
                    fit(run,source,cfg,config,'encoder',arm,grid[0],device,fold=f)
            # Incomplete classical comparator also blocks selection/test.
            with self.assertRaises(RuntimeError):
                summarize(run,cfg,config,grid,select=True)
            from research_v3.classical import scenarios
            for arm in cfg['classical_arms']:
                for f in range(1,6):
                    fit(run,source,cfg,config,'extratrees',arm,scenarios(cfg)[0],device,fold=f)
            summarize(run,cfg,config,grid,select=True)
            final(run,source,cfg,config,grid,'encoder',device)
            with self.assertRaises(RuntimeError):
                require_final_complete(run,cfg)
            final(run,source,cfg,config,grid,'extratrees',device)
            require_final_complete(run,cfg)
            self.assertFalse((run/'representations/r_candidate/data/test').exists())
            for directory in [source/'data/test',run/'representations/r_candidate/data/test']:
                directory.mkdir(parents=True)
                np.save(directory/'X.npy',rng.normal(size=(6,12,32)).astype('float32'))
                pd.DataFrame({'hospital_id':[str(i+100) for i in range(6)],'label':[0,1,2,0,1,2]}).to_csv(directory/'segments.csv',index=False)
            with patch.object(exp,'prepare'):
                evaluate(run,source,cfg,config,{},device)
            self.assertEqual(read_json(run/'test/encoder/seed_42/metrics.json')['n_patients'],6)
            path=run/'cv/window_patient/encoder/S01/fold_1/metrics.json'
            before=path.read_bytes()
            fit(run,source,cfg,config,'encoder','window_patient',grid[0],device,fold=1)
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(RuntimeError):
                fit(run,source,cfg,config,'encoder','window_patient',{**grid[0],'learning_rate':.2},device,fold=1)

    def test_candidate_preparation_keeps_patients_without_test_reads(self):
        import research_v3.data as data
        from src.data import LEADS
        cfg=copy.deepcopy(self.cfg);cfg['minimum_free_gib']=0
        v2cfg=yaml.safe_load((ROOT/'research_v2/protocol.yaml').read_text())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';rawdir=root/'PVCVTRawECGData';rawdir.mkdir()
            records=[{'hospital_id':str(i),'label':i%3,'sublocation':['LCC','LC','FreeWall'][i%3],
                      'pool':'development' if i<6 else 'test'} for i in range(9)]
            write_json(run/'data/patients.json',records)
            write_json(run/'protocol.json',{'experiment_hash':'fixture','protocol':cfg})
            time=np.arange(5000)/2000
            wave=np.exp(-((time-.7)/.015)**2)+np.exp(-((time-1.5)/.015)**2)
            for i in range(6):
                values=np.zeros((5000,12)) if i==0 else np.tile(wave[:,None],(1,12))
                pd.DataFrame(values,columns=LEADS).to_csv(rawdir/f'{i}.csv',index=False)
            with patch.object(data,'ROOT',root):
                directory=data.prepare(run,root,cfg,{'sampling_rate':2000},v2cfg)
                X,segments=data.load_pool(run,root,'r_candidate_patient')
                self.assertEqual(set(segments.hospital_id),set(map(str,range(6))))
                self.assertEqual(X.shape[1:],(12,1000))
                self.assertTrue(segments.loc[segments.hospital_id=='0','is_fallback'].all())
                self.assertTrue(np.isfinite(X).all())
                self.assertEqual(data.prepare(run,root,cfg,{'sampling_rate':2000},v2cfg),directory)
                with self.assertRaises(RuntimeError):
                    data.prepare(run,root,cfg,{'sampling_rate':2000},v2cfg,'test')
                self.assertFalse((run/'representations/r_candidate/data/test').exists())


if __name__=='__main__':
    unittest.main()
