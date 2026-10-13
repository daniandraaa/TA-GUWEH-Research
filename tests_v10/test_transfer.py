"""Synthetic CPU regression tests only; never load genuine ECG or old fits."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.artifacts import seed_everything
from research_v10.data import adapt, fit_feature_scale
from research_v10.model import ECGFMEncoder, config, TailClassifier, official_unmasked_forward
from research_v10.training import FeatureClassifier, Patients, train_epoch, fit
from research_v10.common import grid, select_best


class TransferTests(unittest.TestCase):
    def test_adapter_preserves_length_and_leads(self):
        t=np.arange(5000)/2000
        signal=np.sin(2*np.pi*10*t)
        native=np.broadcast_to(signal,(2,12,5000)).copy()
        native[:,1]=3
        out=adapt(native)
        self.assertEqual(out.shape,(2,12,1250))
        self.assertTrue(np.all(out[:,1]==0))
        self.assertTrue(np.allclose(out[:,0].mean(-1),0,atol=1e-6))
        self.assertTrue(np.allclose(out[:,0].std(-1),1,atol=1e-6))
        frequency=np.fft.rfftfreq(1250,1/500)[np.abs(np.fft.rfft(out[0,0])).argmax()]
        self.assertAlmostEqual(frequency,10,delta=.2)
        with self.assertRaises(ValueError):adapt(np.zeros((1,12,2500)))

    def test_scaler_training_only(self):
        x=np.array([[1,2],[3,2],[999999,-99999]],dtype='float32')
        mean,scale=fit_feature_scale(x,[0,1])
        np.testing.assert_array_equal(mean,[2,2]);np.testing.assert_array_equal(scale,[1,1])

    def test_official_assembly_tail_and_gradient(self):
        seed_everything(4)
        cfg=config({'encoder_layers':2,'encoder_embed_dim':32,'encoder_ffn_embed_dim':64,
            'encoder_attention_heads':4,'conv_pos_groups':4,'dropout':0,'attention_dropout':0})
        model=ECGFMEncoder(cfg).eval().requires_grad_(False)
        with torch.no_grad():
            x=torch.randn(2,12,1250);actual=model(x);before=model.until_last(x)
            self.assertTrue(torch.equal(actual,official_unmasked_forward(model,x)))
            self.assertTrue(torch.equal(actual,model.finish(before)))
        tail=TailClassifier(model,np.zeros(32,dtype='float32'),np.ones(32,dtype='float32')).eval()
        torch.testing.assert_close(tail.patient_embedding(before),actual.mean((0,1)),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(tail.patient_embedding(before,chunk=1),tail.patient_embedding(before,chunk=2),rtol=1e-5,atol=1e-6)
        tail.train();loss=tail(before).square().sum();loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().max()>0 for p in tail.tail.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_effective_batch_class_weight_gradient(self):
        seed_everything(8)
        data=Patients(['a','b','c','d'],[0,0,1,2],features=np.arange(20,dtype='float32').reshape(4,5)/20)
        first=FeatureClassifier(np.zeros(5,dtype='float32'),np.ones(5,dtype='float32'))
        second=deepcopy(first);weights=torch.tensor([.6,1.2,2.4])
        optimizer=torch.optim.SGD(first.parameters(),lr=.01)
        generator=torch.Generator().manual_seed(91)
        train_epoch(first,data,optimizer,torch.device('cpu'),weights,4,generator,1e6)
        optimizer=torch.optim.SGD(second.parameters(),lr=.01)
        logits=second(torch.tensor(np.array(data.features)))
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor(data.labels),weight=weights)
        loss.backward();optimizer.step()
        for a,b in zip(first.parameters(),second.parameters()):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)

    def test_exact_epoch_resume_and_history_repair(self):
        x=np.random.default_rng(1).normal(size=(12,5)).astype('float32')
        train=Patients([f'p{i}' for i in range(12)],[0,1,2]*4,features=x)
        settings={'effective_seed':3,'learning_rate':.01,'weight_decay':.001,'epochs':4,
            'scheduler_t_max':4,'minimum_lr':0.,'batch_size':4,'gradient_clip':1.,'patience':4}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def model():
                seed_everything(3)
                return FeatureClassifier(np.zeros(5,dtype='float32'),np.ones(5,dtype='float32'))
            uninterrupted=model();fit(uninterrupted,train,None,torch.device('cpu'),settings,root/'a','a',root/'resume')
            partial=model();result=fit(partial,train,None,torch.device('cpu'),settings,root/'b','b',root/'resume',stop_after=2)
            self.assertTrue(result['partial'])
            (root/'b/history.csv').write_text('broken csv\n')
            resumed=model();fit(resumed,train,None,torch.device('cpu'),settings,root/'b','b',root/'resume')
            for name,tensor in uninterrupted.state_dict().items():self.assertTrue(torch.equal(tensor,resumed.state_dict()[name]))
            a=pd.read_csv(root/'a/history.csv');b=pd.read_csv(root/'b/history.csv');pd.testing.assert_frame_equal(a,b)

    def test_nine_grid_inner_tie_break(self):
        p={'head_learning_rates':[1e-4,1e-3,1e-2],'tail_learning_rates':[1e-6,3e-6,1e-5],'patient_batch_sizes':[8,16,32]}
        scenarios=grid(p,'pretrained_probe');self.assertEqual(len(scenarios),9)
        rows=[{**r,'f1':.5} for r in scenarios]
        choice=select_best(rows);self.assertEqual(choice['learning_rate'],1e-4);self.assertEqual(choice['batch_size'],8)
        with self.assertRaises(RuntimeError):select_best(rows[:8])

    def test_synthetic_inner_grid_outer_refit_and_audit(self):
        from research_v10.experiment import benchmark
        from research_v10.analysis import analyze
        ids=[f'p{i:02}' for i in range(30)];labels=[i%3 for i in range(30)]
        patients=pd.DataFrame({'hospital_id':ids,'label':labels,'windows':1})
        windows=pd.DataFrame({'hospital_id':ids,'label':labels,'segment_index':range(30)})
        encoder=ECGFMEncoder(config({'encoder_layers':2,'encoder_embed_dim':32,'encoder_ffn_embed_dim':64,
            'encoder_attention_heads':4,'conv_pos_groups':4,'dropout':0,'attention_dropout':0})).eval().requires_grad_(False)
        tokens=np.random.default_rng(2).normal(size=(30,5,32)).astype('float32')
        with torch.no_grad(): x=encoder.finish(torch.tensor(tokens)).mean(1).numpy()
        arrays={'random_probe':x,'pretrained_probe':x.copy(),'tokens':tokens}
        p={'head_learning_rates':[1e-4,1e-3,1e-2],'tail_learning_rates':[1e-6,3e-6,1e-5],
            'patient_batch_sizes':[8,16,32],'probe_arms':['random_probe','pretrained_probe'],
            'folds':[1,2,3,4,5],'outer_seeds':[42,43,44],'cv_seed':42,'minimum_free_gib':3,
            'head_epochs':3,'head_patience':2,'tail_epochs':3,'tail_patience':2,'weight_decay':1e-4,'gradient_clip':1.,
            'selection':'inner_macro_f1_then_lower_learning_rate_then_smaller_batch'}
        splits=[{'fold':i,'gan_fit_ids':ids[:18],'gan_reference_ids':ids[18:24],
            'outer_validation_ids':ids[24:]} for i in range(1,6)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            study=SimpleNamespace(run=root/'run',cache=root/'cache',signature='synthetic-only',protocol=p,
                records=[{'hospital_id':hid,'label':y} for hid,y in zip(ids,labels)],folds=splits,classes=['A','B','C'],
                inspection={'embedded_model_config':{}})
            with patch('research_v10.experiment.open_features',return_value=(patients,windows,arrays,'synthetic-cache')):
                benchmark(study,torch.device('cpu'),[1],p['probe_arms'],[42])
                # Identical paired representations must reproduce scores exactly.
                a=pd.read_csv(study.run/'cases/random_probe/fold_1/outer_seed_42/predictions.csv')
                b=pd.read_csv(study.run/'cases/pretrained_probe/fold_1/outer_seed_42/predictions.csv')
                pd.testing.assert_frame_equal(a,b)
                before=len(list((study.run/'cases').glob('*/*/*/completed.json')))
                benchmark(study,torch.device('cpu'),[1],p['probe_arms'],[42])
                self.assertEqual(before,len(list((study.run/'cases').glob('*/*/*/completed.json'))))
                with patch('research_v10.experiment.load_encoder',return_value=encoder):
                    benchmark(study,torch.device('cpu'),[1],['pretrained_tail'],[42])
                with self.assertRaises(ValueError):benchmark(study,torch.device('cpu'),[1,2],['pretrained_tail'],[42])
            audit=analyze(study)
            self.assertEqual(audit['inner_cases'],27);self.assertEqual(audit['outer_cases'],3)
            self.assertFalse(audit['full_probe_benchmark_complete'])
            self.assertEqual(audit['inner_choices_verified'],3)


if __name__=='__main__':unittest.main()
