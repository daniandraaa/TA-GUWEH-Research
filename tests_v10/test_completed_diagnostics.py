"""Synthetic-only tests for new saved-output audits and inference diagnostics."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from src.artifacts import save_torch,sha256_file
from src.evaluation import compute_metrics,prediction_table
from research_v10.model import ECGFMEncoder,config,TailClassifier
from research_v10.training import FeatureClassifier,Patients,predict
from analysis_v10.audit_completed import macro_f1,paired_patient_bootstrap
from analysis_v10.manual_diagnostics import compare_arrays,replay_case,encode_patient


class DiagnosticTests(unittest.TestCase):
    def test_patient_bootstrap_paired_and_reproducible(self):
        truth=np.array([0,1,2]*10);predictions=np.stack([truth,truth,truth])
        p=np.eye(3)[truth]
        self.assertEqual(macro_f1(truth,truth),compute_metrics(truth,p)['f1'])
        first=paired_patient_bootstrap(truth,predictions,predictions,repetitions=30)
        second=paired_patient_bootstrap(truth,predictions,predictions,repetitions=30)
        self.assertEqual(first,second);self.assertEqual(first['observed_delta'],0.)
        self.assertEqual(first['ci95_lower'],0.);self.assertEqual(first['ci95_upper'],0.)

    def test_array_comparison_detects_failure(self):
        reference=np.arange(15,dtype='float32').reshape(3,5)
        self.assertTrue(compare_arrays(reference,reference)['within_tolerance'])
        self.assertFalse(compare_arrays(reference+1,reference)['within_tolerance'])
        with self.assertRaises(RuntimeError):compare_arrays(np.full((3,5),np.nan),reference)

    def test_frozen_head_saved_inference_replay(self):
        mean=np.zeros(5,dtype='float32');scale=np.ones(5,dtype='float32');model=FeatureClassifier(mean,scale)
        train=Patients([f't{i}' for i in range(12)],[0,1,2]*4,features=np.ones((12,5),dtype='float32'))
        val=Patients([f'v{i}' for i in range(6)],[0,1,2]*2,features=np.ones((6,5),dtype='float32'))
        ids,y,p=predict(model,val,torch.device('cpu'));classes=['A','B','C']
        stored=prediction_table(ids,y,p,classes)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'cases/test';folder.mkdir(parents=True)
            save_torch(folder/'selected.pth',model.state_dict());digest=sha256_file(folder/'selected.pth')
            study=SimpleNamespace(run=root,classes=classes)
            case={'arm':'random_probe','feature_cache_manifest':'synthetic','train_ids':train.ids,'evaluation_ids':val.ids,
                'fold':1,'seed':42,'inner_selection':True}
            result={'selected_epoch':1,'selected_checkpoint_sha256':digest,'f1':compute_metrics(y,p)['f1']}
            with patch('analysis_v10.manual_diagnostics.verify_case',return_value=(case,result,stored)),patch(
                'analysis_v10.manual_diagnostics.make_datasets',return_value=(train,val,mean,scale)):
                row,frames=replay_case(study,folder,torch.device('cpu'),(None,None,None,'synthetic'))
            self.assertTrue(row['replay_probabilities_within_tolerance']);self.assertTrue(row['replay_argmax_identical'])
            self.assertEqual(len(frames['train']),12);self.assertEqual(row['evaluation_macro_f1'],result['f1'])

    def test_tail_saved_inference_replay(self):
        cfg=config({'encoder_layers':2,'encoder_embed_dim':32,'encoder_ffn_embed_dim':64,'encoder_attention_heads':4,
            'conv_pos_groups':4,'dropout':0,'attention_dropout':0})
        encoder=ECGFMEncoder(cfg).eval().requires_grad_(False)
        mean=np.zeros(32,dtype='float32');scale=np.ones(32,dtype='float32')
        model=TailClassifier(encoder,mean,scale).eval()
        tokens=np.random.default_rng(22).normal(size=(18,5,32)).astype('float32')
        train=Patients([f't{i}' for i in range(12)],[0,1,2]*4,tokens=tokens,groups=[np.array([i]) for i in range(12)])
        val=Patients([f'v{i}' for i in range(6)],[0,1,2]*2,tokens=tokens,groups=[np.array([i]) for i in range(12,18)])
        ids,y,p=predict(model,val,torch.device('cpu'));classes=['A','B','C'];stored=prediction_table(ids,y,p,classes)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'cases/test';folder.mkdir(parents=True)
            save_torch(folder/'selected.pth',model.state_dict());digest=sha256_file(folder/'selected.pth')
            study=SimpleNamespace(run=root,classes=classes)
            case={'arm':'pretrained_tail','feature_cache_manifest':'synthetic','train_ids':train.ids,'evaluation_ids':val.ids,
                'fold':1,'seed':42,'inner_selection':False}
            result={'selected_epoch':1,'selected_checkpoint_sha256':digest,'f1':compute_metrics(y,p)['f1']}
            with patch('analysis_v10.manual_diagnostics.verify_case',return_value=(case,result,stored)),patch(
                'analysis_v10.manual_diagnostics.make_datasets',return_value=(train,val,mean,scale)):
                row,frames=replay_case(study,folder,torch.device('cpu'),(None,None,None,'synthetic'),encoder)
            self.assertTrue(row['replay_probabilities_within_tolerance']);self.assertTrue(row['replay_argmax_identical'])

    def test_encoder_pool_and_split_match_synthetic_cache(self):
        cfg=config({'encoder_layers':2,'encoder_embed_dim':32,'encoder_ffn_embed_dim':64,'encoder_attention_heads':4,
            'conv_pos_groups':4,'dropout':0,'attention_dropout':0})
        model=ECGFMEncoder(cfg).eval().requires_grad_(False)
        X=np.random.default_rng(2).normal(size=(5,12,5000)).astype('float32');indices=np.arange(len(X))
        pooled,tokens=encode_patient(model,X,indices,torch.device('cpu'),2,True)
        ordinary,_=encode_patient(model,X,indices,torch.device('cpu'),2,False)
        np.testing.assert_array_equal(pooled,ordinary)
        self.assertEqual(tokens.shape,(5,78,32))


if __name__=='__main__':unittest.main()
