"""New objective gradients, native spectral definition, isolation and exact resume."""
from copy import deepcopy
from pathlib import Path
import ast
import tempfile
import unittest
from unittest.mock import patch
import nbformat
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import welch
from torch.utils.data import DataLoader
from src.artifacts import cpu_weights, seed_everything, sha256_file, read_json
from research_v6 import experiment as old_experiment
from research_v6.common import Study as OldStudy
from research_v6.models import Generator as OldGenerator, Discriminator as OldDiscriminator
from research_v6.quality import BANDS, weighted_quantile
from research_v7.common import Study, validate_protocol, require_prepared, require_quality
from research_v7.data import BalancedPatients
from research_v7.models import Generator, Discriminator
from research_v7.fidelity import FAMILIES, FidelityLoss, build_bank, empirical_quantiles, relative_welch, statistics
from research_v7.training import fit
from research_v7 import experiment

ROOT = Path(__file__).resolve().parents[1]
P = yaml.safe_load((ROOT / 'research_v7/protocol.yaml').read_text())
OLD = yaml.safe_load((ROOT / 'research_v6/protocol.yaml').read_text())


class ResearchV7Tests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_architecture_and_protocol_control_same_as_v6(self):
        self.assertIs(Generator, OldGenerator); self.assertIs(Discriminator, OldDiscriminator)
        validate_protocol(P, OLD)
        for section, key, value in [('training','epochs',200),('training','batch_size',32),
                                   ('quality_gate','normalized_wasserstein_floor',2.)]:
            changed = deepcopy(P); changed[section][key] = value
            with self.assertRaises(ValueError): validate_protocol(changed, OLD)
        changed = deepcopy(P); changed['fidelity']['backend'] = 'mps'
        with self.assertRaises(ValueError): validate_protocol(changed, OLD)

    def test_native_welch_matches_scipy(self):
        X = np.random.default_rng(10).normal(size=(2,12,5000)).astype('float32')
        f, psd = welch(X,fs=2000,nperseg=2000,noverlap=1000,axis=-1)
        expected = np.stack([psd[..., (f >= lo) & ((f <= hi) if hi == 1000 else (f < hi))].sum(-1)
            / psd[..., f >= .5].sum(-1) for lo,hi in BANDS],axis=-1)
        actual = relative_welch(torch.from_numpy(X),2000).numpy()
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-7)

    def test_midpoint_quantiles_match_weighted_target_definition(self):
        value = np.array([[-3.,2.],[-1.,3.],[1.,5.],[3.,9.]],dtype='float32')
        levels = torch.tensor(P['fidelity']['quantiles'])
        expected = [[weighted_quantile(value[:,i],np.ones(4),q) for i in range(2)] for q in levels.tolist()]
        np.testing.assert_allclose(empirical_quantiles(torch.from_numpy(value),levels).numpy(),expected,atol=1e-6)

    def test_each_family_backpropagates_without_detaching_and_noise_increases_spectral_loss(self):
        t = np.arange(5000)/2000
        values = np.stack([np.stack([(1+.1*i)*np.sin(2*np.pi*(6+c)*t + .05*j)
            for j in range(12)]) for c in range(3) for i in range(4)]).astype('float32')
        rows = pd.DataFrame({'hospital_id':[f'p{i}' for i in range(12)],'segment_index':range(12),
            'label':np.repeat(range(3),4),'weight':1.})
        objective = FidelityLoss(build_bank(values,rows,1.,P['fidelity']),P['fidelity'])
        dirty = values + .10*np.sin(2*np.pi*150*t)[None,None,:].astype('float32')
        clean_loss, clean = objective(torch.tensor(values),torch.tensor(rows.label.to_numpy()))
        noisy = torch.tensor(dirty,requires_grad=True)
        total, pieces = objective(noisy,torch.tensor(rows.label.to_numpy()))
        self.assertGreater(float(pieces['spectrum'].detach()),float(clean['spectrum'])+.1)
        for family in FAMILIES:
            gradient = torch.autograd.grad(pieces[family],noisy,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.abs().sum()),0.)
        total.backward();self.assertTrue(torch.isfinite(noisy.grad).all())

    def test_target_bank_patient_balance_ignores_nonfit_signal_rows(self):
        X = np.random.default_rng(14).normal(size=(7,12,5000)).astype('float32')
        X[6] = np.nan
        rows = pd.DataFrame({'hospital_id':['a','a','a','b','c','d'],'segment_index':range(6),
            'label':[0,0,0,0,1,2],'weight':[1/3,1/3,1/3,1,1,1]})
        bank = build_bank(X,rows,1.,P['fidelity'])
        self.assertTrue(all(np.isfinite(value).all() for value in bank.values()))
        f = statistics(torch.from_numpy(X[:6]),P['fidelity'])['amplitude'].numpy()
        expected = weighted_quantile(f[:4,0],rows.weight.iloc[:4],.5)
        self.assertAlmostEqual(float(bank['amplitude_quantiles'][0,2,0]),expected,places=6)

    def test_epoch_resume_is_identical_with_new_cpu_fft_loss(self):
        cfg = {**P['training'],'epochs':2,'batch_size':3,'draws_per_fit_patient_per_epoch':1}
        settings = {**P['fidelity'],'spectral_window_samples':[32,64,128]}
        X = np.random.default_rng(17).normal(size=(12,12,128)).astype('float32')
        table = pd.DataFrame({'hospital_id':[f'p{i}' for i in range(12)],'label':[i%3 for i in range(12)]})
        rows = table.assign(segment_index=np.arange(12),weight=1.)
        bank = build_bank(X,rows,1.,settings)
        def setup():
            seed_everything(142)
            g,d = Generator(base=16,length=128),Discriminator()
            dataset = BalancedPatients(X,table,table.hospital_id.tolist(),1.,1,10142)
            loader = DataLoader(dataset,batch_size=3,generator=torch.Generator().manual_seed(11142))
            return g,d,loader,FidelityLoss(bank,settings)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            a,b = Path(temp)/'full',Path(temp)/'resume';a.mkdir();b.mkdir()
            g,d,loader,obj = setup();fit(g,d,loader,torch.device('cpu'),cfg,a,'case',142,obj)
            expected_g,expected_d = cpu_weights(g),cpu_weights(d)
            g,d,loader,obj = setup();fit(g,d,loader,torch.device('cpu'),cfg,b,'case',142,obj,stop_after=1)
            g,d,loader,obj = setup();fit(g,d,loader,torch.device('cpu'),cfg,b,'case',142,obj)
            for name,value in g.state_dict().items():torch.testing.assert_close(value,expected_g[name],atol=0,rtol=0)
            for name,value in d.state_dict().items():torch.testing.assert_close(value,expected_d[name],atol=0,rtol=0)
            pd.testing.assert_frame_equal(pd.read_csv(a/'history.csv').drop(columns='seconds'),pd.read_csv(b/'history.csv').drop(columns='seconds'))

    def test_failed_pilot_blocks_next_fold_before_data_access(self):
        study = Study(Path('/unused'),Path('/unused'),P,[],[],{},Path('/unused'))
        with patch('research_v7.experiment.require_smoke'),patch('research_v7.experiment.require_prepared'), \
                patch('research_v7.experiment.require_quality',side_effect=RuntimeError('FAIL')) as gate, \
                patch('research_v7.experiment.load_development') as load:
            with self.assertRaises(RuntimeError):experiment.train(study,2,torch.device('cpu'))
            gate.assert_called_once_with(study,1);load.assert_not_called()

    def test_native_synthetic_pipeline_and_frozen_v6_scaler_reuse(self):
        p = deepcopy(P);p['training'].update(epochs=1,batch_size=3,draws_per_fit_patient_per_epoch=1)
        p['generator']['base_channels'] = 16;p['generation']['per_class'] = 2;p['minimum_free_gib'] = 0
        old_p = deepcopy(OLD);old_p['minimum_free_gib'] = 0
        old_p['training'].update(epochs=1,batch_size=3,draws_per_fit_patient_per_epoch=1)
        old_p['generator']['base_channels'] = 16;old_p['generation']['per_class'] = 2
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            root = Path(temp);cache = root/'source/data/development';cache.mkdir(parents=True)
            ids = [f'p{i}' for i in range(12)]
            records = [{'hospital_id':hid,'label':i%3,'class_name':f'class{i%3}'} for i,hid in enumerate(ids)]
            table = pd.DataFrame({'hospital_id':ids,'label':[i%3 for i in range(12)]});table.to_csv(cache/'segments.csv',index=False)
            X = np.random.default_rng(18).normal(size=(12,12,5000)).astype('float32');X[9:] = np.nan;np.save(cache/'X.npy',X)
            fold = {'fold':1,'gan_fit_ids':ids[:6],'gan_reference_ids':ids[6:9],'outer_validation_ids':ids[9:]}
            old = OldStudy(root/'v6',root/'source',old_p,records,[fold],{'synthetic':'old'})
            old_experiment.prepare(old)
            old_experiment.smoke(old,torch.device('cpu'));old_experiment.train(old,1,torch.device('cpu'))
            old_experiment.generate(old,1,torch.device('cpu'));old_experiment.qc(old,1)
            new = Study(root/'v7',root/'source',p,records,[fold],{'synthetic':'new'},old.run)
            experiment.prepare(new);require_prepared(new,1)
            self.assertEqual(sha256_file(old.folder(1)/'amplitude_scale.json'),sha256_file(new.folder(1)/'amplitude_scale.json'))
            self.assertEqual(sha256_file(old.folder(1)/'reference_features.csv'),sha256_file(new.folder(1)/'reference_features.csv'))
            experiment.smoke(new,torch.device('cpu'))
            self.assertTrue(read_json(new.run/'smoke.json')['fidelity_gradient_bridge_passed'])
            experiment.train(new,1,torch.device('cpu'));generator_hash = sha256_file(new.folder(1)/'generator.pth')
            experiment.train(new,1,torch.device('cpu'));self.assertEqual(generator_hash,sha256_file(new.folder(1)/'generator.pth'))
            experiment.generate(new,1,torch.device('cpu'));experiment.qc(new,1)
            self.assertFalse(read_json(new.folder(1)/'quality.json')['passed'])
            with self.assertRaises(RuntimeError):require_quality(new,1)
            from research_v7.analysis import report
            report(new,1);self.assertTrue((new.folder(1)/'figures/waveforms_class_0.png').exists())
            self.assertEqual(len(pd.read_csv(new.folder(1)/'comparison_v6_v7.csv')),30)
            self.assertTrue((new.folder(1)/'figures/comparison_v6_v7.png').exists())

    def test_manual_notebooks_are_valid_unexecuted_python(self):
        paths = [*sorted((ROOT/'notebooks').glob('3[7-9]_*.ipynb')), *sorted((ROOT/'notebooks').glob('40_*.ipynb'))]
        self.assertEqual(len(paths),4)
        for path in paths:
            n = nbformat.read(path,as_version=4);nbformat.validate(n)
            for cell in n.cells:
                if cell.cell_type=='code':ast.parse(cell.source);self.assertIsNone(cell.execution_count);self.assertFalse(cell.outputs)


if __name__=='__main__':unittest.main()
