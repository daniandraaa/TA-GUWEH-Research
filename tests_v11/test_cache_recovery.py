"""Synthetic/read-only fixtures for missing-cache repair, no ECG inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from src.artifacts import write_json,sha256_file,read_json
from research_v11.morphology import columns_for
from research_v11_recovery.common import verify_cached_files,cache_status,load_bundle
from research_v11_recovery.benchmark import annotation_qrs,reference_valid,evaluate_lead
from research_v11_recovery.benchmark import run as run_benchmark
from research_v11.common import verified
import yaml

P=yaml.safe_load((Path(__file__).resolve().parents[1]/'research_v11/protocol.yaml').read_text())


class CacheRecoveryTests(unittest.TestCase):
    def test_annotation_audit_written_without_waveform_benchmark(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);study=SimpleNamespace(run=root,signature='synthetic_only')
            def fake_benchmark(*args,**kwargs):
                folder=root/'ludb';folder.mkdir(exist_ok=True)
                pd.DataFrame({'mode':['reference_anchor','reference_anchor','detected_anchor','detected_anchor'],
                    'reference_boundary_available':[True,False,True,False],
                    'reference_boundary_issue':[None,'missing_reference_boundary',None,'missing_reference_boundary']}).to_csv(folder/'boundary_errors.csv',index=False)
                write_json(folder/'completed.json',{'synthetic':True});return {'passed':False}
            with patch('research_v11_recovery.benchmark.frozen.run',side_effect=fake_benchmark):
                run_benchmark(study)
            audit_dir=root/'ludb/annotation_audit';report=read_json(audit_dir/'annotation_integrity.json')
            self.assertEqual(report['annotated_qrs_retained'],2);self.assertEqual(report['unavailable_reference_qrs'],1)
            self.assertTrue(report['unavailable_reference_qrs_retained_in_boundary_coverage_denominator'])
            identity={'experiment_hash':'synthetic_only','benchmark_manifest_sha256':report['benchmark_manifest_sha256']}
            self.assertIsNotNone(verified(audit_dir,identity))

    def test_missing_or_inconsistent_annotations_keep_all_qrs_peaks(self):
        samples=[500,530,800,815,830,1000,1001,1030,1300,1320]
        symbols=['N',')','(','p',')','N','(',')','N',')']
        rows=annotation_qrs(samples,symbols)
        self.assertEqual(rows,[(None,500,530),(None,1000,None),(None,1300,1320)])
        self.assertFalse(any(reference_valid(row) for row in rows))
        malformed=annotation_qrs([499,500,530],['(','N',')'])
        self.assertTrue(reference_valid(malformed[0]))

    def test_annotation_failure_stays_in_detection_and_boundary_denominators(self):
        e=np.zeros(2000);e[470:530]=1;e[970:1030]=1
        score,errors=evaluate_lead(e,[(470,500,530),(None,1000,1030)],[500,1000],500,P)
        self.assertEqual((score['tp'],score['fp'],score['fn'],score['reference_qrs']),(2,0,0,2))
        self.assertEqual(score['reference_complete_qrs'],1)
        rows=[r for r in errors if r['mode']=='detected_anchor']
        self.assertEqual(sum(r['valid_boundary'] for r in rows),1)
        self.assertTrue(rows[1]['estimated_boundary_available'])
        self.assertNotIn('duration_error_ms',rows[1])

    def test_missing_embeddings_do_not_block_morphology_input_loading(self):
        records=[{'hospital_id':str(i),'label':i%3} for i in range(6)]
        frame=pd.DataFrame({name:np.arange(6,dtype=float) for name in columns_for('morphology_rr')})
        frame['II_width_ms_median']=np.nan
        study=SimpleNamespace(records=records,protocol={'arms':['qrs_rr','morphology','morphology_rr']},identity={'source_v10_hash':'frozen'})
        with patch('research_v11_recovery.common.verify_cached_files',side_effect=AssertionError('unrelated cache must not be opened')), \
             patch('research_v11.benchmark.require_pass',return_value='gate_pass') as gate, \
             patch('research_v11.features.read_features',return_value=(frame,'feature_sha')):
            patients,matrices,sources=load_bundle(study,['qrs_rr','morphology','morphology_rr'])
        self.assertEqual(len(patients),6);self.assertEqual(matrices['qrs_rr'].shape,(6,32))
        self.assertEqual(matrices['morphology'].shape,(6,126));self.assertEqual(matrices['morphology_rr'].shape,(6,132))
        self.assertTrue(np.isnan(matrices['qrs_rr'][:,2]).any() or np.isnan(matrices['qrs_rr']).any())
        self.assertNotIn('verified_used_v10_file_sha256',sources);gate.assert_called_once()

    def test_missing_cache_cannot_skip_ludb_failure(self):
        study=SimpleNamespace(records=[{'hospital_id':'1','label':0}],protocol={'arms':['qrs_rr']},identity={'source_v10_hash':'frozen'})
        with patch('research_v11.benchmark.require_pass',side_effect=RuntimeError('gate FAIL')), \
             patch('research_v11.features.read_features',side_effect=AssertionError('must not read before gate')):
            with self.assertRaisesRegex(RuntimeError,'gate FAIL'):load_bundle(study,['qrs_rr'])

    def test_audited_backup_subset_and_corruption(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);cache=root/'backup';cache.mkdir();patient=cache/'patients.csv';patient.write_text('hospital_id,label\n1,0\n')
            a=np.zeros((230,768),dtype=np.float32);np.save(cache/'random_probe.npy',a)
            audit={'feature_files_sha256':{str(root/'original'/p.name):sha256_file(p) for p in cache.iterdir()}}
            parent=root/'parent';write_json(parent/'analysis_20261006_complete/audit.json',audit)
            study=SimpleNamespace(parent_run=parent,feature_folder=cache)
            used=verify_cached_files(study,['patients.csv','random_probe.npy']);self.assertEqual(len(used),2)
            self.assertFalse((cache/'completed.json').exists())
            state=cache_status(study);self.assertTrue(state['arms']['random_l2']['ready']);self.assertFalse(state['arms']['pretrained_l2']['ready'])
            patient.write_text('hospital_id,label\n1,2\n')
            with self.assertRaisesRegex(RuntimeError,'berbeda dari audit'):verify_cached_files(study,['patients.csv'])

    def test_encoder_inputs_fail_before_any_fit_when_missing(self):
        study=SimpleNamespace(records=[{'hospital_id':'1','label':0}],protocol={'arms':['random_l2']},identity={'source_v10_hash':'frozen'})
        with patch('research_v11_recovery.common.verify_cached_files',side_effect=FileNotFoundError('missing cache')), \
             patch('research_v11.experiment.fit_pipeline',side_effect=AssertionError('no real training')):
            with self.assertRaisesRegex(FileNotFoundError,'missing cache'):load_bundle(study,['random_l2'])


if __name__=='__main__':unittest.main()
