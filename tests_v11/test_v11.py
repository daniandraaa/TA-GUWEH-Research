"""Synthetic CPU only: no local ECG loading, encoder inference or external download."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from src.artifacts import read_json,write_json,sha256_file
from research_v11.common import Study,complete,verified,validate_run_id
from research_v11.morphology import measure_record,delineate,columns_for
from research_v11.benchmark import annotation_qrs,gate_for,evaluate_lead
from research_v11.experiment import fit_pipeline,select,case
from research_v11.analysis import audit_case,audit_choices,paired_bootstrap,macro_f1,analyze

PROTOCOL=yaml.safe_load((Path(__file__).resolve().parents[1]/'research_v11/protocol.yaml').read_text())


def synthetic_ecg(fs):
    t=np.arange(fs*8)/fs
    wave=sum(np.maximum(0,1-np.abs(t-center)/.055) for center in np.arange(1.,7.,.8))
    return np.column_stack([wave*(.4+j*.1)*(-1 if j in {3,6} else 1) for j in range(12)])


class MeasurementTests(unittest.TestCase):
    def test_native_and_benchmark_rates_and_polarity(self):
        widths=[]
        for fs in [500,2000]:
            _,qa,beats,_=measure_record(synthetic_ecg(fs),fs,PROTOCOL)
            self.assertEqual(qa['complete_candidates'],8)
            self.assertGreaterEqual(qa['candidates_with_at_least_six_valid_leads'],6)
            for row in beats[1:-1]:
                self.assertAlmostEqual(row['rr_before_ms'],800,delta=5)
                self.assertAlmostEqual(row['rr_after_ms'],800,delta=5)
                self.assertAlmostEqual(row['II_width_ms'],row['aVR_width_ms'],delta=2)
                self.assertLess(row['II_onset_sample'],row['center_sample'])
                self.assertGreater(row['II_offset_sample'],row['center_sample'])
            widths.append(beats[3]['II_width_ms'])
        self.assertAlmostEqual(*widths,delta=5)

    def test_common_gain_preserves_relative_lead_information(self):
        raw=synthetic_ecg(500)
        a,_,_,_=measure_record(raw,500,PROTOCOL);b,_,_,_=measure_record(7*raw,500,PROTOCOL)
        np.testing.assert_allclose(list(a.values()),list(b.values()),rtol=1e-7,atol=1e-7,equal_nan=True)
        self.assertGreater(a['V6_r_relative_median'],a['II_r_relative_median'])

    def test_biphasic_valley_does_not_end_qrs(self):
        for fs in [500,2000]:
            envelope=np.zeros(fs*3);center=fs
            onset=center-round(.06*fs);offset=center+round(.06*fs)
            envelope[onset:offset]=1
            envelope[center-round(.008*fs):center+round(.008*fs)]=0
            left,right=delineate(envelope,center,fs,PROTOCOL)
            self.assertAlmostEqual(left/fs,.936,delta=1/fs)
            self.assertAlmostEqual(right/fs,1.064,delta=1/fs)

    def test_failed_measurements_remain_missing(self):
        features,qa,beats,_=measure_record(np.zeros((4000,12)),2000,PROTOCOL)
        self.assertEqual(beats,[]);self.assertEqual(qa['feature_missing_fraction'],1)
        self.assertTrue(np.isnan(list(features.values())).all())
        self.assertEqual([len(columns_for(arm)) for arm in ['qrs_rr','morphology','morphology_rr']],[32,126,132])


class BenchmarkTests(unittest.TestCase):
    def test_p_and_t_not_mistaken_for_qrs(self):
        samples=[10,20,30,40,50,70,100,120,140]
        symbols=['(','p',')','(','N',')','(','t',')']
        self.assertEqual(annotation_qrs(samples,symbols),[(40,50,70)])
        with self.assertRaises(RuntimeError):annotation_qrs([40,50],['(','N'])

    def test_missed_detection_is_in_boundary_coverage_denominator(self):
        envelope=np.zeros(1500);envelope[470:530]=1;envelope[970:1030]=1
        score,errors=evaluate_lead(envelope,[(470,500,530),(970,1000,1030)],[500],500,PROTOCOL)
        self.assertEqual((score['tp'],score['fn'],score['fp']),(1,1,0))
        detected=[r for r in errors if r['mode']=='detected_anchor']
        self.assertEqual(len(detected),2);self.assertEqual(sum(r['valid_boundary'] for r in detected),1)

    def test_oracle_pass_cannot_override_operational_failure(self):
        good={'coverage':1.,'median_onset_error_ms':1.,'median_offset_error_ms':1.,'median_duration_error_ms':2.}
        summary={'micro_detection_f1':1.,'boundaries':{'reference_anchor':good,'detected_anchor':{**good,'coverage':.5}}}
        checks,passed=gate_for(summary,PROTOCOL)
        self.assertTrue(checks['reference_coverage']);self.assertFalse(passed)


class FitAndProvenanceTests(unittest.TestCase):
    def test_training_only_imputation_and_scaling(self):
        x=np.tile([[1.,np.nan,np.nan],[3.,5.,np.nan],[5.,7.,np.nan]],(5,1));y=np.tile([0,1,2],5)
        model=fit_pipeline(x,y,.1,PROTOCOL)
        np.testing.assert_allclose(model['imputer'].statistics_,[3.,6.,0.])
        means=model['scaler'].mean_.copy()
        extreme=np.array([[1e8,-1e8,4.]])
        probability=model.predict_proba(extreme)
        np.testing.assert_array_equal(model['scaler'].mean_,means)
        self.assertTrue(np.isfinite(probability).all());self.assertAlmostEqual(probability.sum(),1.)

    def test_smaller_c_tiebreak_and_all_scenarios(self):
        rows=[{'C':c,'evaluation_f1':.5} for c in PROTOCOL['inverse_regularization_C']]
        self.assertEqual(select(rows)['C'],1e-6)
        with self.assertRaises(RuntimeError):select(rows[:-1])

    def test_manifest_tamper_and_path_guard(self):
        for name in ['../escape','a/b','a\\b','.','..','']:
            with self.assertRaises(ValueError):validate_run_id(name)
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);(folder/'value.txt').write_text('original')
            complete(folder,{'synthetic':True},['value.txt']);self.assertIsNotNone(verified(folder,{'synthetic':True}))
            (folder/'value.txt').write_text('tampered')
            with self.assertRaises(RuntimeError):verified(folder,{'synthetic':True})

    def test_synthetic_nine_inner_then_fresh_outer_and_resume(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);rng=np.random.default_rng(17);labels=np.tile([0,1,2],30)
            patients=pd.DataFrame({'hospital_id':[f'{i:03}' for i in range(90)],'label':labels})
            values=rng.normal(size=(90,6));values[:,0]+=labels*3;values[::7,2]=np.nan
            split={'gan_fit_ids':patients.hospital_id[:45].tolist(),'gan_reference_ids':patients.hospital_id[45:66].tolist(),
                   'outer_validation_ids':patients.hospital_id[66:].tolist()}
            study=Study(folder,copy.deepcopy(PROTOCOL),patients.to_dict('records'),[split],['a','b','c'],folder,folder,{'synthetic':True})
            with patch('research_v11.experiment.disk_check'):
                results=[case(study,'random_l2',1,c,'inner',patients,values,{'synthetic':True}) for c in PROTOCOL['inverse_regularization_C']]
                best=select(results);choice=folder/'choices/random_l2/fold_1.json'
                write_json(choice,{'experiment_hash':study.signature,'arm':'random_l2','fold':1,'C':best['C'],'inner_f1':best['evaluation_f1'],
                    'inner_completed_sha256':{f'S{i:02}':sha256_file(folder/f'cases/random_l2/fold_1/inner_S{i:02}/completed.json') for i in range(1,10)}})
                case(study,'random_l2',1,best['C'],'outer',patients,values,{'synthetic':True},choice)
                outer=folder/'cases/random_l2/fold_1/outer';identity,result,_=audit_case(study,outer);audit_choices(study,identity)
                self.assertEqual(identity['training_ids'],patients.hospital_id[:66].tolist())
                self.assertEqual(result['evaluation_patients'],24)
                with patch('research_v11.experiment.fit_pipeline',side_effect=AssertionError('must not retrain')):
                    case(study,'random_l2',1,best['C'],'outer',patients,values,{'synthetic':True},choice)
                    analyze(study)
                audit=read_json(folder/'analysis/audit.json')
                self.assertEqual(audit['completed_training_cases'],10)
                self.assertTrue(audit['all_reconstructed_training_metrics_match'])
                self.assertEqual(audit['inner_choices_reproduced'],1)

    def test_paired_bootstrap_and_macro_f1(self):
        labels=np.tile([0,1,2],10);predicted=np.roll(labels,1)
        self.assertAlmostEqual(macro_f1(labels,predicted),f1_score(labels,predicted,average='macro'))
        frame=pd.DataFrame({'hospital_id':[str(i) for i in range(30)],'true_label':labels})
        for i,name in enumerate(['a','b','c']):frame['probability_'+name]=(labels==i).astype(float)
        result=paired_bootstrap(frame,frame.sample(frac=1,random_state=7),['a','b','c'],repeats=50)
        self.assertEqual(result['conditional_patient_bootstrap_ci95'],[0.,0.])


if __name__=='__main__':unittest.main()
