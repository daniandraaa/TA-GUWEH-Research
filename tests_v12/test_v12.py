"""Synthetic scientific invariants. No real ECG processing, training or downloads."""
import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import yaml
from research_v12.labels import beat_class,annotation_rows,match_labels,pvc_counts,count_metrics
from research_v12.features import feature_names,choose_leads,detect,extract_array,single_features
from research_v12.data import group_folds,source_group
from research_v12.experiment import training_rows,fit_pipeline,probabilities,choose_score,deployment_choice
from research_v12.metrics import evaluate,pvc_decisions,group_bootstrap
from research_v12.local import selected_indices
from research_v12.common import complete,verified,valid_run
from src.artifacts import write_json

P=yaml.safe_load((Path(__file__).resolve().parents[1]/'research_v12/protocol.yaml').read_text())


def toy_block(record,group,labels,leads=2):
    n=len(labels);rng=np.random.default_rng(23)
    return {'record':record,'group':group,'labels':np.array(labels),'centers':np.arange(n)*257+257,
        'matched_reference':np.arange(n)*257+257,
        'features':rng.normal(size=(n,leads,len(feature_names(P,True)))).astype(np.float32),
        'reference':[(int((i+1)*257),int(label),'V' if label==1 else 'N') for i,label in enumerate(labels)]}


class LabelsTests(unittest.TestCase):
    def test_unknown_and_fusion_do_not_become_normal(self):
        self.assertEqual(beat_class('Q'),-1);self.assertEqual(beat_class('+'),-1)
        self.assertEqual(beat_class('F'),2);self.assertEqual(beat_class('/'),2)
        self.assertEqual(beat_class('E'),2);self.assertEqual(beat_class('r'),1)

    def test_edges_use_physical_duration_and_retain_unknown(self):
        refs,edge=annotation_rows([1,257,514,980],['V','N','Q','V'],1000,257,P)
        self.assertEqual(edge,2);self.assertEqual([r[1] for r in refs],[0,-1])

    def test_one_to_one_reference_matching(self):
        refs=[(100,1,'V')];labels,match=match_labels(refs,[99,101],100,.02)
        self.assertEqual(sum(labels==1),1);self.assertEqual(sum(labels==-2),1)

    def test_missed_detector_PVC_remains_false_negative(self):
        refs=[(100,1,'V'),(200,1,'V'),(300,0,'N')]
        counts=pvc_counts(refs,[1,0],[True,False])
        self.assertEqual((counts['tp'],counts['fn']),(1,1))
        self.assertAlmostEqual(count_metrics(counts)['recall'],.5)

    def test_unmatched_PVC_is_false_alarm_unknown_is_reported(self):
        counts=pvc_counts([(100,1,'V')],[1,-2,-1],[True,True,True])
        self.assertEqual(counts['fp'],1);self.assertEqual(counts['unknown_reference_predictions'],1)

    def test_duplicate_reference_predictions_rejected(self):
        with self.assertRaises(ValueError):pvc_counts([(100,1,'V')],[1,1],[True,True])


class GroupAndTrainingTests(unittest.TestCase):
    def test_all_cuts_keep_source_groups_disjoint_and_cover_once(self):
        groups=np.repeat([f'g{i:02}' for i in range(20)],30);y=np.tile(np.repeat([0,1,2],10),20)
        folds=group_folds(y,groups,P);seen=[]
        for s in folds:
            a,b,c=map(set,[s['inner_training_groups'],s['inner_validation_groups'],s['outer_validation_groups']])
            self.assertFalse(a&b);self.assertFalse((a|b)&c);self.assertEqual(a|b|c,set(groups));seen.extend(c)
        self.assertEqual(len(seen),20);self.assertEqual(len(set(seen)),20)

    def test_same_MIT_patient_group(self):
        header=SimpleNamespace(comments=[])
        self.assertEqual(source_group(header,'mitdb','201'),source_group(header,'mitdb','202'))
        self.assertNotEqual(source_group(header,'mitdb','200'),source_group(header,'mitdb','201'))

    def test_missing_INCART_patient_identity_rejected(self):
        with self.assertRaises(RuntimeError):source_group(SimpleNamespace(comments=[]),'incart','I01')

    def test_training_cap_weights_and_arm_pairing(self):
        p=copy.deepcopy(P);p['training_cap_per_group_class']=4
        blocks=[toy_block('a','g1',[0]*12+[1]*8+[2]*9),toy_block('b','g2',[0]*5+[1]*3+[2]*7,leads=1)]
        x,y,w,rows=training_rows(blocks,p,'morphology_rr')
        a,b,c,r2=training_rows(blocks,p,'morphology')
        self.assertTrue(rows.equals(r2));self.assertEqual(x.shape[1]-a.shape[1],3)
        self.assertTrue(np.allclose(w,c));self.assertAlmostEqual(w.mean(),1.)
        sums=rows.groupby('group').weight_raw.sum().to_numpy();self.assertTrue(np.allclose(sums,[.5,.5]))
        beats=rows.drop_duplicates(['record','center_sample']).groupby(['group','class']).size()
        self.assertLessEqual(beats.max(),4)

    def test_unknown_and_unmatched_never_enter_training(self):
        b=toy_block('x','g',[0,1,2,-1,-2]);_,y,_,rows=training_rows([b],P,'morphology')
        self.assertEqual(set(y),{0,1,2});self.assertEqual(len(rows),6)

    def test_three_class_imputer_scaler_fit_synthetic_only(self):
        rng=np.random.default_rng(9);y=np.repeat([0,1,2],20);x=rng.normal(size=(60,len(feature_names(P,True))))
        x[:,0]+=y*4;x[::7,-1]=np.nan
        model=fit_pipeline(x,y,np.ones(60),.1,P);prob=model.predict_proba(x)
        self.assertEqual(model.classes_.tolist(),[0,1,2]);self.assertTrue(np.allclose(prob.sum(1),1.))
        self.assertTrue(np.isfinite(prob).all())
        block=toy_block('x','g',[])
        self.assertEqual(probabilities(model,[block],P,'morphology_rr')[0].shape,(0,3))

    def test_selection_tie_breaks_and_deployment_use_inner_joint_modes(self):
        scores=[{'C':1.,'threshold':.5,'metrics':{'f1':.8,'precision':.9}},
            {'C':.1,'threshold':.7,'metrics':{'f1':.8,'precision':.9}}]
        self.assertEqual(choose_score(scores)['C'],.1)
        rows=[{'arm':a,'f1':.8,'C':.1,'threshold':.7} for a in P['arms'] for _ in range(5)]
        self.assertEqual(deployment_choice(rows,P['arms'])['arm'],'morphology')


class FeatureTests(unittest.TestCase):
    def test_actual_lead_names_not_relabelled_or_duplicated(self):
        self.assertEqual(choose_leads(['V1','MLII'],'mitdb',P),[1,0])
        self.assertEqual(choose_leads(['MLII','V5'],'mitdb',P),[0])
        self.assertEqual(choose_leads(['V5'],'mitdb',P),[0])

    def test_synthetic_detector_and_time_equivalent_features(self):
        vectors=[]
        for fs in [257,500]:
            t=np.arange(fs*9)/fs;wave=sum(np.exp(-((t-c)/.025)**2) for c in range(1,8))
            raw=np.column_stack([wave,-wave*.8]);centers,_=detect(raw,fs,P);x=extract_array(raw,centers,fs,P)
            self.assertGreater(len(centers),3);self.assertEqual(x.shape[1:],(2,len(feature_names(P,True))))
            self.assertTrue(np.isfinite(x[:,:,:-3]).all());vectors.append(x[len(x)//2,0,:64])
        self.assertGreater(np.corrcoef(vectors)[0,1],.98)

    def test_polarity_preserved_and_amplitude_scale_cancels(self):
        fs=257;t=np.arange(round(.8*fs))/fs;wave=np.exp(-((t-.3)/.025)**2)
        a=single_features(wave,fs,1.,P);b=single_features(wave*3,fs,3.,P)
        negative=single_features(-wave,fs,1.,P)
        self.assertTrue(np.allclose(a,b));self.assertTrue(np.allclose(a[:64],-negative[:64]))


class MetricAndExportTests(unittest.TestCase):
    def test_end_to_end_and_conditional_metrics_are_separate(self):
        b=toy_block('a','g',[0,1,2]);b['reference'].append((1300,1,'V'))
        prob=np.eye(3);m,rows=evaluate([b],[prob],.5)
        self.assertEqual(m['matched_known_macro_f1'],1.)
        self.assertEqual(m['missed_reference_PVC'],1);self.assertAlmostEqual(m['recall'],.5)

    def test_out_of_range_probability_rejected(self):
        with self.assertRaises(ValueError):pvc_decisions(np.array([[-.1,1.1,0.]]),.5)

    def test_group_bootstrap_one_group_is_fixed(self):
        rows=[{'group':'g','tp':8,'fp':2,'fn':2}]
        ci=group_bootstrap(rows,repeats=30);self.assertTrue(np.allclose(ci['f1'],[.8,.8]))

    def test_pairing_never_forces_missing_type_or_overlaps(self):
        prob=np.array([[.95,.03,.02],[.1,.8,.1],[.05,.1,.85]])
        pvc,normal=selected_indices(prob,.7,.9,4)
        self.assertEqual(pvc,[1]);self.assertEqual(normal,[0]);self.assertFalse(set(pvc)&set(normal))
        pvc,normal=selected_indices(np.array([[.05,.1,.85]]),.7,.9,4)
        self.assertEqual((pvc,normal),([],[]))

    def test_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);write_json(folder/'a.json',{'value':1})
            complete(folder,{'run':'toy'},['a.json']);self.assertIsNotNone(verified(folder,{'run':'toy'}))
            write_json(folder/'a.json',{'value':2})
            with self.assertRaises(RuntimeError):verified(folder)

    def test_invalid_run_path(self):
        for value in ['../x','.','a/b','a\\b','']:
            with self.assertRaises(ValueError):valid_run(value)


if __name__=='__main__':unittest.main()
