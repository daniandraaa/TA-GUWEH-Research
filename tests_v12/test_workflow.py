"""Exercise case resume, selection, audit and external gate on synthetic arrays."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.artifacts import write_json,read_json,canonical_hash
from research_v12.common import Study,verified,complete
from research_v12.data import group_folds,extraction
from research_v12 import experiment,analysis
from research_v12.local import export
from test_v12 import P,toy_block


class WorkflowTests(unittest.TestCase):
    def test_external_source_cannot_open_before_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Study(Path(directory),copy.deepcopy(P),{'synthetic_only':True})
            with patch('research_v12.data.metadata') as reader:
                with self.assertRaisesRegex(RuntimeError,'Prepare dan split'):
                    extraction(study,'mitdb',download=True)
                reader.assert_not_called()
                self.assertFalse((study.run/'external_source.json').exists())

    def test_complete_synthetic_CV_resume_and_external_evaluation(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            self.addCleanup(patch.stopall)
            patch.dict(os.environ,{'MPLCONFIGDIR':str(Path(directory)/'matplotlib')}).start()
            p=copy.deepcopy(P);p['training_cap_per_group_class']=8
            study=Study(Path(directory),p,{'synthetic_workflow':True})
            blocks=[]
            for i in range(20):
                labels=np.tile([0,1,2],8)
                b=toy_block(f'r{i:02}',f'g{i:02}',labels)
                b['features'][:,:,0]+=labels[:,None]*12
                blocks.append(b)
            y=np.concatenate([b['labels'] for b in blocks]);g=np.concatenate([np.repeat(b['group'],len(b['labels'])) for b in blocks])
            saved={'groups':sorted(set(g)),'folds':group_folds(y,g,p)}
            write_json(study.run/'data/incart.json',saved)
            complete(study.run/'data',{'experiment_hash':study.signature},['incart.json'])
            write_json(study.run/'features/incart/completed.json',{'synthetic_only':True})
            with patch.object(experiment,'load_bundle',return_value=(blocks,saved)),patch.object(analysis,'load_bundle',return_value=(blocks,saved)):
                experiment.run(study,list(range(1,6)),p['arms'])
                paths=sorted((study.run/'cases').glob('*/*/*/completed.json'))
                self.assertEqual(len(paths),40)
                timestamps={str(q):q.stat().st_mtime_ns for q in paths}
                experiment.run(study,[1],p['arms'])
                self.assertEqual(timestamps,{str(q):q.stat().st_mtime_ns for q in paths})
                analysis.analyze(study)
                audit=read_json(study.run/'analysis/audit.json')
                self.assertEqual(audit['fit_cases_verified'],40)
                self.assertTrue(audit['all_saved_evaluation_metrics_reproduced'])
                experiment.deploy(study)
                self.assertEqual(experiment.require_deployment(study)['arm'],read_json(study.run/'deployment/choice.json')['arm'])
                # A self-consistent manifest for a different split still must
                # fail verification against the frozen preparation manifest.
                target=study.run/'cases/morphology/fold_1/outer/completed.json'
                original=target.read_bytes();changed=read_json(target)
                changed['identity']['training_groups']=['wrong-source-group']
                payload={k:v for k,v in changed.items() if k!='manifest_hash'}
                write_json(target,{**payload,'manifest_hash':canonical_hash(payload)})
                with self.assertRaisesRegex(RuntimeError,'Identitas tahap berubah'):
                    experiment.completed_choices(study)
                target.write_bytes(original)
                # Once external evaluation opens, an incomplete fit cannot be
                # restarted within the same study; completed fits can resume.
                write_json(study.run/'external_source.json',{'synthetic_only':True})
                split=saved['folds'][0]
                with self.assertRaisesRegex(RuntimeError,'dibekukan'):
                    experiment.fit_case(study,blocks,'morphology',split['inner_training_groups'],
                        split['inner_validation_groups'],.1,[.5],study.run/'new_fit','inner')
            external=[]
            for i in range(47):
                labels=np.tile([0,1,2],3);b=toy_block(f'ex{i:02}',f'exg{i:02}',labels,leads=1)
                b['features'][:,:,0]+=labels[:,None]*12;external.append(b)
            write_json(study.run/'features/mitdb/completed.json',{'synthetic_only':True})
            with patch.object(experiment,'load_bundle',return_value=(external,{'groups':[b['group'] for b in external]})),patch.object(analysis,'load_bundle',return_value=(external,{})):
                experiment.external_score(study);analysis.external_analysis(study)
            result=read_json(study.run/'external_evaluation/summary.json')
            self.assertFalse(result['threshold_tuned_externally'])
            self.assertFalse(result['anatomical_origin_validation'])
            self.assertTrue(read_json(study.run/'external_evaluation/analysis.json')['all_saved_metrics_reproduced'])
            # The ordinary verification checks exact saved artifacts, not a stub.
            self.assertIsNotNone(verified(study.run/'deployment'))
            # Preserve a failed gate test rather than using fake local waveform data.
            with patch('research_v12.local.verified',return_value={'fake_manifest':True}),patch('research_v12.local.read_json',side_effect=[{'choice':{'arm':'morphology'},'engineering_gate_pass':False},{'arm':'morphology'}]):
                with self.assertRaisesRegex(RuntimeError,'Gate eksternal FAIL'):export(study)


if __name__=='__main__':unittest.main()
