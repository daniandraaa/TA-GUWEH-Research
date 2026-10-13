import argparse
import json
from pathlib import Path
import numpy as np
from src.artifacts import ROOT,lock,write_json,sha256_file
from .common import context,DEFAULT_RUN,LEADS,validate_run_id


def synthetic_smoke(study):
    from .morphology import measure_record
    from .experiment import fit_pipeline
    fs=2000;t=np.arange(fs*8)/fs
    wave=sum(np.maximum(0,1-np.abs(t-center)/.055) for center in np.arange(1.,7.,.8))
    raw=np.column_stack([wave*(.4+j*.1)*(-1 if j in {3,6} else 1) for j in range(12)])
    features,quality,beats,_=measure_record(raw,fs,study.protocol)
    if not beats or not any(np.isfinite(v) for v in features.values()):raise RuntimeError('Synthetic candidate extraction failed')
    rng=np.random.default_rng(11);values=rng.normal(size=(45,9));labels=np.repeat([0,1,2],15);values[:,0]+=labels*2
    values[::7,3]=np.nan;model=fit_pipeline(values,labels,.1,study.protocol);prob=model.predict_proba(values)
    if prob.shape!=(45,3) or not np.isfinite(prob).all() or not np.allclose(prob.sum(1),1):raise RuntimeError('Synthetic CPU L2 head failed')
    result={'synthetic_candidate_count':len(beats),'synthetic_cpu_head_pass':True,'real_data_training':False,'real_waveform_inference':False}
    write_json(study.run/'synthetic_smoke.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description='V11 manual feature QA and regularized heads; no final/test endpoint')
    parser.add_argument('stage',choices=['prepare','smoke','ludb','extract','qa','pilot','cv','analyze','status'])
    parser.add_argument('--run-id',default=DEFAULT_RUN);parser.add_argument('--download',action='store_true')
    parser.add_argument('--arm',choices=['qrs_rr','morphology','morphology_rr','random_l2','pretrained_l2','fusion_random','fusion_pretrained'])
    args=parser.parse_args()
    try:validate_run_id(args.run_id)
    except ValueError as error:parser.error(str(error))
    if args.download and args.stage!='ludb':parser.error('--download only applies to the manual LUDB stage')
    # Only this new version owns a lock; no parent context/lock is entered.
    with lock(ROOT/'results/research_v11'/args.run_id/'.endpoint.lock'):
        study=context(args.run_id,initialize=args.stage=='prepare')
        if args.stage=='prepare':
            raw_files=[ROOT/study.protocol['raw_directory']/f"{r['hospital_id']}.csv" for r in study.records]
            if not all(q.is_file() for q in raw_files):raise RuntimeError('Raw continuous development files missing')
            from .benchmark import wfdb_module
            wfdb=wfdb_module()
            result={'patients':230,'classes':study.classes,'folds':5,'arms':study.protocol['arms'],'inner_scenarios_per_arm_fold':9,
                'expected_inner_fits':315,'expected_outer_refits':35,'deterministic_cpu_heads':True,'wfdb_version':wfdb.__version__,
                'real_training_performed':False,'real_waveform_inference_performed':False,'test_enabled':False,'final_enabled':False}
            write_json(study.run/'preparation.json',result);print(json.dumps(result,indent=2),flush=True)
        elif args.stage=='smoke':synthetic_smoke(study)
        elif args.stage=='ludb':
            from .benchmark import run
            run(study,args.download)
        elif args.stage in {'extract','qa'}:
            from .features import extract,qa
            (extract if args.stage=='extract' else qa)(study)
        elif args.stage in {'pilot','cv'}:
            from .experiment import run
            run(study,[1] if args.stage=='pilot' else study.protocol['folds'],[args.arm] if args.arm else study.protocol['arms'])
        elif args.stage=='analyze':
            from .analysis import analyze
            analyze(study)
        else:
            result={'experiment_hash':study.signature,'completed_cases':len(list((study.run/'cases').glob('*/*/*/completed.json'))),
                'expected_cases':350,'ludb_complete':(study.run/'ludb/completed.json').exists(),'features_complete':(study.run/'features/completed.json').exists(),
                'test_enabled':False,'final_enabled':False}
            print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
