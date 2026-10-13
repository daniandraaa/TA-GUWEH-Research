"""Independent manual endpoints for annotated beat identity (CPU)."""
import argparse
import json
import numpy as np
from src.artifacts import ROOT,lock,write_json
from .common import context,DEFAULT_RUN,valid_run


def synthetic_smoke(study):
    from .features import detect,extract_array,feature_names
    from .experiment import fit_pipeline
    p=study.protocol;fs=257;t=np.arange(fs*10)/fs
    wave=sum(np.exp(-((t-c)/.025)**2) for c in range(1,9))
    raw=np.column_stack([wave,-.7*wave])
    centers,_=detect(raw,fs,p);x=extract_array(raw,centers,fs,p)
    if not len(centers) or x.shape[2]!=len(feature_names(p,True)):
        raise RuntimeError('Synthetic feature/detector smoke failed')
    rng=np.random.default_rng(12);y=np.repeat([0,1,2],25);values=rng.normal(size=(len(y),len(feature_names(p,True))))
    values[:,0]+=y*3;values[::11,-1]=np.nan
    model=fit_pipeline(values,y,np.ones(len(y)),.1,p);prob=model.predict_proba(values)
    if prob.shape!=(75,3) or not np.isfinite(prob).all() or not np.allclose(prob.sum(1),1.):
        raise RuntimeError('Synthetic head smoke failed')
    result={'synthetic_only':True,'candidate_count':len(centers),'features_per_lead':x.shape[2],
        'three_class_head_pass':True,'real_training':False,'real_waveform_inference':False,'origin_training':False}
    write_json(study.run/'synthetic_smoke.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','smoke','extract','qa','pilot','cv','analyze','deploy','external-extract','external-score','external-analyze','local-pairs','status'])
    parser.add_argument('--run-id',default=DEFAULT_RUN)
    parser.add_argument('--arm',choices=['morphology','morphology_rr'])
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    try:valid_run(args.run_id)
    except ValueError as error:parser.error(str(error))
    if args.download and args.stage!='external-extract':
        parser.error('--download hanya untuk external-extract MIT-BIH')
    with lock(ROOT/'results/research_v12'/args.run_id/'.endpoint.lock'):
        study=context(args.run_id,initialize=args.stage=='prepare')
        if args.stage=='prepare':
            from .data import prepare
            prepare(study)
        elif args.stage=='smoke':synthetic_smoke(study)
        elif args.stage in ['extract','external-extract']:
            from .data import extraction
            extraction(study,'mitdb' if args.stage=='external-extract' else 'incart',args.download)
        elif args.stage=='qa':
            from .analysis import feature_qa
            feature_qa(study)
        elif args.stage in ['pilot','cv']:
            from .experiment import run
            run(study,[1] if args.stage=='pilot' else list(range(1,6)),[args.arm] if args.arm else study.protocol['arms'])
        elif args.stage=='analyze':
            from .analysis import analyze
            analyze(study)
        elif args.stage=='deploy':
            from .experiment import deploy
            deploy(study)
        elif args.stage=='external-score':
            from .experiment import external_score
            external_score(study)
        elif args.stage=='external-analyze':
            from .analysis import external_analysis
            external_analysis(study)
        elif args.stage=='local-pairs':
            from .local import export
            export(study)
        else:
            result={'run':str(study.run),'CV_completed_cases':len(list((study.run/'cases').glob('*/*/*/completed.json'))),
                'expected_CV_cases':40,'expected_pilot_cases':8,'INCART_features_ready':(study.run/'features/incart/completed.json').exists(),
                'deployment_ready':(study.run/'deployment/completed.json').exists(),
                'external_evaluation_ready':(study.run/'external_evaluation/completed.json').exists(),
                'local_candidates_ready':(study.run/'local_candidates/completed.json').exists(),
                'origin_training':False,'historical_origin_test':False}
            print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
