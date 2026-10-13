"""Manual endpoints using frozen algorithms and repaired input dependencies."""
import argparse
import json
from unittest.mock import patch
from src.artifacts import ROOT,lock,write_json
from research_v11.common import validate_run_id
from .common import context,cache_status,load_bundle,DEFAULT_RUN


def main():
    parser=argparse.ArgumentParser(description='V11 dependency repair; no automatic training or encoder regeneration')
    parser.add_argument('stage',choices=['prepare','smoke','ludb','extract','qa','pilot','cv','analyze','status'])
    parser.add_argument('--run-id',default=DEFAULT_RUN);parser.add_argument('--download',action='store_true')
    parser.add_argument('--arm',choices=['qrs_rr','morphology','morphology_rr','random_l2','pretrained_l2','fusion_random','fusion_pretrained'])
    args=parser.parse_args()
    try:validate_run_id(args.run_id)
    except ValueError as error:parser.error(str(error))
    if args.download and args.stage!='ludb':parser.error('--download only applies to LUDB')
    with lock(ROOT/'results/research_v11'/args.run_id/'.endpoint.lock'):
        study=context(args.run_id,initialize=args.stage=='prepare')
        if args.stage=='prepare':
            if any(not (ROOT/study.protocol['raw_directory']/f"{r['hospital_id']}.csv").is_file() for r in study.records):raise RuntimeError('Raw development files missing')
            from research_v11.benchmark import wfdb_module
            result={'patients':230,'classes':study.classes,'folds':5,'arms':study.protocol['arms'],'expected_inner_fits':315,'expected_outer_refits':35,
                'wfdb_version':wfdb_module().__version__,'embedding_cache_status':cache_status(study),
                'cache_free_stages':['prepare','smoke','ludb','extract','qa','analyze','status'],
                'cache_free_feature_arms':['qrs_rr','morphology','morphology_rr'],'feature_arms_still_require_ludb_pass':True,
                'real_training_performed':False,'real_waveform_inference_performed':False,'test_enabled':False,'final_enabled':False}
            write_json(study.run/'preparation.json',result);print(json.dumps(result,indent=2),flush=True)
        elif args.stage=='smoke':
            from research_v11.__main__ import synthetic_smoke
            synthetic_smoke(study)
        elif args.stage=='ludb':
            from .benchmark import run
            run(study,args.download)
        elif args.stage in {'extract','qa'}:
            from research_v11.features import extract,qa
            (extract if args.stage=='extract' else qa)(study)
        elif args.stage in {'pilot','cv'}:
            from research_v11 import experiment
            # Only input loading changes; original fit/selection/outer-refit code remains frozen.
            with patch.object(experiment,'load_bundle',load_bundle):
                experiment.run(study,[1] if args.stage=='pilot' else study.protocol['folds'],[args.arm] if args.arm else study.protocol['arms'])
        elif args.stage=='analyze':
            from research_v11.analysis import analyze
            analyze(study)
        else:
            print(json.dumps({'experiment_hash':study.signature,'completed_cases':len(list((study.run/'cases').glob('*/*/*/completed.json'))),
                'expected_cases':350,'ludb_complete':(study.run/'ludb/completed.json').exists(),'features_complete':(study.run/'features/completed.json').exists(),
                'embedding_cache_status':cache_status(study),'test_enabled':False,'final_enabled':False},indent=2),flush=True)


if __name__=='__main__':main()
