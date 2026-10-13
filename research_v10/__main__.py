"""Explicit manual stage endpoints. No final/test endpoint."""
import argparse
import json
import shutil
from src.artifacts import choose_device, lock, write_json
from .common import DEFAULT_RUN, context, grid, storage_check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['download','compatibility','prepare','smoke','extract','pilot','cv','analyze','status'])
    parser.add_argument('--run-id',default=DEFAULT_RUN);parser.add_argument('--device',default='mps',choices=['cpu','mps','cuda','auto'])
    parser.add_argument('--arm',choices=['random_probe','pretrained_probe']);parser.add_argument('--fold',type=int,choices=[1,2,3,4,5])
    args=parser.parse_args()
    if args.stage=='download':
        from .assets import download
        download();return
    if args.stage=='compatibility':
        from .compatibility import inspect
        inspect();return
    study=context(args.run_id,initialize=args.stage=='prepare')
    with lock(study.run/'.v10_stage.lock'):
        if args.stage=='prepare':
            value={'experiment_hash':study.signature,'patients':len(study.records),'classes':study.classes,
                'head_scenarios':grid(study.protocol,'pretrained_probe'),'tail_scenarios':grid(study.protocol,'pretrained_tail'),
                'full_probe_plan':{'inner_fits':90,'outer_refits':30},'tail_pilot_plan':{'inner_fits':9,'outer_refits':1},
                'cache':str(study.cache),'storage':storage_check(study),'no_genuine_inference':True,'no_training':True,
                'final_enabled':False,'test_enabled':False}
            write_json(study.run/'preparation.json',value);print(json.dumps(value,indent=2))
        elif args.stage=='status':
            from .features import feature_paths
            print(json.dumps({'run':str(study.run),'experiment_hash':study.signature,
                'feature_cache_complete':(feature_paths(study)/'completed.json').exists(),
                'completed_cases':len(list((study.run/'cases').glob('*/*/*/completed.json'))),
                'expected_total_cases':130,'free_gib':shutil.disk_usage(study.run).free/1024**3,
                'minimum_reserve_gib':study.protocol['minimum_free_gib'],'final_enabled':False,'test_enabled':False},indent=2))
        elif args.stage=='analyze':
            from .analysis import analyze
            analyze(study)
        else:
            device=choose_device(args.device)
            if args.stage=='extract':
                from .features import extract
                extract(study,device)
            elif args.stage=='smoke':
                from .experiment import smoke
                smoke(study,device)
            else:
                from .experiment import benchmark
                if args.stage=='pilot':
                    benchmark(study,device,[1],study.protocol['probe_arms'],[42])
                    benchmark(study,device,[1],['pretrained_tail'],[42])
                else:
                    benchmark(study,device,[args.fold] if args.fold else study.protocol['folds'],
                        [args.arm] if args.arm else study.protocol['probe_arms'],study.protocol['outer_seeds'])


if __name__=='__main__':main()
