"""Run one manual development stage; final/test stages intentionally absent."""
import argparse
import json
from src.artifacts import choose_device
from .common import DEFAULT_RUN, context, progress


def main():
    parser = argparse.ArgumentParser(description='V5 K4/K8 development ablation; 30 new fits, verified V4 controls.')
    parser.add_argument('stage', choices=['prepare', 'status', 'smoke', 'pilot', 'cv', 'report'])
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--model', choices=['all', 'hierarchical', 'encoder'], default='all')
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    parser.add_argument('--seed', type=int, choices=[42, 43, 44])
    args = parser.parse_args()
    if args.stage not in {'cv', 'status'} and (args.fold is not None or args.seed is not None):
        parser.error('--fold/--seed are available only for cv/status; pilot is fixed fold1 seed42')
    study = context(args.run_id, initialize=args.stage == 'prepare')
    models = study.protocol['models'] if args.model == 'all' else [args.model]
    if args.stage == 'status':
        p = progress(study)
        print(json.dumps({k: v for k, v in p.items() if k != 'missing'}, indent=2))
        print('First missing:', p['missing'][:5], flush=True)
    elif args.stage == 'report':
        from .analysis import analyze
        analyze(study)
    else:
        from . import experiment
        if args.stage == 'prepare':
            experiment.prepare(study)
        else:
            device = choose_device(args.device)
            if args.stage == 'smoke':
                experiment.smoke(study, models, device)
            else:
                experiment.train(study, models, [args.fold] if args.fold else study.protocol['folds'],
                                 [args.seed] if args.seed else study.protocol['seeds'], device,
                                 pilot=args.stage == 'pilot')


if __name__ == '__main__':
    main()
