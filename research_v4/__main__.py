"""Run one phase/model at a time: python -m research_v4 --help."""
import argparse
import json
from src.artifacts import choose_device
from .common import DEFAULT_RUN, context, progress, scenarios
from . import experiment


def main():
    parser = argparse.ArgumentParser(description='V4 patient MIL research; manual training, historical test locked.')
    parser.add_argument('stage', choices=['prepare', 'status', 'diagnostic', 'pilot', 'cv', 'report', 'select', 'stability', 'final', 'test'])
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument('--model', choices=['all', 'encoder', 'patchtst', 'hierarchical', 'residual_multiscale', 'lead_time'], default='all')
    parser.add_argument('--pooling', choices=['all', 'mean', 'attention'], default='all')
    parser.add_argument('--scenario', choices=['all'] + [f'S{i:02}' for i in range(1, 10)], default='all')
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    parser.add_argument('--recipe', choices=['Adam', 'AdamW'], default='Adam', help='paired native diagnostic only')
    args = parser.parse_args()
    run, source, old, cfg, config = context(args.run_id, initialize=args.stage == 'prepare')
    models = cfg['models'] if args.model == 'all' else [args.model]
    pools = cfg['pooling'] if args.pooling == 'all' else [args.pooling]
    folds = range(1, 6) if args.fold is None else [args.fold]
    if args.stage == 'prepare':
        experiment.prepare(run, source, old, cfg, config)
    elif args.stage == 'status':
        p = progress(run, cfg)
        print(json.dumps({k: v for k, v in p.items() if k != 'missing'}, indent=2))
        print('First missing:', p['missing'][:5])
    elif args.stage in {'report', 'select'}:
        experiment.summarize(run, cfg, select=args.stage == 'select')
        from .analysis import analyze
        analyze(run, cfg, config)
    else:
        device = choose_device(args.device)
        if args.stage == 'diagnostic':
            if args.model == 'all':
                models = ['encoder', 'patchtst', 'hierarchical']
            if any(m not in {'encoder', 'patchtst', 'hierarchical'} for m in models):
                parser.error('Native V3 diagnosis uses only the three original models')
            experiment.diagnostic(run, source, old, cfg, config, device, models, folds, args.recipe)
        elif args.stage in {'pilot', 'cv'}:
            selected = (cfg['pilot_scenario'] if args.stage == 'pilot' else args.scenario)
            grid = [s for s in scenarios(cfg) if selected == 'all' or s['id'] == selected]
            experiment.cv(run, source, cfg, config, device, models, pools, grid, folds)
            experiment.summarize(run, cfg)
        elif args.stage == 'stability':
            experiment.stability(run, source, cfg, config, device, models)
        elif args.stage == 'final':
            experiment.final(run, source, cfg, config, device, models)
        elif args.stage == 'test':
            experiment.summarize(run, cfg, select=True)
            experiment.evaluate(run, source, cfg, config, device)
        if args.stage in {'stability', 'final', 'test', 'diagnostic'}:
            from .analysis import analyze
            analyze(run, cfg, config)


if __name__ == '__main__':
    main()
