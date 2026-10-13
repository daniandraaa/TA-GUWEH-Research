import argparse
import json
from src.artifacts import choose_device
from .common import context, DEFAULT_RUN
from .data import prepare
from .experiment import fit, summarize, final, evaluate
from .classical import scenarios as tree_scenarios


def main():
    parser = argparse.ArgumentParser(description='V3 manual stages; real training is never started by prepare/status.')
    parser.add_argument('stage', choices=['prepare', 'status', 'cv', 'classical', 'select', 'final', 'test', 'report'])
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument('--model', choices=['encoder', 'patchtst', 'hierarchical', 'extratrees'])
    parser.add_argument('--arm')
    parser.add_argument('--scenario')
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    run, source, cfg, config, grid, v2cfg = context(args.run_id, initialize=args.stage == 'prepare')
    if args.stage == 'prepare':
        prepare(run, source, cfg, config, v2cfg)
    elif args.stage in {'status', 'select'}:
        frame = summarize(run, cfg, config, grid, select=args.stage == 'select')
        print(frame.to_string(index=False) if len(frame) else 'Belum ada CV lengkap.', flush=True)
    elif args.stage in {'cv', 'classical'}:
        if args.stage == 'cv' and args.model not in cfg['models']:
            parser.error('cv memerlukan --model encoder/patchtst/hierarchical')
        model = 'extratrees' if args.stage == 'classical' else args.model
        arms = cfg['classical_arms'] if model == 'extratrees' else cfg['arms']
        scenarios = tree_scenarios(cfg) if model == 'extratrees' else grid
        if args.arm and args.arm not in arms:
            parser.error('Unknown arm')
        if args.scenario and args.scenario not in {s['id'] for s in scenarios}:
            parser.error('Unknown scenario')
        if args.fold and not 1 <= args.fold <= config['n_folds']:
            parser.error('fold must be 1..5')
        device = choose_device('cpu' if model == 'extratrees' else args.device)
        for arm in ([args.arm] if args.arm else arms):
            for sc in [s for s in scenarios if not args.scenario or s['id'] == args.scenario]:
                for fold in ([args.fold] if args.fold else range(1, config['n_folds'] + 1)):
                    fit(run, source, cfg, config, model, arm, sc, device, fold=fold)
        summarize(run, cfg, config, grid)
    elif args.stage == 'final':
        if args.model is None:
            parser.error('final requires --model')
        final(run, source, cfg, config, grid, args.model, choose_device(args.device))
    elif args.stage == 'test':
        evaluate(run, source, cfg, config, v2cfg, choose_device(args.device))
    elif args.stage == 'report':
        from .reporting import make_report
        make_report(run, cfg, config, grid)


if __name__ == '__main__':
    main()
