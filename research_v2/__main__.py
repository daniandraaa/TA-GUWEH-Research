"""Manual endpoints: python -m research_v2 <stage>."""
import argparse
import torch
from .common import DEFAULT_RUN, context, verify_content
from .data import prepare_pool
from .experiment import fit, summarize, train_finals, evaluate
from .reporting import audit, final_report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'audit', 'verify', 'preprocess', 'cv', 'summary', 'select', 'final', 'test', 'report'])
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--arm', choices=['real_only', 'weighted_real'])
    parser.add_argument('--model', choices=['encoder', 'patchtst', 'hierarchical'])
    parser.add_argument('--scenario', help='Optional: one scenario ID, e.g. S01')
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    args = parser.parse_args()
    run, source, cfg, config, scenarios = context(args.run_id, initialize=args.stage == 'prepare')
    device = torch.device(args.device)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS tidak tersedia. Gunakan device cpu.')
    if args.stage == 'prepare':
        print(f'Prepared {run}; training belum dijalankan.')
    elif args.stage == 'audit':
        print(audit(run, source, cfg, config).to_string(index=False))
    elif args.stage == 'verify':
        verify_content(run); print('All frozen input checksums verified.')
    elif args.stage == 'preprocess':
        prepare_pool(run, source, cfg, config)
    elif args.stage == 'cv':
        if not args.model or not args.arm:
            parser.error('cv requires --model and --arm')
        if args.scenario and args.scenario not in {s['id'] for s in scenarios}:
            parser.error('Unknown scenario')
        for scenario in scenarios:
            if args.scenario and scenario['id'] != args.scenario:
                continue
            for fold in ([args.fold] if args.fold else range(1, config['n_folds'] + 1)):
                fit(run, cfg, config, args.model, args.arm, scenario, device, fold=fold)
    elif args.stage in {'summary', 'select'}:
        print(summarize(run, cfg, config, scenarios, select=args.stage == 'select').to_string(index=False))
    elif args.stage == 'final':
        if not args.model:
            parser.error('final requires --model')
        train_finals(run, cfg, config, scenarios, args.model, device)
    elif args.stage == 'test':
        evaluate(run, source, cfg, config, device)
    elif args.stage == 'report':
        print(final_report(run, cfg, config).to_string(index=False))


if __name__ == '__main__':
    main()
