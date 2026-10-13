import argparse
import json
from src.artifacts import choose_device
from .common import DEFAULT_RUN, context


def main():
    parser = argparse.ArgumentParser(description='Manual V7 GAN distribution objective; native development only')
    parser.add_argument('stage', choices=['prepare', 'smoke', 'train', 'generate', 'qc', 'report', 'status'])
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    args = parser.parse_args()
    if args.stage in {'train', 'generate', 'qc', 'report'} and args.fold is None:
        parser.error('--fold is required; start with --fold 1')
    if args.stage in {'prepare', 'smoke', 'status'} and args.fold is not None:
        parser.error('--fold is only available for train/generate/qc/report')
    study = context(args.run_id, initialize=args.stage == 'prepare')
    if args.stage == 'status':
        from .analysis import status
        print(json.dumps(status(study), indent=2)); return
    if args.stage == 'report':
        from .analysis import report
        report(study, args.fold); return
    from . import experiment
    if args.stage == 'prepare': experiment.prepare(study)
    elif args.stage == 'qc': experiment.qc(study, args.fold)
    else:
        device = choose_device(args.device)
        if args.stage == 'smoke': experiment.smoke(study, device)
        elif args.stage == 'train': experiment.train(study, args.fold, device)
        elif args.stage == 'generate': experiment.generate(study, args.fold, device)


if __name__ == '__main__': main()
