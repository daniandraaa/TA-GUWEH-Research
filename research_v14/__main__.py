import argparse
import json
from src.artifacts import choose_device
from .common import DEFAULT_RUN, context, progress

STAGES = ['prepare', 'smoke', 'cv', 'compare', 'status']


def main():
    parser = argparse.ArgumentParser(description='V14: model biner LVOT vs RVOT, CV development saja')
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--models', nargs='+', choices=['encoder', 'patchtst', 'hierarchical'])
    parser.add_argument('--arms', nargs='+', choices=['baseline', 'finetune'])
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 6))
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 43, 44])
    args = parser.parse_args()
    if args.stage != 'cv' and (args.models or args.arms or args.folds or args.seeds):
        parser.error('Filter hanya untuk cv')
    study = context(args.run_id, initialize=args.stage == 'prepare')
    if args.stage == 'status':
        print(json.dumps(progress(study), indent=2)); return
    from . import experiment
    if args.stage == 'prepare': experiment.prepare(study); return
    if args.stage == 'compare': experiment.compare(study); return
    device = choose_device(args.device)
    if args.stage == 'smoke': experiment.smoke(study, device)
    elif args.stage == 'cv': experiment.cv(study, device, args.models, args.arms, args.folds, args.seeds)


if __name__ == '__main__':
    main()
