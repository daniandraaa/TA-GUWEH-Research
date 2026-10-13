import argparse
import json
from src.artifacts import choose_device
from .common import DEFAULT_RUN, all_scenarios, context, progress

STAGES = ['prepare', 'smoke', 'gan-train', 'gan-generate', 'gan-qc', 'cv', 'select', 'final', 'test', 'analyze', 'shap', 'status']


def main():
    parser = argparse.ArgumentParser(description='V13 manual: baseline vs fine-tuning sesuai proposal')
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--models', nargs='+', choices=['encoder', 'patchtst', 'hierarchical'])
    parser.add_argument('--scenarios', nargs='+')
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 6))
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--arms', nargs='+', choices=['baseline', 'finetune'])
    args = parser.parse_args()
    if args.stage not in {'cv', 'final'} and (args.models or args.scenarios or args.folds or args.seeds or args.arms):
        parser.error('Filter hanya untuk cv/final')
    study = context(args.run_id, initialize=args.stage == 'prepare')
    if args.scenarios and set(args.scenarios) - {s['id'] for s in all_scenarios(study.protocol)}:
        parser.error('Skenario tidak dikenal')
    if args.seeds and set(args.seeds) - set(study.protocol['cv_seeds']):
        parser.error('Seed CV tidak terdaftar')
    if args.stage == 'status':
        print(json.dumps(progress(study), indent=2)); return
    from . import experiment, gan_final
    if args.stage == 'prepare': experiment.prepare(study); return
    if args.stage == 'gan-qc': gan_final.qc(study); return
    if args.stage == 'select': experiment.select(study); return
    if args.stage == 'analyze':
        from .analysis import analyze
        analyze(study); return
    if args.stage == 'shap':
        from .explain import explain
        explain(study, 'cpu'); return
    device = choose_device(args.device)
    if args.stage == 'smoke': experiment.smoke(study, device)
    elif args.stage == 'gan-train': gan_final.train(study, device)
    elif args.stage == 'gan-generate': gan_final.generate(study, device)
    elif args.stage == 'cv': experiment.cv(study, device, args.models, args.scenarios, args.folds, args.seeds)
    elif args.stage == 'final': experiment.final(study, device, args.models, tuple(args.arms or ('baseline', 'finetune')))
    elif args.stage == 'test': experiment.evaluate(study, device)


if __name__ == '__main__':
    main()
