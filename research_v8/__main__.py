import argparse
import json
from src.artifacts import choose_device
from .common import DEFAULT_RUN,context,progress


def main():
    parser=argparse.ArgumentParser(description='Manual paired classifier augmentation; development only')
    parser.add_argument('stage',choices=['prepare','smoke','pilot','train','analyze','status'])
    parser.add_argument('--run-id',default=DEFAULT_RUN)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--models',nargs='+',choices=['hierarchical','encoder'])
    parser.add_argument('--folds',nargs='+',type=int,choices=range(1,6))
    parser.add_argument('--seeds',nargs='+',type=int,choices=[42,43,44])
    parser.add_argument('--pilot-report',action='store_true')
    args=parser.parse_args()
    if args.pilot_report and args.stage!='analyze':parser.error('--pilot-report only with analyze')
    if args.stage not in {'train','smoke'} and (args.models or args.folds or args.seeds):parser.error('Filters only for train/smoke')
    if args.stage=='smoke' and (args.folds or args.seeds):parser.error('Smoke uses fixed fold/seed')
    study=context(args.run_id,initialize=args.stage=='prepare')
    if args.stage=='status':print(json.dumps(progress(study),indent=2));return
    if args.stage=='analyze':
        from .analysis import analyze
        analyze(study,pilot=args.pilot_report);return
    from . import experiment
    if args.stage=='prepare':experiment.prepare(study);return
    device=choose_device(args.device)
    if args.stage=='smoke':experiment.smoke(study,args.models or study.protocol['models'],device)
    elif args.stage=='pilot':experiment.train(study,study.protocol['models'],[1],[42],device,pilot=True)
    else:experiment.train(study,args.models or study.protocol['models'],args.folds or study.protocol['folds'],args.seeds or study.protocol['seeds'],device)


if __name__=='__main__':main()
