"""Plan, check or run the continuous scientific workflow from declared inputs."""
import argparse
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'library'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','check','run','smoke','_abm'],nargs='?',default='plan')
    parser.add_argument('--workspace',type=Path,default=BASE)
    parser.add_argument('--from',dest='first')
    parser.add_argument('--to',dest='last')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args(argv)
    if args.retry_failed and not args.resume:parser.error('--retry-failed requires --resume')
    from steel_supply.reproduction.execution import execution_environment,preflight,run_stage
    os.environ.update(execution_environment(args.workspace))
    os.environ.pop('STEEL_INPUT_MODE',None)
    if args.action=='_abm':
        from steel_supply.hazard_dynamic.reproduction_grid import run_grid
        run_grid(args.workspace,resume=args.resume);return 0
    if args.action=='smoke':
        from steel_supply.smoke import run_smoke
        print(json.dumps(run_smoke(args.workspace/'work/smoke'),indent=2));return 0
    from steel_supply.reproduction.plan import build_plan,external_inputs
    try:stages=build_plan(first=args.first,last=args.last)
    except ValueError as error:parser.error(str(error))
    if args.action=='plan':
        print(json.dumps({'workspace':str(args.workspace.resolve()),'mode':'full',
                          'external_inputs':external_inputs(stages),'stages':stages},indent=2));return 0
    report=preflight(args.workspace,stages)
    if args.action=='check' or report['status']!='PASS':
        print(json.dumps(report,indent=2));return 0 if report['status']=='PASS' else 1
    for stage in stages:run_stage(stage,args.workspace,resume=args.resume,retry_failed=args.retry_failed)
    return 0


if __name__=='__main__':raise SystemExit(main())
