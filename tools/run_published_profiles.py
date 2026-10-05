"""Offline synthetic CPU/GPU cost matrix with explicit shape-matched core reuse."""
import argparse
import json
from pathlib import Path
import torch
from distrixsense.matrix import expand_matrix
from distrixsense.profiling import run_suite

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'runs/published-dummy/profiles')
    parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--targets',nargs='+',choices=['cpu','cuda','mixed'],default=['cpu','cuda','mixed'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--core',choices=['gemma3_1b','qwen3_4b','qwen3_8b'])
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    plan=expand_matrix(ROOT/'configs/published_matrix.json')
    plan['datasets'].sort(key=lambda s:s['core_name'])
    if args.core:
        plan['datasets']=[s for s in plan['datasets'] if s['core_name']==args.core]
    plan['reuse_language_shapes']=True
    plan['execution_notes']={'dummy_features':True,'pretrained_core':True,'sensor_weights':'random',
        'accuracy_evaluation':False,'frontend_costs_included':False,'torch_threads':args.threads,
        'core_shape_reuse':'Measured frozen-core stages shared only for matching runtime shapes; reported as estimates when reused.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    path=args.output.parent/(args.output.name+'_plan.json')
    path.write_text(json.dumps(plan,indent=2))
    run_suite(path,args.output,args.targets,plan['methods'],args.iterations,args.warmup,resume=args.resume)


if __name__=='__main__':
    main()
