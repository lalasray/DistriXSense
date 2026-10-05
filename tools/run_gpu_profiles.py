"""Run GPU suites after CPU suites finish, avoiding overlapping timing jobs."""
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/published-dummy'
assert torch.cuda.is_available(), 'GPU execution needs access outside the sandbox.'
print('Waiting for CPU suites to finish before timing GPU suites.',flush=True)
deadline=time.monotonic()+4*3600
while not all((OUT/core/'summary.csv').exists() for core in ('gemma3_1b','qwen3_4b','qwen3_8b')):
    if time.monotonic()>deadline:
        raise TimeoutError('CPU suites did not finish; inspect CPU logs before resuming.')
    time.sleep(10)
with (OUT/'regression-tests.log').open('w') as handle:
    subprocess.run([str(ROOT/'.venv/bin/python'),'-m','unittest','discover','-s','tests'],cwd=ROOT,
                   stdout=handle,stderr=subprocess.STDOUT,check=True)
for core in ('gemma3_1b','qwen3_4b','qwen3_8b'):
    output=OUT/(core+'_gpu')
    print('GPU and mixed suites:',core,flush=True)
    command=[sys.executable,str(ROOT/'tools/run_published_profiles.py'),'--core',core,
             '--targets','cuda','mixed','--output',str(output)]
    if output.exists():
        command.append('--resume')
    with (OUT/(core+'_gpu.log')).open('a') as handle:
        subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
for tool in ('summarize_published_profiles.py','validate_published_results.py'):
    subprocess.run([sys.executable,str(ROOT/'tools'/tool)],cwd=ROOT,check=True)
print('CPU, GPU and mixed suites complete and validated.',flush=True)
