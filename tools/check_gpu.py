"""Functional GPU checks; not a hardware benchmark or dataset download."""
import json
from dataclasses import replace
from pathlib import Path
import torch
from distrixsense.config import Config, METHODS
from distrixsense.models import DistributedModel
from distrixsense.profiling import profile_model
from distrixsense.profiling import synthetic_batch, profile_language

torch.set_num_threads(2)
assert torch.cuda.is_available()
cfg=Config(modalities={'imu':6,'audio':4},hidden=8,heads=2,layers=1,tokens=2,resample_length=8,dropout=0,modality_dropout=0)
batch=synthetic_batch(cfg,{'imu':{'frames':20},'audio':{'frames':30}},3)
reports=[]
for target in ('cuda','mixed'):
    for method in METHODS:
        model=DistributedModel(replace(cfg,method=method)).eval()
        model.posthoc_ready.fill_(True)
        topology={'core_device':'cuda','stream_devices':{'imu':'wrist','audio':'head'},
                  'edges':{n:{'device':'cpu' if target=='mixed' else 'cuda'} for n in ('wrist','head')}}
        result=profile_model(model,batch,topology,1,0)
        assert result['packet_logit_max_difference']<1e-5, (target,method,result['packet_logit_max_difference'])
        reports.append({'target':target,'method':method,'packet_difference':result['packet_logit_max_difference'],
                        'counted_flops':result['total_deployed']['counted_flops']})
root=Path(__file__).resolve().parents[1]
model=DistributedModel(cfg).cuda().eval()
lm=profile_language(model,batch,{'checkpoint':'checkpoints/gemma-3-1b-it','dtype':'bfloat16','decode_tokens':8},root,'cuda',1,0)
assert lm['generated_tokens']==8
out={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
     'checks':reports,'language':lm,'purpose':'functional smoke, concurrent host load; not final timings'}
(root/'runs/published-dummy/gpu_smoke.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'checks':len(reports),'gpu':out['gpu'],'gemma_generated_tokens':lm['generated_tokens']}))
