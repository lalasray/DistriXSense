"""Verify actual mixed-device packet paths separately from cross-device re-encoding.

Updates only validation fields; all original timing and FLOP measurements remain.
No LLM is reloaded because this check concerns the sensor packet interface.
"""
from dataclasses import replace
import json
from pathlib import Path
import torch
from distrixsense.config import Config
from distrixsense.models import DistributedModel
from distrixsense.profiling import synthetic_batch, move_streams
from distrixsense.transport import serialize, deserialize

ROOT=Path(__file__).resolve().parents[1]/'runs/published-dummy'
torch.set_num_threads(8)
assert torch.cuda.is_available()
cache={}
max_delta=0.
for core in ('gemma3_1b','qwen3_4b','qwen3_8b'):
    for path in sorted((ROOT/(core+'_gpu')).glob('*-mixed-*.json')):
        report=json.loads(path.read_text())
        key=json.dumps([report['configuration'],report['input_shapes'],
                        {n:e['streams'] for n,e in report['edges'].items()}],sort_keys=True)
        if key not in cache:
            cfg=Config(**report['configuration'])
            torch.manual_seed(cfg.seed)
            model=DistributedModel(cfg).eval()
            if cfg.method=='posthoc_vq':
                model.posthoc_ready.fill_(True)
            batch=synthetic_batch(cfg,{n:{'frames':s[1]} for n,s in report['input_shapes'].items()},3)
            with torch.no_grad():
                messages={}
                for node in sorted(report['edges']):
                    streams={n:batch['streams'][n] for n in report['edges'][node]['streams']}
                    messages.update(model.peripheral(streams)[0])
                model.cuda()
                direct={n:replace(m,values=m.values.cuda(),times=m.times.cuda(),valid=m.valid.cuda()) for n,m in messages.items()}
                received={}
                for n,m in messages.items():
                    packet=serialize(m,0,model.names.index(n))
                    if packet:
                        received[n]=deserialize(packet,model.names,model.vocabulary,'cuda')
                delta=float((model.central(direct)['logits']-model.central(received)['logits']).abs().max())
                assert delta<1e-5, (path.name,delta)
                colocated=model.central(model.peripheral(move_streams(batch['streams'],'cuda'))[0])['logits']
                device_delta=float((model.central(received)['logits']-colocated).abs().max())
            cache[key]=(delta,device_delta)
            max_delta=max(max_delta,delta)
        delta,device_delta=cache[key]
        report.setdefault('colocated_reference_logit_max_difference',report['packet_logit_max_difference'])
        report['packet_logit_max_difference']=delta
        report['transport_validation']='Same CPU edge outputs sent directly vs serialized/deserialized to GPU core; independently rechecked after profiling.'
        report['rechecked_colocated_reference_logit_max_difference']=device_delta
        path.write_text(json.dumps(report,indent=2))
    print(core,'mixed packet checks passed',flush=True)
(ROOT/'mixed_transport_validation.json').write_text(json.dumps({'unique_sensor_cases':len(cache),'reports_checked':3*len(cache),
    'max_packet_logit_difference':max_delta,'cross_device_rounding_separate':True},indent=2))
