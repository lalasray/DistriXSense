"""Check completed factorial coverage and accounting invariants, without more timing runs."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
from build_published_profiles import build

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/published-dummy'


def main():
    spec,_=build()
    rows=list(csv.DictReader((OUT/'summary.csv').open()))
    expected={(d['dataset'],group,c['name'],m,target) for d in spec['datasets']
              for group in d['modality_sets'] for c in spec['cores'] for m in spec['methods'] for target in ('cpu','cuda','mixed')}
    actual=[(r['dataset'],r['modality_set'],r['core_name'],r['method'],r['target']) for r in rows]
    assert len(actual)==len(set(actual)) and set(actual)==expected
    measured={}
    reuse=[]
    max_difference=0.0
    for row in rows:
        r=json.loads(Path(row['report_path']).read_text())
        lm=r['core']['language']
        assert math.isfinite(r['packet_logit_max_difference']) and r['packet_logit_max_difference']<1e-5
        max_difference=max(max_difference,r['packet_logit_max_difference'])
        assert lm['generated_tokens']==8 and len(lm['generated_token_ids'])==8
        for key in ('prefill','prefill_and_fixed_decode','prompt_and_adapter'):
            assert math.isfinite(lm[key]['median_ms']) and lm[key]['median_ms']>0
        assert r['total_deployed']['parameters']==r['core']['storage']['parameters']+sum(e['storage']['parameters'] for e in r['edges'].values())
        assert r['total_deployed_with_language']['parameters']==r['total_deployed']['parameters']+lm['storage']['parameters']
        assert r['total_deployed']['wire_bytes']==sum(e['network']['wire_bytes'] for e in r['edges'].values())
        k=lm.get('language_measurement_id',lm['language_shape_id'])
        if lm['language_measurement']=='measured_this_case':
            assert k not in measured
            measured[k]=lm
        else:
            reuse.append(lm)
    for lm in reuse:
        reference=measured[lm.get('language_measurement_id',lm['language_shape_id'])]
        assert lm['prefill_and_fixed_decode']==reference['prefill_and_fixed_decode']
        assert lm['generation_arithmetic']==reference['generation_arithmetic']
    result=dict(combinations=len(rows),core_shape_measurements=len(measured),
        unique_core_shapes=len({r['language_shape_id'] for r in rows}),shape_reused_combinations=len(reuse),
        packet_max_logit_difference=max_difference,coverage='complete',accounting_invariants='passed',
        by_core=dict(Counter(r['core_name'] for r in rows)),task_accuracy_tested=False,
        datasets_downloaded=False,gpu_measured=True,by_target=dict(Counter(r['target'] for r in rows)))
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
