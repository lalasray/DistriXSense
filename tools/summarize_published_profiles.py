"""Collect completed core suites without disguising reused timings as new measurements."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/published-dummy'


def main():
    rows=[]
    device_rows=[]
    availability=[]
    counts={}
    folders=[OUT/(core+suffix) for core in ('gemma3_1b','qwen3_4b','qwen3_8b') for suffix in ('','_gpu')]
    for folder in folders:
        if not (folder/'summary.csv').exists():
            raise ValueError(f'Incomplete suite: {folder}')
        with (folder/'summary.csv').open() as f:
            part=list(csv.DictReader(f))
        counts[folder.name]=len(part)
        for row in part:
            name=f"{row['scenario_id'].replace('+','plus')}-{row['target']}-{row['method']}.json"
            report=json.loads((folder/name).read_text())
            lm=report['core']['language']
            row.update(language_measurement=lm['language_measurement'],language_shape_id=lm['language_shape_id'],
                       language_measurement_id=lm.get('language_measurement_id',lm['language_shape_id']),
                       report_path=str(folder/name),
                       prompt_tokens=lm['prompt_tokens_including_prefix'],decode_tokens=lm['decode_tokens'],
                       full_parallel_network_estimate_ms=report['with_language_latency_estimates']['parallel_edges_and_links_ms'],
                       inverse_colocated_latency_windows_per_s=1000/float(row['full_pipeline_estimate_ms']),
                       lm_counted_gflop_per_s=lm['generation_arithmetic']['counted_flops']/(lm['prefill_and_fixed_decode']['median_ms']*1e6),
                       total_counted_gflop_per_s=int(row['full_counted_flops'])/(float(row['full_pipeline_estimate_ms'])*1e6))
        rows.extend(part)
        with (folder/'devices.csv').open() as f:
            device_rows.extend(csv.DictReader(f))
        availability.extend({**r,'execution_environment':folder.name,
                             'scope':'Historical availability in this run environment, not host GPU availability',
                             'host_gpu_available':True}
                            for r in json.loads((folder/'availability.json').read_text()))
    for name,table in [('summary.csv',rows),('devices.csv',device_rows)]:
        with (OUT/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (OUT/'availability.json').write_text(json.dumps(availability,indent=2))
    inventory=json.loads((OUT/'modality_inventory.json').read_text())
    profiles=sorted({r['profile'] for r in inventory})
    lines=['# Pretrained-core synthetic cost results','',
        f'{len(rows):,} completed CPU/GPU/mixed combinations; 21 methods/ablations, three cores, 41 dataset/modality settings.',
        'No dataset files downloaded. Random sensor/adapter weights; pretrained frozen language cores. These are cost/integration tests, not accuracy results.',
        'BF16 core, FP32 sensor/adapter, batch1, eight CPU threads, three repetitions plus one warm-up, eight generated tokens.',
        'GPU: NVIDIA RTX PRO 6000 Blackwell (96GB), CUDA PyTorch2.14.1+cu130. Mixed: CPU edges, GPU core. GPU access requires execution outside the sandbox. Same-shape core timing reuse is labeled in summary.csv.',
        '', '|Input profile|Streams|Known decoded numeric MiB / 3s|Feature MiB / 3s|Unknown raw-size streams|',
        '|---|---:|---:|---:|---:|']
    for profile in profiles:
        part=[r for r in inventory if r['profile']==profile]
        raw=sum(r['raw_numeric_bytes'] or 0 for r in part)
        lines.append(f"|{profile}|{len(part)}|{raw/2**20:.3f}|{sum(r['feature_bytes'] for r in part)/2**20:.3f}|{sum(r['raw_numeric_bytes'] is None for r in part)}|")
    lines += ['', 'Input amounts use the explicit assumptions in modality_inventory.csv; decoded volumes are not compressed corpus sizes. Media preprocessing costs are excluded.', '',
              '|All-family proposed model|Core|Target|Deployed parameters (B)|Weights GiB|Counted GFLOPs|Wire KiB/window|Core generation ms|Summed pipeline ms|',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['method']=='distrixsense' and row['modality_set']=='all':
            lines.append(f"|{row['dataset']}|{row['core_name']}|{row['target']}|{int(row['full_parameters'])/1e9:.4f}|{int(row['full_weight_bytes'])/2**30:.3f}|{int(row['full_counted_flops'])/1e9:.3f}|{int(row['wire_bytes'])/1024:.3f}|{float(row['lm_generation_ms']):.2f}|{float(row['full_pipeline_estimate_ms']):.2f}|")
    actual=sum(r['language_measurement']=='measured_this_case' for r in rows)
    lines += ['',f'{actual} core-shape benchmarks measured (including remeasurements after resume); {len(rows)-actual} combinations reuse a matched core stage. Each sensor/adapter configuration is separately profiled.',
              'Full comparisons: summary.csv; each physical/aggregate edge and core: devices.csv; detailed stage and unsupported-FLOP data: individual core directories.',
              'Transfer estimates assume10Mbps links with2ms latency and28-byte packet overhead. FLOPs omit unsupported operators. Three timing samples do not establish reliable p95 performance.',
              'ImageBind is excluded because pretrained ImageBind encoder/features were not tested. Corpus compressed modality sizes and energy consumption are unknown.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows':len(rows),'by_core':counts,'actual_core_shapes':actual,'report':str(OUT/'RESULTS.md')}))


if __name__=='__main__':
    main()
