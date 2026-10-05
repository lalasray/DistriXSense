"""Per-device deployment accounting with explicit input and network assumptions."""
import copy
import csv
import io
import inspect
import json
import math
import platform
import resource
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from .benchmark import deployed_modules, synchronize
from .config import Config, METHODS
from .data import ManifestDataset, collate, to_device
from .models import DistributedModel, DISCRETE, LOCAL, RAW
from .training import load_checkpoint
from .transport import HEADER, serialize, deserialize


def edge_modules(model, names):
    """Modules actually stored on ONE device; shared copies counted per device."""
    method = model.cfg.method
    if method in RAW or method == "imagebind":
        return []
    modules = [collection[n] for collection in (model.resamplers, model.encoders) for n in names]
    if method in LOCAL:
        modules += [model.local_heads[n] for n in names]
    elif method in DISCRETE:
        if method != "posthoc_vq":
            modules += [collection[n] for collection in (model.poolers, model.projections) for n in names]
        if method in ("joint_vq", "posthoc_vq"):
            # Current nearest-neighbor runtime stores the complete bank on each node.
            modules += [model.bank]
        else:
            modules += [model.selectors[n] for n in names]
    elif method in ("tokenlearner", "dense_tokens"):
        modules += [model.poolers[n] for n in names]
        if method == "dense_tokens":
            modules += [model.projections[n] for n in names]
    elif method == "bottleneck":
        modules += [model.compressors[n] for n in names]
    return modules


def storage(modules):
    parameters = {id(p): p for m in modules for p in m.parameters()}
    buffers = {id(p): p for m in modules for p in m.buffers()}
    weight_bytes = sum(p.numel()*p.element_size() for p in parameters.values())
    archived_bytes = None
    if weight_bytes <= 256*2**20:
        state = {f"module{i}.{k}": v.detach().cpu() for i, m in enumerate(modules) for k, v in m.state_dict().items()}
        archive = io.BytesIO()
        torch.save(state, archive)
        archived_bytes = len(archive.getvalue()) if modules else 0
    return {"parameters": sum(p.numel() for p in parameters.values()),
            "weight_bytes": weight_bytes,
            "buffer_bytes": sum(p.numel()*p.element_size() for p in buffers.values()),
            "serialized_state_bytes": archived_bytes}


class OperationCounter(FlopCounterMode):
    def __init__(self):
        def cpu_attention(query, key, value, *args, **kwargs):
            # QK^T and attention*V; matches PyTorch's SDPA dense-MAC convention.
            return 2*math.prod(query[:-2])*query[-2]*key[-2]*(query[-1]+value[-1])
        mapping = {}
        if hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu"):
            mapping[torch.ops.aten._scaled_dot_product_flash_attention_for_cpu] = cpu_attention
        super().__init__(display=False, custom_mapping=mapping)
        self.unmodeled = set()

    def _count_flops(self, func_packet, out, args, kwargs):
        if func_packet not in self.flop_registry:
            self.unmodeled.add(str(func_packet))
        return super()._count_flops(func_packet, out, args, kwargs)


def arithmetic(fn):
    # Fused inference kernels can conceal operations from dispatch accounting.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    mkldnn = torch.backends.mkldnn.enabled
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.mkldnn.enabled = False
    try:
        with torch.no_grad(), OperationCounter() as counter:
            fn()
        return {"counted_flops": counter.get_total_flops(),
                "counted_operations": {str(k): v for k, v in counter.get_flop_counts().get("Global", {}).items()},
                "unmodeled_operations": sorted(counter.unmodeled),
                "convention": "Formula-based matrix/convolution/attention FLOPs; multiply-add=2. Unregistered operations excluded. This is not a complete instruction count."}
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        torch.backends.mkldnn.enabled = mkldnn


def measure(fn, device, iterations, warmup):
    for _ in range(warmup):
        fn()
    synchronize(device)
    baseline = torch.cuda.memory_allocated(device) if str(device).startswith("cuda") else None
    if baseline is not None:
        torch.cuda.reset_peak_memory_stats(device)
    values = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        values.append((time.perf_counter()-start)*1000)
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95)),
            "min_ms": min(values), "iterations": iterations, "warmup": warmup,
            "cuda_peak_incremental_allocated_bytes": max(0, torch.cuda.max_memory_allocated(device)-baseline) if baseline is not None else None}


def link_cost(size, node, cadence):
    rate = float(node.get("bandwidth_mbps", 10))
    latency = float(node.get("latency_ms", 0))
    overhead = int(node.get("overhead_bytes_per_packet", 0))
    mtu = int(node.get("mtu_payload_bytes", 1472))
    if rate <= 0 or latency < 0 or overhead < 0 or not np.isfinite([rate, latency]).all():
        raise ValueError("Network bandwidth must be positive; latency/overhead nonnegative")
    if mtu < 1:
        raise ValueError("MTU payload must be positive")
    fragments = sum(math.ceil(v/mtu) for v in size.get("application_packet_sizes", [size["application_bytes"]]) if v)
    wire = size["application_bytes"]+fragments*overhead
    tx = wire*8/(rate*1000)
    return {**size, "protocol_overhead_bytes": fragments*overhead, "wire_bytes": wire,
            "transport_fragments": fragments, "mtu_payload_bytes": mtu,
            "bandwidth_mbps": rate, "latency_ms": latency,
            "serialization_on_link_ms": tx, "transfer_ms": tx+latency if wire else 0.,
            "required_mbps": wire*8/(cadence*1e6), "link_utilization": wire*8/(cadence*rate*1e6)}


def packet_sizes(messages, names):
    result = {"packets": 0, "application_bytes": 0, "header_bytes": 0,
              "timestamp_bytes": 0, "representation_bytes": 0, "application_packet_sizes": []}
    for n, message in messages.items():
        packet = serialize(message, 0, names.index(n))
        if packet:
            frames = int(message.valid[0].sum())
            result["packets"] += 1
            result["application_bytes"] += len(packet)
            result["application_packet_sizes"].append(len(packet))
            result["header_bytes"] += HEADER.size
            result["timestamp_bytes"] += frames*4
            result["representation_bytes"] += len(packet)-HEADER.size-frames*4
    return result


def move_streams(streams, device):
    return {n: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in s.items()}
            for n, s in streams.items()}


@torch.no_grad()
def profile_model(model, batch, topology, iterations=20, warmup=5, cadence=1.5):
    if iterations < 1 or warmup < 0 or cadence <= 0 or batch["labels"].numel() != 1:
        raise ValueError("Profiling needs batch size one, positive iterations/cadence and nonnegative warmup")
    mapping = topology.get("stream_devices", {n: n for n in model.names})
    if set(mapping) != set(model.names):
        raise ValueError("stream_devices must assign every stream exactly once")
    nodes = topology.get("edges", {})
    core_device = topology.get("core_device", "cpu")
    model = model.to(core_device).eval()
    models = {core_device: model}
    all_messages, edges, edge_jobs = {}, {}, []
    for node_name in sorted(set(mapping.values())):
        names = [n for n in model.names if mapping[n] == node_name]
        spec = nodes.get(node_name, {})
        device = spec.get("device", "cpu")
        if device not in models:
            models[device] = copy.deepcopy(model).to(device).eval()
        edge_model = models[device]
        streams = move_streams({n: batch["streams"][n] for n in names}, device)
        edge_jobs.append((edge_model, streams))
        def edge_fn():
            return edge_model.peripheral(streams)[0]
        messages = edge_fn()
        all_messages.update(messages)
        def encode_fn():
            return [serialize(m, 0, model.names.index(n)) for n, m in messages.items()]
        edges[node_name] = {"streams": names, "execution_device": device,
                            "storage": storage(edge_modules(model, names)),
                            "compute": measure(edge_fn, device, iterations, warmup),
                            "encoding": measure(encode_fn, device, iterations, warmup),
                            "arithmetic": arithmetic(edge_fn),
                            "input_tensor_bytes": sum(s["x"].numel()*s["x"].element_size() for s in streams.values()),
                            "network": link_cost(packet_sizes(messages, model.names), spec, cadence)}
    packets = [serialize(m, 0, model.names.index(n)) for n, m in all_messages.items()]
    packets = [p for p in packets if p]
    def decode_fn():
        result = {}
        for packet in packets:
            message = deserialize(packet, model.names, model.vocabulary, core_device)
            result[message.modality] = message
        return result
    received = decode_fn()
    def core_fn():
        return model.central(received)
    core = {"execution_device": core_device, "storage": storage(deployed_modules(model)[1]),
            "compute": measure(core_fn, core_device, iterations, warmup),
            "decoding": measure(decode_fn, core_device, iterations, warmup), "arithmetic": arithmetic(core_fn)}
    def pipeline_fn():
        decoded = {}
        for edge_model, streams in edge_jobs:
            msgs = edge_model.peripheral(streams)[0]
            for name, message in msgs.items():
                packet = serialize(message, 0, model.names.index(name))
                if packet:
                    received_message = deserialize(packet, model.names, model.vocabulary, core_device)
                    decoded[name] = received_message
        return model.central(decoded)
    pipeline = measure(pipeline_fn, core_device, iterations, warmup)
    pipeline["windows_per_second"] = 1000/max(pipeline["median_ms"], 1e-12)
    combined, _ = model.peripheral(move_streams(batch["streams"], core_device))
    reference = model.central(combined)["logits"]
    difference = float((core_fn()["logits"]-reference).abs().max())
    def duration(e):
        return e["compute"]["median_ms"]+e["encoding"]["median_ms"]+e["network"]["transfer_ms"]
    critical = max(map(duration, edges.values()), default=0.)
    core_ms = core["decoding"]["median_ms"]+core["compute"]["median_ms"]
    sequential = sum(map(duration, edges.values()))+core_ms
    totals = {}
    for k in ("parameters", "weight_bytes", "buffer_bytes", "serialized_state_bytes"):
        values = [e["storage"][k] for e in edges.values()]+[core["storage"][k]]
        totals[k] = sum(values) if all(v is not None for v in values) else None
    totals.update(counted_flops=sum(e["arithmetic"]["counted_flops"] for e in edges.values())+core["arithmetic"]["counted_flops"],
                  wire_bytes=sum(e["network"]["wire_bytes"] for e in edges.values()))
    raw_payload = sum(int(s["mask"].sum())*s["x"].shape[-1]*4 for s in batch["streams"].values())
    representation = sum(e["network"]["representation_bytes"] for e in edges.values())
    compression = {"raw_numeric_payload_bytes": raw_payload, "representation_bytes": representation,
                   "raw_to_representation_ratio": raw_payload/representation if representation else None,
                   "raw_to_wire_ratio": raw_payload/totals["wire_bytes"] if totals["wire_bytes"] else None}
    shared = topology.get("shared_uplink")
    shared_link = link_cost({"application_bytes": totals["wire_bytes"], "packets": len(packets)}, shared, cadence) if shared else None
    # Shared uplink is a serialized bottleneck after edge readiness; not parallel links.
    shared_ms = (max((e["compute"]["median_ms"]+e["encoding"]["median_ms"] for e in edges.values()), default=0.)
                 + shared_link["transfer_ms"]+core_ms) if shared_link else None
    return {"method": model.cfg.method, "edges": edges, "core": core, "total_deployed": totals,
            "measured_colocated_sequential_pipeline": pipeline,
            "compression": compression,
            "packet_logit_max_difference": difference,
            "schedule_estimates": {"parallel_independent_edges_and_links_ms": critical+core_ms,
                                   "sequential_edges_and_links_ms": sequential, "shared_uplink_ms": shared_ms,
                                   "window_cadence_seconds": cadence,
                                   "parallel_can_keep_up": critical+core_ms <= cadence*1000},
            "research_model_parameters": sum(p.numel() for p in model.parameters()),
            "host": {"platform": platform.platform(), "cpu": platform.processor(),
                     "torch": torch.__version__, "threads": torch.get_num_threads(),
                     "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                     "process_peak_rss_bytes_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
            "energy_joules": None,
            "scope": {"sensor_interface_and_classifier": "included", "language_model": "not_configured",
                      "media_or_pretrained_frontend": "excluded", "normalization_and_data_loading": "excluded"},
            "limitations": ["Stage medians combined in schedule estimates; not measured network or simultaneous-device latency.",
                            "CPU peak RSS is cumulative process memory, not isolated model activation memory.",
                            "FLOPs are registered arithmetic only; unmodeled operators listed per stage.",
                            "Front-end media/feature extraction and optional language model excluded unless separately profiled."]}


@torch.no_grad()
def profile_language(model, batch, spec, base, device, iterations, warmup, saved=None):
    """Optional real local LM: fixed decode length, prompt-specific costs, no downloads."""
    from .language import SensorLanguageModel
    mode = spec.get("mode", "sensor")
    if model.cfg.method in LOCAL and mode == "sensor":
        mode = "classifier"
    if saved and "language_adapter" in saved:
        provenance = saved["provenance"]
        if Path(provenance["language_checkpoint"]).resolve() != (base/spec["checkpoint"]).resolve() or provenance["language_mode"] != mode:
            raise ValueError("Saved language adapter belongs to a different LLM core or prompt mode")
    lm = SensorLanguageModel.from_local(base/spec["checkpoint"], model.cfg.hidden, device, mode=mode,
                                       dtype=spec.get("dtype", "float32"), prompt_style=spec.get("prompt_style", "auto")).eval()
    if saved and "language_adapter" in saved:
        lm.adapter.load_state_dict(saved["language_adapter"])
    query_batch = to_device(batch, device)
    query_batch["queries"] = [spec.get("query", "What activity is happening?")]
    sensor_output = model(query_batch)
    def embedding_fn():
        return lm.embeddings(sensor_output, query_batch, 0)[0]
    inputs = embedding_fn()
    mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=device)
    tokens = int(spec.get("decode_tokens", 16))
    if tokens < 1:
        raise ValueError("decode_tokens must be positive")
    forward_args = inspect.signature(lm.lm.forward).parameters
    last_logit = {"logits_to_keep": 1} if "logits_to_keep" in forward_args else ({"num_logits_to_keep": 1} if "num_logits_to_keep" in forward_args else {})
    def prefill_fn():
        return lm.lm(inputs_embeds=inputs, attention_mask=mask, use_cache=True, **last_logit)
    def generate_fn():
        return lm.lm.generate(inputs_embeds=inputs, attention_mask=mask, do_sample=False,
                              min_new_tokens=tokens, max_new_tokens=tokens,
                              pad_token_id=lm.tokenizer.pad_token_id, eos_token_id=lm.tokenizer.eos_token_id)
    return {"mode": mode, "checkpoint": str(base/spec["checkpoint"]), "prompt_tokens_including_prefix": inputs.shape[1],
            "dtype": str(lm.lm.get_input_embeddings().weight.dtype), "prompt_style": lm.prompt_style,
            "prefill_logit_policy": "last_position" if last_logit else "model_default",
            "decode_tokens": tokens, "storage": storage([lm.lm]+([lm.adapter] if mode == "sensor" else [])),
            "prompt_and_adapter": measure(embedding_fn, device, iterations, warmup),
            "prompt_and_adapter_arithmetic": arithmetic(embedding_fn),
            "prefill": measure(prefill_fn, device, iterations, warmup),
            "prefill_arithmetic": arithmetic(prefill_fn),
            "prefill_and_fixed_decode": measure(generate_fn, device, iterations, warmup),
            "generation_arithmetic": arithmetic(generate_fn),
            "adapter_weights": "checkpoint" if saved and "language_adapter" in saved else "random cost fixture"}


def synthetic_batch(cfg, inputs, window):
    if set(inputs) != set(cfg.modalities):
        raise ValueError("Input shapes must describe every modality")
    generator = torch.Generator().manual_seed(cfg.seed)
    streams = {}
    for n, c in cfg.modalities.items():
        length = int(inputs[n]["frames"])
        if length < 1:
            raise ValueError("Input frames must be positive")
        streams[n] = {"x": torch.randn(1, length, c, generator=generator),
                      "time": torch.linspace(0, window, length)[None],
                      "mask": torch.full((1, length), inputs[n].get("available", True), dtype=torch.bool)}
    return {"streams": streams, "labels": torch.zeros(1, dtype=torch.long)}


def run_suite(spec_path, output, targets=("cpu", "cuda", "mixed"), methods=METHODS, iterations=20, warmup=5):
    path, root = Path(spec_path), Path(output)
    spec = json.loads(path.read_text())
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty profiling output directory")
    root.mkdir(parents=True, exist_ok=True)
    (root/"input_spec.json").write_text(json.dumps(spec, indent=2))
    results, skipped = [], []
    for entry in spec["datasets"]:
        cfg = Config(**entry["config"]).validate()
        for target in targets:
            if target not in ("cpu", "cuda", "mixed"):
                raise ValueError("Targets: cpu, cuda, mixed")
            if target != "cpu" and not torch.cuda.is_available():
                skipped.append({"dataset": entry["dataset"], "target": target, "status": "unavailable",
                                "reason": "CUDA-enabled PyTorch and accessible GPU required"})
                continue
            for method in methods:
                if method in entry.get("excluded_methods", []):
                    skipped.append({"dataset": entry["dataset"], "scenario_id": entry.get("scenario_id"),
                                    "target": target, "method": method, "status": "unsupported_modality_subset"})
                    continue
                if method not in METHODS:
                    raise ValueError(f"Unknown method: {method}")
                scenario = {**entry, **entry.get("method_profiles", {}).get(method, {})}
                cfg = Config(**scenario["config"]).validate()
                torch.manual_seed(cfg.seed)
                checkpoint = scenario.get("checkpoints", {}).get(method)
                method_cfg = replace(cfg, method=method)
                if checkpoint:
                    model, saved = load_checkpoint(path.parent/checkpoint)
                    if model.cfg.method != method or model.cfg.modalities != cfg.modalities:
                        raise ValueError("Checkpoint method/input configuration mismatch")
                else:
                    saved = None
                    model = DistributedModel(method_cfg).eval()
                    # Shapes of random reference codebooks suffice for cost profiling only.
                    if method == "posthoc_vq":
                        model.posthoc_ready.fill_(True)
                if scenario.get("manifest"):
                    ds = ManifestDataset(path.parent/scenario["manifest"], "test", cfg.modalities,
                                         saved["stats"] if checkpoint else None)
                    batch = collate([ds[scenario.get("sample_index", 0)]], cfg.modalities)
                else:
                    batch = synthetic_batch(cfg, scenario["inputs"], scenario["window_seconds"])
                topology = copy.deepcopy(scenario.get("topology", {}))
                mapping = topology.get("stream_devices", {n: n for n in cfg.modalities})
                topology["stream_devices"] = mapping
                topology["core_device"] = "cpu" if target == "cpu" else "cuda"
                topology.setdefault("edges", {})
                for node in set(mapping.values()):
                    topology["edges"].setdefault(node, {})["device"] = "cuda" if target == "cuda" else "cpu"
                report = profile_model(model, batch, topology, iterations, warmup, scenario["stride_seconds"])
                if scenario.get("language"):
                    language = profile_language(model, batch, scenario["language"], path.parent,
                                                topology["core_device"], iterations, warmup, saved)
                    report["core"]["language"] = language
                    report["scope"]["language_model"] = "included_separate_prompt_specific_stage"
                    report["total_deployed_with_language"] = {
                        **{k: (report["total_deployed"][k]+language["storage"][k]) if report["total_deployed"][k] is not None and language["storage"][k] is not None else None
                           for k in ("parameters", "weight_bytes", "buffer_bytes", "serialized_state_bytes")},
                        "counted_flops": report["total_deployed"]["counted_flops"]+language["generation_arithmetic"]["counted_flops"]+language["prompt_and_adapter_arithmetic"]["counted_flops"],
                        "wire_bytes": report["total_deployed"]["wire_bytes"]}
                    report["with_language_latency_estimates"] = {
                        "colocated_pipeline_ms": report["measured_colocated_sequential_pipeline"]["median_ms"]+language["prompt_and_adapter"]["median_ms"]+language["prefill_and_fixed_decode"]["median_ms"],
                        "parallel_edges_and_links_ms": report["schedule_estimates"]["parallel_independent_edges_and_links_ms"]+language["prompt_and_adapter"]["median_ms"]+language["prefill_and_fixed_decode"]["median_ms"]}
                report.update(dataset=scenario["dataset"], target=target, configuration=model.cfg.to_dict(),
                              scenario_id=entry.get("scenario_id", entry["dataset"]),
                              core_name=entry.get("core_name", "sensor_only"), modality_set=entry.get("modality_set", "all"),
                              input_shapes={n: list(s["x"].shape) for n, s in batch["streams"].items()},
                              evidence="checkpoint" if checkpoint else "random_weights_cost_fixture",
                              input_provenance=scenario.get("description", "Explicit user-provided input assumptions"))
                if method == "imagebind":
                    report["limitations"].append("ImageBind fusion branch only. Input dimensions must be actual ImageBind embeddings for a valid ImageBind cost comparison; pretrained encoders are not included.")
                filename = f"{entry.get('scenario_id', entry['dataset']).replace('+','plus')}-{target}-{method}.json"
                (root/filename).write_text(json.dumps(report, indent=2))
                results.append(report)
                print(f"profile {entry['dataset']} {target} {method}: {report['total_deployed']['parameters']} parameters", flush=True)
    (root/"availability.json").write_text(json.dumps(skipped, indent=2))
    columns = ["dataset", "target", "method", "scenario_id", "core_name", "modality_set", "parameters", "weight_bytes", "counted_flops", "wire_bytes",
               "core_parameters", "edge_parameters", "core_median_ms", "parallel_estimate_ms", "sequential_estimate_ms",
               "full_parameters", "full_weight_bytes", "full_counted_flops", "lm_parameters", "lm_generation_ms", "full_pipeline_estimate_ms"]
    with (root/"summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in results:
            full = r.get("total_deployed_with_language", r["total_deployed"])
            language = r["core"].get("language", {})
            writer.writerow({**{k: r[k] for k in ("dataset", "target", "method", "scenario_id", "core_name", "modality_set")},
                             **{k: r["total_deployed"][k] for k in ("parameters", "weight_bytes", "counted_flops", "wire_bytes")},
                             "core_parameters": r["core"]["storage"]["parameters"],
                             "edge_parameters": sum(e["storage"]["parameters"] for e in r["edges"].values()),
                             "core_median_ms": r["core"]["compute"]["median_ms"],
                             "parallel_estimate_ms": r["schedule_estimates"]["parallel_independent_edges_and_links_ms"],
                             "sequential_estimate_ms": r["schedule_estimates"]["sequential_edges_and_links_ms"],
                             "full_parameters": full["parameters"], "full_weight_bytes": full["weight_bytes"], "full_counted_flops": full["counted_flops"],
                             "lm_parameters": language.get("storage", {}).get("parameters", 0),
                             "lm_generation_ms": language.get("prefill_and_fixed_decode", {}).get("median_ms"),
                             "full_pipeline_estimate_ms": r.get("with_language_latency_estimates", {}).get("colocated_pipeline_ms", r["measured_colocated_sequential_pipeline"]["median_ms"])})
    text = "# Deployment cost fixtures\n\nThese are shape-dependent cost measurements, not real-dataset performance results. FLOPs exclude unmodeled operators. GPU availability is recorded separately.\n\n|Dataset|Target|Method|Deployed parameters|Weight MiB|Counted MFLOPs|Wire KiB|Core ms|Parallel estimate ms|\n|---|---|---|---:|---:|---:|---:|---:|---:|\n"
    for r in results:
        v = r.get("total_deployed_with_language", r["total_deployed"])
        text += f"|{r['scenario_id']}|{r['target']}|{r['method']}|{v['parameters']}|{v['weight_bytes']/2**20:.3f}|{v['counted_flops']/1e6:.3f}|{v['wire_bytes']/1024:.3f}|{r['core']['compute']['median_ms']:.3f}|{r['schedule_estimates']['parallel_independent_edges_and_links_ms']:.3f}|\n"
    (root/"summary.md").write_text(text)
    write_device_table(root, results)
    return root/"summary.md"


def write_device_table(root, results):
    fields = ["dataset", "target", "method", "scenario_id", "core_name", "modality_set", "side", "node", "streams", "parameters", "weight_bytes",
              "buffer_bytes", "serialized_state_bytes", "counted_flops", "compute_median_ms", "compute_p95_ms",
              "encoding_or_decoding_ms", "wire_bytes", "transfer_ms", "bandwidth_mbps", "link_utilization"]
    with (Path(root)/"devices.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in results:
            stages = [(node, stage, True) for node, stage in report["edges"].items()]+[("core", report["core"], False)]
            language = report["core"].get("language")
            if language:
                stages.append(("core_language", {"storage": language["storage"], "compute": language["prefill_and_fixed_decode"],
                    "decoding": language["prompt_and_adapter"], "arithmetic": {"counted_flops": language["generation_arithmetic"]["counted_flops"]+language["prompt_and_adapter_arithmetic"]["counted_flops"]}}, False))
            for node, stage, edge in stages:
                network = stage.get("network", {})
                writer.writerow({**{k: report.get(k) for k in ("dataset", "target", "method", "scenario_id", "core_name", "modality_set")},
                                 "side": "edge" if edge else "core", "node": node,
                                 "streams": ",".join(stage.get("streams", [])), **stage["storage"],
                                 "counted_flops": stage["arithmetic"]["counted_flops"],
                                 "compute_median_ms": stage["compute"]["median_ms"], "compute_p95_ms": stage["compute"]["p95_ms"],
                                 "encoding_or_decoding_ms": stage["encoding" if edge else "decoding"]["median_ms"],
                                 **{k: network.get(k) for k in ("wire_bytes", "transfer_ms", "bandwidth_mbps", "link_utilization")}})
