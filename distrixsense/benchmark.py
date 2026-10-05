import json
import platform
import time
from typing import Any, TypedDict, cast
from pathlib import Path
import numpy as np
import torch
from .models import DISCRETE, LOCAL, RAW
from .temporal_baselines import METHODS as TEMPORAL
from .transport import roundtrip


class TimingStats(TypedDict):
    median: float
    p95: float


class BenchmarkResult(TypedDict):
    peripheral_ms: TimingStats
    serialize_decode_ms: TimingStats
    central_ms: TimingStats
    compute_pipeline_ms: TimingStats
    bytes_per_window: float
    modeled_link_ms: float
    modeled_end_to_end_ms: float
    packet_logit_max_difference: float
    iterations: int
    warmup: int
    device: str
    host: str
    torch_threads: int
    inventory: dict[str, Any]
    gpu_peak_allocated_bytes: int | None
    energy_joules: float | None
    flops: int | None
    limitations: str


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def deployed_modules(model):
    """Only modules executed at inference; training decoders are excluded."""
    method = model.cfg.method
    if method in RAW:
        edge = []
        if method in TEMPORAL:
            central = [model.resamplers, model.temporal_baseline]
        elif method == "deepconvlstm":
            central = [model.resamplers, model.conv_lstm, model.lstm, model.lstm_head]
        elif method == "early_fusion":
            central = [model.resamplers, model.early_projection, model.reasoner]
        else:
            central = [model.resamplers, model.encoders, model.reasoner]
    elif method == "imagebind":
        # Encoders executed before exported features require separately measured costs.
        edge, central = [], [model.external_projections, model.reasoner]
    else:
        edge = [model.resamplers, model.encoders]
        if method in LOCAL:
            edge += [model.local_heads]
            central = []
        elif method in DISCRETE:
            if method != "posthoc_vq":
                edge += [model.poolers, model.projections]
            if method not in ("joint_vq", "posthoc_vq"):
                edge += [model.selectors]
            else:
                # Nearest-neighbor controls must store their codebook at the edge.
                edge += [model.bank]
            central = [model.separate_banks if method == "separate_banks" else model.bank, model.reasoner]
        elif method in ("tokenlearner", "dense_tokens"):
            edge += [model.poolers] + ([model.projections] if method == "dense_tokens" else [])
            central = [model.reasoner]
        elif method == "bottleneck":
            edge += [model.compressors]
            central = [model.decompressors, model.reasoner]
        else:
            central = [model.reasoner]
    return edge, central


def inventory(model):
    def count(modules):
        params = {id(p): p for module in modules for p in module.parameters()}
        return {"parameters": sum(p.numel() for p in params.values()),
                "weight_bytes": sum(p.numel()*p.element_size() for p in params.values())}
    edge, central = deployed_modules(model)
    return {"peripheral_total": count(edge), "central": count(central),
            "research_checkpoint_parameters": sum(p.numel() for p in model.parameters()),
            "note": "Peripheral total is across all sensors, not per device. Shared bank copies in nearest-neighbor baselines are counted on both sides."}


@torch.no_grad()
def benchmark(model, batch, iterations=30, warmup=5) -> BenchmarkResult:
    if batch["labels"].shape[0] != 1 or iterations < 1 or warmup < 0:
        raise ValueError("Benchmark requires batch size one, positive iterations and nonnegative warmup")
    model.eval()
    device = model.cfg.device
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    phases = {n: [] for n in ("peripheral_ms", "serialize_decode_ms", "central_ms", "compute_pipeline_ms")}
    sizes, max_difference = [], 0.
    for i in range(warmup+iterations):
        synchronize(device)
        start = time.perf_counter()
        messages, _ = model.peripheral(batch["streams"])
        synchronize(device)
        edge_end = time.perf_counter()
        received, packets = roundtrip(messages, model.names, device)
        synchronize(device)
        wire_end = time.perf_counter()
        output = model.central(received)
        synchronize(device)
        end = time.perf_counter()
        expected = model.central(messages)
        max_difference = max(max_difference, float((output["logits"]-expected["logits"]).abs().max()))
        if i >= warmup:
            phases["peripheral_ms"].append((edge_end-start)*1000)
            phases["serialize_decode_ms"].append((wire_end-edge_end)*1000)
            phases["central_ms"].append((end-wire_end)*1000)
            phases["compute_pipeline_ms"].append((end-start)*1000)
            sizes.append(sum(len(p)+model.cfg.protocol_overhead for p in packets))
    stats: dict[str, Any] = {n: {"median": float(np.median(v)), "p95": float(np.quantile(v, .95))} for n, v in phases.items()}
    link_ms = float(np.mean(sizes))*8/(model.cfg.bandwidth_mbps*1000) + model.cfg.link_latency_ms
    stats.update({"bytes_per_window": float(np.mean(sizes)), "modeled_link_ms": link_ms,
                  "modeled_end_to_end_ms": stats["compute_pipeline_ms"]["median"]+link_ms,
                  "packet_logit_max_difference": max_difference, "iterations": iterations,
                  "warmup": warmup, "device": str(device), "host": platform.platform(),
                  "torch_threads": torch.get_num_threads(), "inventory": inventory(model),
                  "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
                  "energy_joules": None, "flops": None,
                  "limitations": "Timing starts after data loading and normalization. Link latency is modeled, not measured. CPU peak memory, FLOPs and energy require external instrumentation. ImageBind feature extraction cost is external."})
    return cast(BenchmarkResult, stats)


def export_deployment(model, output, stats):
    """Reviewable edge/central weight bundles and the immutable bank version."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    from .deployment import PeripheralRuntime, CentralRuntime
    PeripheralRuntime(model).save(root/"peripheral.pt", stats)
    CentralRuntime(model).save(root/"central.pt", stats)
    import hashlib
    if model.cfg.method == "separate_banks":
        bank = b"".join(p.detach().cpu().numpy().tobytes() for p in model.separate_banks.values())
    else:
        bank = model.bank.weight.detach().cpu().numpy().tobytes()
    (root/"deployment.json").write_text(json.dumps({"packet_version": "DXS1", "bank_sha256": hashlib.sha256(bank).hexdigest(),
        "modalities": model.names, "bank_offsets": model.offsets, "bank_sizes": model.bank_sizes,
        "inventory": inventory(model), "loading": "Use distrixsense.deployment.PeripheralRuntime.load and CentralRuntime.load."}, indent=2))
