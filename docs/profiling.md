# Device, inference and communication costs

```bash
python -m distrixsense profile-suite \
  --spec configs/profiling.json --output runs/my-device-profiles \
  --targets cpu cuda mixed --iterations 20 --warmup 5
```

This covers all 22 implemented methods and three dataset profiles. `cpu` measures
edge/core execution on this CPU; `cuda` measures both on an accessible GPU; `mixed`
measures CPU edges and a GPU core. CUDA cases are marked unavailable when CUDA cannot
run. CPU timings are not extrapolated into measured GPU timings. The global
`--threads` option precedes the command and controls PyTorch CPU threads.

The supplied profile is **illustrative synthetic feature input**, not a real release
inventory or experiment. Assumptions: ten classes, hidden width 32, two layers,
32 resampled positions, eight selected tokens, three-second windows, 1.5-second
cadence, and explicit channel/frame counts. Inertial families may overlap in real
recordings. Device assignments are illustrative. Images are pooled features, audio
is spectral features, and points are statistical features. Extraction is excluded.
Replace dimensions, device multiplicity, frame counts, classes and placement with
your prepared data before using the results in a paper.

## Outputs and scope

- `summary.csv` / `summary.md`: each dataset/method/target, deployed parameters,
  weights, counted FLOPs, communication volume, core timing and schedule estimates.
- `devices.csv`: **each physical edge** and core, with storage, arithmetic,
  median/p95 compute time, encoding/decoding and link costs.
- Per-method JSON: shapes, config, provenance, operation breakdown, unmodeled
  operations, compression ratios, measured pipeline throughput and network assumptions.
- `input_spec.json` / `availability.json`: reproducible assumptions and unavailable
  GPU targets.

The proposed peripheral stores resampling, encoding, pooling, projections and
selectors; bank and reasoner reside centrally. Unused training decoders and comparison
modules are excluded. Streams on one node share that node's storage. Nearest-neighbor
VQ currently replicates the full bank per node, and copies are counted. Centralized
baselines have no learned edge weights but still transmit observations. Research-model
parameters are reported separately from deployed parameters.

Actual tensor dtype determines weight bytes. Buffers and serialized archive overhead
are separate. PyTorch weight archives are measured for bundles up to 256 MiB;
larger bundles return null to avoid copying enormous weights merely for sizing.
Archive size is not firmware or compiled executable size.

The base report includes the sensor interface/classifier. Data loading, normalization,
media feature extraction and pretrained frontends are excluded. Optional local-LM
costs are reported separately and can be added to the total. Energy remains null
without measurement. These fixture reports are not complete raw-media-to-answer costs.

## FLOPs, timing and memory

The counter uses [PyTorch registered FLOP formulas](https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py)
for matrix, convolution and attention operations, with multiply-add counted as two.
A matching formula covers CPU fused attention. FLOP counting disables opaque
Transformer/oneDNN inference paths; actual timing retains them. Each stage lists
unmodeled operators: interpolation, indexing, normalization, activations and some
nearest-neighbor work are excluded. `counted_flops` is an arithmetic subtotal, not a
complete hardware instruction count. FLOPs are not an energy measurement.

Timings use batch size one, evaluation mode, no gradients, warm-up, median and p95.
CUDA execution is synchronized. Edges, encoding, decoding and core reasoning are
measured separately. The co-located sequential pipeline also runs through actual
packets and reports windows per second. Packet equivalence is checked against
combined model inference. Untrained models are marked as cost fixtures.

CUDA incremental peak allocated memory excludes the stage's initial resident
allocation. CPU RSS is a cumulative process high-water mark, not isolated model
activation memory. A single-host measurement does not establish physical-device
concurrency, measured network latency, power or retransmissions.

## Packet sizes and transfer estimates

Accounting separates representation, float32 timestamps, application headers and
transport overhead. Application packets are fragmented using `mtu_payload_bytes`
(default 1472); overhead is charged per fragment. The example's 28-byte overhead
and 10 Mbps / 2 ms links are assumptions, not measured network properties.

`transfer_ms = 8 * wire_bytes / (1000 * bandwidth_mbps) + latency_ms`

Absent streams incur no bytes or link delay. Required Mbps and utilization use the
stride/cadence; overlapping windows produce one transmission per stride. Reports
include raw-to-representation and raw-to-wire compression ratios.

The parallel estimate takes the slowest edge's compute + encoding + transfer, then
adds core decoding + reasoning. The sequential estimate sums edge paths. Optional
`shared_uplink` models a serialized shared bottleneck after edge readiness, forwarding
edge wire volume with configured gateway overhead. These combine stage medians;
they are not measured network critical paths or steady-state queue simulations.

## Real data, checkpoints and device placement

Each dataset entry accepts the actual model `config`, explicit `inputs` frame counts,
`window_seconds`, `stride_seconds`, and a topology. For example:

```json
{"stream_devices":{"wrist_imu":"wrist_node","rgb":"camera_node"},"edges":{"wrist_node":{"bandwidth_mbps":1,"latency_ms":8,"overhead_bytes_per_packet":28,"mtu_payload_bytes":1472},"camera_node":{"bandwidth_mbps":100,"latency_ms":2}},"shared_uplink":{"bandwidth_mbps":100,"latency_ms":2}}
```

Assign every configured stream exactly once. Omit `stream_devices` for one node per
stream. Give sensors on one wearable/camera processor the same node name.
`method_profiles` overrides input/config/topology by method. The example uses it
for separately labeled ImageBind fusion on its supported subset with assumed
1024-dimensional embeddings. Pretrained ImageBind encoders are excluded, and this
is not an all-modality comparison.

Provide `manifest` and optional `sample_index` for a local test window instead of
synthetic inputs. Provide `checkpoints: {"distrixsense":"path/to/best.pt", ...}`
for trained models and normalization. Paths resolve relative to the spec. Without
a checkpoint, post-hoc VQ uses a random reference bank for inference costs only;
neither a fitted quantizer nor accuracy is claimed.

## Optional local language model

Add this field at dataset or method level:

```json
{"language":{"checkpoint":"/absolute/local/LM","mode":"sensor","query":"What activity is happening?","decode_tokens":16}}
```

Local files only are loaded. Reports include LM/active adapter storage, prompt/prefix
length, prompt/adapter timing, prefill, fixed-length generation and arithmetic.
Local prediction methods automatically use classifier prompts. Saved adapters are
restored when available; otherwise random adapter weights are labeled cost fixtures.
Generation forces the declared token count. Results depend on prompt, context,
architecture, dtype and device.

`total_deployed_with_language` adds LM, active adapter and their arithmetic to the
sensor pipeline. Without a checkpoint, the scope marks the LM not configured;
no overall LLM size or latency is invented. Add measured media encoder costs before
claiming complete raw-media-to-answer resource costs.
