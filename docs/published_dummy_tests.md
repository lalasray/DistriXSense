# Downloaded-core / synthetic-input cost experiment

This experiment downloads **models only**, then profiles random sensor features with
frozen pretrained Gemma 3 1B IT, Qwen3 4B and Qwen3 8B. It does not download recordings
or measure task accuracy. Sensor encoders, codebooks and sensor-to-LLM adapters are
random cost fixtures. These runs cannot demonstrate that the proposed method works
better than any baseline on recognition or question answering.

## Reproduce

```bash
.venv/bin/python tools/download_llm_cores.py
.venv/bin/python tools/build_published_profiles.py
for core in gemma3_1b qwen3_4b qwen3_8b; do
  .venv/bin/python tools/run_published_profiles.py --core "$core" --output "runs/published-dummy/$core"
done
```

The downloader pins model revisions, records file lengths and remote LFS hashes in
`checkpoints/download_manifest.json`, and uses existing Hugging Face authorization
for gated Gemma. Recorded remote hashes are provenance, not a separately computed
local checksum. All inference is local-files-only. Checkpoint files and result
artifacts are ignored by Git.

Use `--core gemma3_1b`, `--core qwen3_4b` or `--core qwen3_8b` to run separate core
reports with separate `--output` directories. Default: batch one, three timing
samples, one warm-up, eight CPU threads, BF16 core weights, FP32 sensor networks
and adapter, exactly eight generated tokens. Three samples are a cost smoke test;
they do not support stable tail-latency claims. Increase `--iterations` for those.

## Where input sizes come from

The generator writes `configs/published_matrix.json` and
`runs/published-dummy/modality_inventory.{json,csv}`. Each inventory row includes its
source, three-second decoded shape, decoded numeric bytes, feature shape/bytes,
and explicit assumptions. These are **window sizes**, not whole-dataset archive
sizes. Per-modality compressed corpus sizes were not established and remain null.
Image uint8 and other numeric float32 representations are accounting conventions,
not assertions about the release's file encoding. Missing samples and compression
will change real storage volumes.

* [OPPORTUNITY++ paper](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.792065/full):
  242 sensor attributes partitioned into body145, object60 and ambient37, plus
  640×480 video at10fps and BODY25 poses. Sensor subsystems use aggregate gateways;
  this does not invent a physical-device assignment for each of the145 attributes.
  The30Hz synchronized fixture and single-person pose selection are assumptions.
* [OpenMarcie paper](https://arxiv.org/html/2603.02390v1): Table2 channel counts are
  represented separately for bicycle and printer settings. They are not combined
  into an impossible simultaneous capture. Image224×224,8fps and most non-audio
  rates are explicit fixture settings, not documented native resolutions/rates.
  Stereo audio uses the16kHz benchmark representation. The RGBD-IMU28-channel
  decomposition needs release-schema verification before real-data use.
* [Nymeria paper](https://arxiv.org/html/2406.09905v1): AppendixA supplies head and
  wrist image/rate profiles, separate800/1000Hz IMUs, seven-channel48kHz audio,
  magnetometer10Hz, barometer50Hz and body motion240Hz. AppendixB supplies1kHz
  trajectories. Variable scene-point counts remain unknown. Body output widths,
  gaze representation and observer availability have explicit fixture assumptions.

All families use shared deterministic feature conventions: image8×8 pooling,
audio512-point STFT with256-sample hop, flattened numeric streams and12-dimensional
point moments. **Dummy features are generated directly at these resulting shapes.**
Media decoding, feature extraction, OpenPose, MPS and XSens estimation costs are
excluded, and derived streams may be privileged targets for some real tasks.
Consequently the `all` group is a deployment stress test, not a leakage-safe task
protocol. The baseline named `raw` transfers these uncompressed **features**, not
full-resolution pixels or waveform samples. Inventory raw bytes are separate.

## Matrix and measurement interpretation

There are2,583 combinations: three pretrained cores ×21 implemented methods and
ablations ×41 all/single-family settings over three datasets (OpenMarcie has two
settings). ImageBind is excluded: the repository's supplementary fusion-only
branch would require actual pretrained ImageBind features and encoder accounting
for a defensible end-to-end comparison. Methods are repository implementations,
not automatically exact reproductions of all published baseline architectures.

To avoid repeating the same large model computation, identical frozen-core
checkpoint/device/dtype/batch/sequence/width/decode shapes share measured prefill
and generation stages. Every report exposes `language_shape_id` and
`language_measurement`: `measured_this_case` or `shape_matched_reuse`. Reused values
are shape-matched estimates, not independent executions or independent timing
samples. Sensor inference, adapter/prompt construction, wire serialization and
per-edge/core storage are evaluated for every combination. This assumes dense
core execution cost is predominantly shape-dependent; it does not generalize to
content-dependent routing or variable early stopping. Decode length is fixed.

JSON contains counted FLOPs and unsupported operations, per-node parameters and
bytes, stage times, packet sizes and estimated transfer time. Counted FLOPs are a
subtotal, not a complete operation count. Network assumptions are10Mbps,2ms
one-way latency and28-byte packet overhead, not a measured physical network.
GPU/mixed results are skipped explicitly when CUDA is unavailable; no CPU number
is relabeled as GPU performance. CPU RSS is cumulative process memory and does
not isolate LLM peak activations or energy consumption. Network timing and summed
pipeline timing are estimates. No energy or hardware-specific GPU estimates are
invented.

Classifier widths are12 for OpenMarcie and an assumed10 for the other fixtures;
they do not define official label taxonomies. OPPORTUNITY++ sensor costs are by
body/object/ambient subsystem, not separate acceleration/gyro/magnetic channels.
Nymeria observer and derived outputs are included for cost coverage; not every
real application can use them as input. CSV per-device rows identify aggregate
`*_gateway` nodes explicitly. Hardware-specific per-sensor processors require a
real deployment map, rather than assigning an invented chip to each channel.

The combined CSV also reports inverse batch-one pipeline latency (windows/s) and
counted GFLOP/s. The latter divides the registered FLOP subtotal by elapsed time;
it is not peak device FLOPS, and the pipeline inverse is not measured concurrent
or saturated-server throughput. Full parallel-edge/network-plus-core latency is
available in `full_parallel_network_estimate_ms`.

## GPU environment on this workstation

The initial `nvidia-smi` failure was due to sandbox device restrictions. Outside
the sandbox, the RTX PRO6000 Blackwell96GB and driver595.91.07 work correctly.
`.venv-gpu` contains PyTorch2.14.1+cu130 and Transformers4.57.6; `.venv` retains
PyTorch2.14.1+cpu for CPU measurements. No system driver changes were needed.
Installation follows the [official PyTorch CUDA instructions](https://pytorch.org/get-started/locally/).

```bash
.venv/bin/python -m venv .venv-gpu --system-site-packages
.venv-gpu/bin/python -m pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv-gpu/bin/python -m pip install -e '.[language]' transformers==4.57.6
.venv-gpu/bin/python tools/check_gpu.py
.venv-gpu/bin/python tools/run_gpu_profiles.py
```

GPU commands must have device access (outside the Codex sandbox on this host).
The GPU driver script waits for the three CPU suites, runs the regression suite,
then runs each core on GPU and mixed CPU-edge/GPU-core configurations. It writes
separate `*_gpu` directories and combines all three targets into the final report:
7,749 combinations in total. `check_gpu.py` is functional validation only; its
single-iteration timings are not the final benchmark.

Interrupted suites support `--resume`, preserving completed reports and rejecting
changed input specifications, timing settings or PyTorch builds. A resumed process
may remeasure a previously encountered core shape. `language_measurement_id`
identifies a particular measurement and its reused copies, whereas
`language_shape_id` identifies the computational shape. Initial CPU-only runtime
GPU skips remain historical records in availability.json, not a statement that
the GPU is unavailable after installing the CUDA environment.

CUDA edge and core stages are measured sequentially on the same workstation GPU;
per-node accounting does not imply multiple physical GPUs were deployed. Parallel
network/device schedules are projections, not observed simultaneous distributed
execution. Environment package snapshots and CPU/GPU hardware records are stored
beside the reports. The resumed CPU and full GPU timing suites run sequentially.

## Completed validation

All7,749 cost combinations completed (2,583 each CPU/GPU/mixed), with286 measured
core-stage records and7,463 explicitly reused core-stage records. All55 regression
tests pass. The combined report is `runs/published-dummy/RESULTS.md`; exhaustive
coverage and accounting checks are in `validation.json`.

Three mixed-device post-hoc VQ reports have an8.10e-5 logit difference when CPU
edge encoding is compared with re-encoding on GPU. The original profiler conflated
this with packet fidelity. It now compares the same edge outputs sent directly
and serialized to the core, and reports the cross-device comparison separately.
Independent rechecks across861 sensor configurations (2,583 mixed reports) gave
exactly zero transport difference. Original compute timings and FLOP measurements
were preserved; corrected validation fields remain alongside the original
cross-device discrepancy. `tools/verify_mixed_transport.py` reproduces that audit.
