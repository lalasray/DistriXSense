# DistriXSense

PyTorch implementation of the proposed distributed sensor-token interface:
timestamp-aware learnable resampling, pretrained frozen sensor encoders, peripheral
categorical selectors, a central modality-reserved embedding bank, temporal fusion,
and optional querying with a frozen local language model.

The data layer targets **OPPORTUNITY++, OpenMarcie and Nymeria**. It reads local
recordings with explicit manifests and preserves native modality shapes, clocks,
device locations, missing data and annotation tracks. Dataset downloaders and the
old VQ-VAE code have been removed.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
python -m distrixsense smoke --output runs/smoke
```

Install a CUDA-enabled PyTorch build appropriate to your machine before selecting
`--device cuda`. The smoke suite trains all 21 main methods and ablations on small
synthetic fixtures and verifies actual binary-packet inference. These scores test
software, not scientific superiority. Completed output directories cannot be reused.

## Local datasets

Start from a recording template in [configs/datasets](configs/datasets), replacing
placeholder paths, clocks, recording spans and participants with your local files.
Templates describe the schema; they are not release-specific file inventories.
All declared available input streams are loaded by default.

```bash
pip install -e '.[media]'  # image/video/audio file codecs
python -m distrixsense inspect-dataset \
  --records recordings.jsonl --dataset openmarcie
python -m distrixsense prepare-records \
  --records recordings.jsonl --dataset openmarcie \
  --feature-policies configs/datasets/feature_policies.json \
  --target-track activity --output dataset/prepared/openmarcie
python -m distrixsense suite \
  --manifest dataset/prepared/openmarcie/manifest.jsonl \
  --config dataset/prepared/openmarcie/config.json \
  --output runs/openmarcie --seeds 0 1 2 --device cpu
```

Use `--dataset opportunity++` or `--dataset nymeria` for the other datasets.
The raw PyTorch loaders support numeric sensors, RGB/depth/thermal images and video,
audio, structured poses, point clouds and annotation/text tracks. Aria VRS requires
the optional `aria` extra in a compatible Python environment; PLY/PCD uses
`pointcloud`. Nothing downloads dataset files.

The existing model consumes numeric feature sequences. Export requires explicit
feature policies for raw media, or locally computed encoder features. The supplied
pooling/spectral/statistical policies are integration baselines, not pretrained
multimodal encoders. The raw loader retains concurrent annotations; the single-label
export defaults to excluding ambiguous windows. See the
[data interface](docs/data_and_language.md) for native formats and synchronization.

Use `train --method METHOD --seed 0` for one method. Post-hoc VQ additionally requires
`--dense-checkpoint` from a matched dense run. [Baseline definitions](docs/baselines.md)
explain the controls and published sources. The default suite now also includes
`tcn`, `ts_transformer` and `patchtst` temporal baselines.

## Evaluate and deploy

Use `profile-suite` for model size, per-edge/core FLOPs and timings, packet volume
and modeled transfer costs. The [profiling guide](docs/profiling.md) explains CPU,
GPU and mixed runs, local checkpoints and the illustrative three-dataset spec.

Use `matrix-suite` for each sensing modality/group with multiple local LLM cores.
The [LLM comparison guide](docs/llm_matrix.md) covers matched training/evaluation,
complete core costs and the configured 1B, 4B and 8B checkpoint choices.

Each run writes `best.pt`, `config.json`, `norm_stats.json`, `metrics.json`,
`predictions.jsonl`, and training history when applicable. The suite creates
`summary.csv`, `summary.md` and `comparisons.json` with participant-level paired
tests and Holm correction. Macro F1 includes all configured classes. A bootstrap
interval requires at least two test participants; with one test participant it is null. For
broader evaluation, prepare explicit participant folds. Overlapping windows and
random seeds are not independent participants.

```bash
python -m distrixsense infer \
  --checkpoint runs/openmarcie/distrixsense/seed-0/best.pt \
  --manifest dataset/prepared/openmarcie/manifest.jsonl \
  --index 0 --output runs/openmarcie/inference.json
python -m distrixsense benchmark \
  --checkpoint runs/openmarcie/distrixsense/seed-0/best.pt \
  --manifest dataset/prepared/openmarcie/manifest.jsonl \
  --iterations 100 --output runs/openmarcie/timing.json
python -m distrixsense export \
  --checkpoint runs/openmarcie/distrixsense/seed-0/best.pt \
  --output runs/openmarcie/deployment
python -m distrixsense evaluate \
  --checkpoint runs/openmarcie/distrixsense/seed-0/best.pt \
  --manifest dataset/prepared/openmarcie/manifest.jsonl \
  --drop-modalities ambient --output runs/openmarcie/missing-ambient.json
python -m distrixsense attack \
  --checkpoint runs/openmarcie/distrixsense/seed-0/best.pt \
  --manifest dataset/prepared/openmarcie/manifest.jsonl \
  --epochs 20 --output runs/openmarcie/reconstruction-attack.json
```

Inference serializes and decodes packets before central reasoning. Indices are
bit-packed; headers include modality, representation type, shape and float32
timestamps. Missing modalities send no packet. Configure `protocol_overhead`,
`bandwidth_mbps` and `link_latency_ms` for a particular link. Payload counts use
actual encoded bytes. Host processing latency is measured; link latency is modeled.
Data loading and normalization are outside the timed path. Energy, CPU peak memory
and FLOPs require external instrumentation and are not fabricated.

Deployment exports contain loadable PyTorch runtimes, normalization and a bank hash,
rather than MCU binaries. Research checkpoints retain unused comparison
modules; deployment inventories count the selected inference modules. Categorical
selectors require no edge bank. Nearest-neighbor VQ controls do require an edge copy.
The reconstruction attacker sees only wire representations and uses a matched
maximum parameter budget, training-only fitting and validation selection. Its
participant-averaged normalized RMSE is an empirical leakage measurement; no
sensitive-attribute attack or formal privacy guarantee is inferred.

```python
from distrixsense.deployment import PeripheralRuntime, CentralRuntime
from distrixsense.transport import roundtrip

edge, stats = PeripheralRuntime.load("runs/openmarcie/deployment/peripheral.pt")
central, _ = CentralRuntime.load("runs/openmarcie/deployment/central.pt")
# streams is a normalized, masked batch from the data interface.
messages = edge(streams)
received, packets = roundtrip(messages, edge.names, "cpu")
prediction = central(received)
```

The exported proposed peripheral runtime contains neither bank weights nor the
central reasoner. `roundtrip` is a batch-size-one local transport demonstration;
use `serialize` / `deserialize` over your network transport for separate machines.

## Other datasets and language

[Data and language interfaces](docs/data_and_language.md) describe the schemas,
timestamp conventions, optional ImageBind extraction and text-only controls.

```bash
pip install -e '.[language]'
python -m distrixsense train \
  --manifest dataset/prepared/openmarcie/manifest.jsonl \
  --config dataset/prepared/openmarcie/config.json \
  --output runs/openmarcie/language \
  --language-model /absolute/path/to/local/huggingface-checkpoint \
  --language-mode sensor
```

The LM loads locally and is frozen; its adapter learns from real QA annotations.
`classifier` mode supplies predicted activity labels; `summary` supplies measured
statistics. Ground-truth labels never enter the sensor prompt. Language training requires explicit question/answer annotations. Exact match and token F1 are available for annotated queries; semantic
correctness and unsupported-answer rates require independent evaluation annotations.
The raw loader preserves concurrent actions; the current classification head is single-label.
