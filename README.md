# DistriXSense

PyTorch implementation of the proposed distributed sensor-token interface:
timestamp-aware learnable resampling, pretrained frozen sensor encoders, peripheral
categorical selectors, a central modality-reserved embedding bank, temporal fusion,
and optional querying with a frozen local language model.

Manuscript placeholders are explicit configurable defaults. Original OPPORTUNITY
is the first experiment; OPPORTUNITY++ and OpenMarcie use a timestamped recording
adapter. Original VQ-VAE code is preserved: [legacy instructions](docs/legacy_vqvae.md).

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
python -m distrixsense smoke --output runs/smoke
```

Install a CUDA-enabled PyTorch build appropriate to your machine before selecting
`--device cuda`. The smoke suite trains all 18 main methods and ablations on small
synthetic fixtures and verifies actual binary-packet inference. These scores test
software, not scientific superiority. Completed output directories cannot be reused.

## Original OPPORTUNITY

```bash
python -m distrixsense download-opportunity --output dataset/Opportunity
python -m distrixsense prepare-opportunity \
  --root dataset/Opportunity --output dataset/prepared/opportunity \
  --task gestures --window 96 --stride 48 \
  --train-people S1 S2 --val-people S3 --test-people S4
python -m distrixsense suite \
  --manifest dataset/prepared/opportunity/manifest.jsonl \
  --config dataset/prepared/opportunity/config.json \
  --output runs/opportunity --seeds 0 1 2 --device cpu
```

The UCI archive is about 292 MB. Preparation groups channels from `column_names.txt`,
or accepts `--group-map` with **raw-file, zero-based** indices. Without metadata,
documented body-worn/object/ambient families are used. Only raw indices 1–242 are
sensor features: index 0 is time and 243–249 are annotations. Use the generated
config with actual dimensions and training label vocabulary; `configs/opportunity.json`
is an illustrative coarse-family configuration.

Targets default to majority-vote gestures including null, with actual frame labels
supervising the temporal head. Options include `--exclude-null`, `--task locomotion`,
`--task activity`, and debug-only `--max-windows N` per participant. Partitions precede
windowing; windows never cross recordings. Statistics, encoder pretraining and
k-means use training data only. These participant-independent splits differ from
the published challenge protocol; published scores are not directly comparable.

Use `train --method METHOD --seed 0` for one method. Post-hoc VQ additionally requires
`--dense-checkpoint` from a matched dense run. [Baseline definitions](docs/baselines.md)
explain the controls, published sources, and which implementations are adaptations.

## Evaluate and deploy

Each run writes `best.pt`, `config.json`, `norm_stats.json`, `metrics.json`,
`predictions.jsonl`, and training history when applicable. The suite creates
`summary.csv`, `summary.md` and `comparisons.json` with participant-level paired
tests and Holm correction. Macro F1 includes all configured classes. A bootstrap
interval requires at least two test participants; with S4 alone it is null. For
broader evaluation, prepare explicit participant folds. Overlapping windows and
random seeds are not independent participants.

```bash
python -m distrixsense infer \
  --checkpoint runs/opportunity/distrixsense/seed-0/best.pt \
  --manifest dataset/prepared/opportunity/manifest.jsonl \
  --index 0 --output runs/opportunity/inference.json
python -m distrixsense benchmark \
  --checkpoint runs/opportunity/distrixsense/seed-0/best.pt \
  --manifest dataset/prepared/opportunity/manifest.jsonl \
  --iterations 100 --output runs/opportunity/timing.json
python -m distrixsense export \
  --checkpoint runs/opportunity/distrixsense/seed-0/best.pt \
  --output runs/opportunity/deployment
python -m distrixsense evaluate \
  --checkpoint runs/opportunity/distrixsense/seed-0/best.pt \
  --manifest dataset/prepared/opportunity/manifest.jsonl \
  --drop-modalities ambient --output runs/opportunity/missing-ambient.json
python -m distrixsense attack \
  --checkpoint runs/opportunity/distrixsense/seed-0/best.pt \
  --manifest dataset/prepared/opportunity/manifest.jsonl \
  --epochs 20 --output runs/opportunity/reconstruction-attack.json
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

edge, stats = PeripheralRuntime.load("runs/opportunity/deployment/peripheral.pt")
central, _ = CentralRuntime.load("runs/opportunity/deployment/central.pt")
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
statistics. Ground-truth labels never enter the sensor prompt. Original OPPORTUNITY
has no native QA targets, so its labels are not presented as open-ended reasoning
supervision. Exact match and token F1 are available for annotated queries; semantic
correctness and unsupported-answer rates require independent evaluation annotations.
Concurrent multilabel actions and raw video/audio/LiDAR decoding require explicit
dataset-specific preprocessing; the current task head is single-label.
