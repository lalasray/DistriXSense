# Data and language interfaces

## Window schema

A JSONL manifest contains one row per synchronized window; paths are relative:

```json
{"path":"s1/window000.npz","participant":"S1","split":"train","label":0,"query":"What is happening?","answer":"The worker is tightening a screw."}
```

NPZ modality arrays have shape `(T,C)`, plus optional `name__time` vectors in seconds
relative to a common window origin. Omit absent modalities. Without timestamps,
uniform 0–1 positions are used. Supply real timestamps for timing experiments.
Each participant occupies exactly one split, and every path is unique. The producer
must prevent duplicate recordings under different paths. Invalid rows are masked;
partial missing channels are interpolated within each window. Training-only moments
normalize every partition. No synthetic captions or query targets are generated.

Optional `events` in a row contains `times` and encoded `labels` lists, using `-100`
for ignored frames. The temporal head receives timestamp-nearest valid frame targets.
Without event targets this head is untrained and must not be evaluated as an event detector.

## Recording adapters

OPPORTUNITY++ and OpenMarcie adapters accept exported numeric sensor arrays or
frozen-encoder features and synchronized annotations:

```json
{"participant":"worker01","split":"train","streams":{"imu":{"values":"worker01/imu.npy","timestamps":"worker01/imu_seconds.npy"},"vision":{"values":"worker01/video_features.npy","timestamps":"worker01/video_seconds.npy"}},"annotations":[{"start":0.0,"end":10.0,"label":"tightening screw","query":"What tool is used?","answer":"A screwdriver."}]}
```

This is an illustrative schema, not a real dataset annotation. All times use a
common recording clock in seconds, dimensions are stable, and actual annotations
determine labels and queries. Raw camera calibration, video/audio transforms and
LiDAR decoding must be supplied explicitly before this adapter.

```bash
python -m distrixsense prepare-records \
  --records recordings.jsonl --dataset openmarcie \
  --window-seconds 3 --stride-seconds 1.5 \
  --output dataset/prepared/openmarcie
```

Use `--dataset opportunity++` for that extension. Each retained window must be fully
covered by exactly one annotation. Ambiguous/concurrent windows are skipped.
The generated configuration contains actual dimensions and training-derived labels.
Concurrent-action experiments require a separately specified multilabel protocol.

## ImageBind

Install the [official package](https://github.com/facebookresearch/ImageBind) in a
compatible environment and provide official pretrained weights. `extract-imagebind`
consumes per-window NPZ arrays containing **officially preprocessed inputs**, with
leading batch dimension one, and a participant/split/label JSONL manifest.
Supported sensor keys are `vision`, `audio`, `depth`, `thermal`, `imu`. Text is
excluded from sensor evidence. The published IMU convention uses six channels;
arbitrary inertial channel concatenation is not equivalent preprocessing.

```bash
python -m distrixsense extract-imagebind \
  --manifest preprocessed/manifest.jsonl \
  --weights /absolute/path/to/imagebind_huge.pth \
  --output dataset/prepared/imagebind --device cuda
python -m distrixsense train \
  --manifest dataset/prepared/imagebind/manifest.jsonl \
  --config dataset/prepared/imagebind/config.json --method imagebind \
  --output runs/imagebind --device cuda
```

Extraction records checkpoint hash and latency. Compare methods on the same supported
subset; the feature-manifest benchmark does not include encoder cost. Real pretrained
ImageBind inference is optional and is not validated by the generic CPU fixture.

## Language model

Install `pip install -e '.[language]'` and supply a local Hugging Face causal-LM
directory with tokenizer. The LM remains frozen. A two-layer adapter projects
timestamp- and modality-aware central tokens into its input dimension. CE supervises
actual answer tokens only; prefix and query positions are ignored. Gradients pass
through the frozen LM to the adapter and sensor model. Checkpoints save adapter
weights and the local LM path; generation uses greedy decoding, 64 new tokens.

The exact prompt is:

```text
Interpret the supplied sensor evidence. Answer the query concisely. If the evidence is insufficient, say 'Insufficient evidence'.
Evidence: Modality availability: <ordered boolean vector>.
Query: <annotated query>
Answer:
```

Sensor mode adds projected sensor embeddings before the prompt. Classifier mode
adds a predicted window activity and confidence; summary mode adds measured normalized
means, standard deviations and durations. Optional `--class-names` names encoded
predictions; without it classifier mode uses numeric IDs. Answers are targets only,
never inference inputs. Oversized contexts fail explicitly. Original OPPORTUNITY
has no native QA supervision, so a QA-free manifest is rejected for language training.
Tiny random local language models are integration fixtures, not useful reasoners.
