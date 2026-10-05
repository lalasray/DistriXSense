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

## Local recording loaders

Only `opportunity++`, `openmarcie` and `nymeria` are accepted. The classes
`OpportunityPlusPlusDataset`, `OpenMarcieDataset` and `NymeriaDataset` use the same
explicit JSONL recording schema. Paths resolve relative to the manifest.

```json
{"dataset":"openmarcie","recording":"session01","participant":"worker01","split":"train","start":0,"end":10,"streams":{"wrist_imu":{"modality":"imu","device":"wrist","format":"array","path":"imu.npz","key":"values","timestamp_key":"timestamps","time_unit":"ns","time_origin":1000000000}},"annotations":[{"start":0,"end":10,"track":"activity","label":"tightening screw"}]}
```

This is an illustrative schema. Set real clocks and labels yourself; rates, camera
calibration, coordinate transformations and recording lengths are never guessed.
See the three `*.recordings.jsonl.example` files under `configs/datasets` for
modality declarations. Repeated devices/cameras use independent stream names.
Declare every participant in multi-person recordings through `participants`.
Participant and source-file reuse across splits is rejected before windowing.

```python
from distrixsense.datasets import NymeriaDataset, multimodal_collate

dataset = NymeriaDataset("recordings.jsonl", split="train",
                        window_seconds=3, stride_seconds=1.5)
sample = dataset[0]
batch = multimodal_collate([sample], modalities=dataset.modalities)
```

Images retain T,C,H,W axes; poses retain joint/coordinate axes; audio retains all
channels; point clouds retain variable point counts. Collation supplies temporal,
finite-value, availability and point masks. Differing image resolutions need an
explicit transform. Missing streams are omitted or declared `available: false`.
All overlapping annotations, their tracks and original intervals are preserved.
Text defaults to target annotations; text as evidence needs explicit `role: input`.

| Dataset | Supported modality families |
| --- | --- |
| OPPORTUNITY++ | Body/shoe/object inertial sensors, orientation, UWB, ambient accelerometers/switches, RGB video, BODY25 poses, annotation/subtitle tracks |
| OpenMarcie | Inertial/magnetic/environmental/spectral/thermal sensors, ego/exo RGB-D, LiDAR depth/points, multichannel audio, pose/object/position tracks and annotations |
| Nymeria | Available Aria camera/inertial/magnetic/barometric/audio streams per device, gaze, trajectories, scene points, XSens body motion/contact streams and narration |

Availability depends on the actual recording/release. Catalog entries do not
invent absent streams. Additional channels use an explicit `kind` and modality.

### Local formats and clocks

- `array`: NPY or named NPZ arrays, explicit timestamps or sample rate.
- `csv` / `table`: explicit feature columns and time column; named columns with
  `header: true`. Set `integer_timestamps: true` for headered epoch-nanosecond CSV.
- `opportunity_dat`: sensor columns 1–242 only (zero-based); labels cannot enter
  features. `opportunity_sensor_streams` in `datasets.native` reads column metadata.
- `images`, `video`, `audio`: local image frames, timestamped video, and audio
  slices. Install `.[media]` for Pillow/PyAV/soundfile.
- `openpose`: JSON frames, explicit person index, missing detections kept invalid.
- `point_frames` / `static_points`: NPY, numeric XYZ/CSV, or PLY/PCD with the
  optional `.[pointcloud]` dependency. Static maps are marked as static context.
- `vrs`: native Aria through `.[aria]`, selecting a stream label/ID and time domain.
  Use common TIME_CODE for synchronized devices. SDK availability depends on Python.
- XSens: `nymeria_xsens_streams` exposes timestamp-aligned numeric arrays from
  `body/xdata.npz`, preserving structured coordinates and excluding static metadata.
- Annotation files: JSON, JSONL, CSV with field mapping, or SRT.

Timestamps use `time_unit` (s/ms/us/ns), `time_origin` in native units,
`clock_scale` and `offset_seconds`. Integer origins are subtracted before float
conversion. Windows are half-open and returned times are relative to window start.
For device-clock arrays/tables, `clock_vrs` plus `timecode_origin_ns` enables Aria
device-to-timecode conversion. Coordinate frames and calibration are retained as
metadata; XSens and Aria spatial frames are not automatically aligned.

### Export into the current model

```bash
python -m distrixsense prepare-records \
  --records recordings.jsonl --dataset nymeria \
  --window-seconds 3 --stride-seconds 1.5 \
  --feature-policies configs/datasets/feature_policies.json \
  --target-track activity --output dataset/prepared/nymeria
```

All input streams must produce numeric T,C features: signal/features can pass
through; raw media require explicit spatial pooling, pose flattening, audio
log-spectrum or point-moment policies, or your own precomputed encoder features.
The provided policies do not claim pretrained semantic representations.
Strict target selection requires one activity covering the entire window; optional
`--target-policy majority` selects by total overlap duration. Concurrent annotations
remain in exported metadata. Train/val/test partitions are required, and labels
derive from training only. The raw loader itself does not discard unlabelled or
concurrent windows.

Official dataset references: [OPPORTUNITY++](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.792065/full),
[OpenMarcie](https://github.com/HymalaiDFKI/OpenMarcie),
[original Nymeria release](https://github.com/facebookresearch/nymeria_dataset/tree/nymeria_dataset_legacy).
The upstream Nymeria branch name refers to that release, not retained local legacy code.

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
never inference inputs. Oversized contexts fail explicitly. A QA-free manifest is rejected for language training.
Tiny random local language models are integration fixtures, not useful reasoners.
