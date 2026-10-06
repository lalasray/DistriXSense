# Implementation validation

Validated on CPU using Python 3.14.6, PyTorch 2.14.1+cpu, NumPy 2.5.2 and
Transformers 4.57.6. CUDA functional validation also passes with PyTorch
2.14.1+cu130 on an RTX PRO6000 Blackwell96GB: all22 methods on GPU and on
CPU-edge/GPU-core placement (44 checks), plus real Gemma generation. The package
declares broader compatible minimum versions; those versions have not been
separately exercised here.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m distrixsense smoke --output runs/validation
```

All 55 tests passed. The tests cover participant leakage rejection, training-only
normalization, annotation-column exclusion, asynchronous recording preparation,
timestamp interpolation, padding invariance, all-missing sensors, frozen encoders,
selector/bank gradients, reserved-bank enforcement, packet corruption and bit widths,
checkpoint restoration, temporal-head supervision and standalone deployment exports.
They also check source-based shape accounting, explicit LLM shape reuse and safe
resumption of completed profiling cases.
Transport tests distinguish serialization fidelity from re-encoding differences
between hardware backends.

The smoke suite completed training, validation, test evaluation and packet inference
for all 21 main methods/ablations. The optional ImageBind fusion branch was tested
with numeric fixtures, but genuine pretrained ImageBind extraction requires the
official package and weights and was not run. Published baseline reproductions are
not claimed; adaptation differences are documented in baselines.md.

Language tests construct a tiny random GPT-2 and local tokenizer without downloads.
They verify frozen-LM gradients, trainable prefix/selector gradients, answer-only
loss masking, ground-truth exclusion from prompts, generation, joint sensor/adapter
training and checkpoint restoration. This is a software fixture, not useful QA.

A separate ten-epoch synthetic training check confirmed that the proposed pipeline
can learn the fixture. Its checkpoint was exercised through `infer`, `benchmark`,
`evaluate --drop-modalities ambient`, `export` and a train/validation/test reconstruction
attack. Generated local artifacts are under `runs/learning-check`; they are ignored
by Git and are not experimental paper results. Suite artifacts are under
`runs/validation`, including per-method packet timings and a summary table.

Dataset tests cover every catalog modality across OPPORTUNITY++, OpenMarcie and
Nymeria with generated local fixtures, including native image/depth and video
decoding, stereo audio slicing, OpenPose missing detections, variable point counts,
XSens NPZ arrays, integer timestamp precision, annotation preservation, split
leakage guards and feature export into the model. Device-to-timecode conversion is
mocked; native Aria VRS and Open3D PLY/PCD decoding have not been exercised here.

No datasets were downloaded. No real-data performance or release-wide compatibility
is claimed from these fixtures. Full real-data training, native Aria SDK validation,
official ImageBind pretrained inference, useful local-LM QA, hardware deployment,
measured network latency and energy require the actual data, weights or instruments.

Additional temporal tests verify TCN causality, PatchTST tail coverage and missing-channel exclusion, gradient flow, masked-input invariance and checkpoint restoration. The 21-method smoke suite completed in `runs/temporal-baselines-validation`.

The modality/LLM matrix is validated with two different tiny local causal-LM widths, including per-modality cost tables and core-specific training/checkpoints. BF16-core gradients and native chat-template prefix handling are tested. Downloaded Gemma3 1B IT and Qwen3 4B/8B cores have also executed on source-documented synthetic feature shapes. See the [downloaded-core experiment](published_dummy_tests.md) for reproduction, measurement reuse and input assumptions. These cost tests do not require annotated recordings and do not establish useful sensing QA.
