# Implementation validation

Validated on CPU using Python 3.14.6, PyTorch 2.14.1+cpu, NumPy 2.5.2 and
Transformers 4.57.6. The package declares broader compatible minimum versions;
those versions and CUDA have not been separately exercised here.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m distrixsense smoke --output runs/validation
```

All 21 tests passed. The tests cover participant leakage rejection, training-only
normalization, annotation-column exclusion, asynchronous recording preparation,
timestamp interpolation, padding invariance, all-missing sensors, frozen encoders,
selector/bank gradients, reserved-bank enforcement, packet corruption and bit widths,
checkpoint restoration, temporal-head supervision and standalone deployment exports.

The smoke suite completed training, validation, test evaluation and packet inference
for all 18 main methods/ablations. The optional ImageBind fusion branch was tested
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

Python compilation and `git diff --check` passed. Original OPPORTUNITY preparation
was validated against constructed files with the official 250-column layout. Full
real-data training, official ImageBind pretrained inference, useful local-LM QA,
hardware deployment, measured network latency and energy remain empirical validation
steps requiring their actual data, weights or instruments.
