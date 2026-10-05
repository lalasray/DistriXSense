# Sensing modality × LLM core comparisons

`matrix-suite` expands dataset × sensing-modality set × core × sensor baseline.
Training uses the same manifest partitions, class vocabulary, seeds, optimizer
budget and validation selection for every core. Sensor prefix adapters are trained
separately for each LLM; the LLM itself stays frozen.

The example [configuration](../configs/llm_matrix.json) has three core choices:

| Name | Official model | Nominal size |
| --- | --- | --- |
| `gemma3_1b` | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | 1B |
| `qwen3_4b` | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4B |
| `qwen3_8b` | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8B |

These are concrete supported model choices, not a claim that they are the newest
available models. Gemma and Qwen differ in architecture/training as well as size;
this comparison does not isolate parameter count alone. Actual parameter counts
come from loaded tensors, not the rounded size names. The language extra requires
Transformers >=4.51 for the selected architectures.

Replace `/LOCAL_MODELS/...` placeholders with existing local checkpoint directories.
No model or dataset downloads occur. Example sensor dimensions remain illustrative
until replaced by the prepared dataset configurations. BF16 cores use FP32 trainable
adapters; projected prefix tensors are cast to the core's embedding dtype while
preserving gradients. Supported core dtypes are float32, bfloat16, float16 and auto.

Inspect the full plan before loading weights or data:

```bash
python -m distrixsense matrix-suite \
  --spec configs/llm_matrix.json --output runs/my-matrix-plan --dry-run
```

The example generates 159 core/modality scenarios across OPPORTUNITY++, OpenMarcie
and Nymeria: all available configured modalities plus every modality on its own,
with each of three LLM cores. It selects the 21 main methods/ablations; ImageBind
requires a separate matched subset/feature manifest. Running all combinations is
substantial work, so `--methods` can select a smaller comparison without changing
the modality/core matrix.

## Deployment costs

```bash
python -m distrixsense matrix-suite \
  --spec configs/llm_matrix.json --output runs/core-costs --mode profile \
  --targets cpu cuda mixed --methods distrixsense dense_tokens deepconvlstm
```

The profile summaries identify `dataset`, `modality_set`, `core_name`, `method`
and execution target. Sensor-only counts remain available alongside `full_parameters`,
`full_weight_bytes`, `full_counted_flops`, LLM generation time and a complete
pipeline latency estimate. Device CSVs include a `core_language` row in addition
to sensor edges/core. JSON retains adapter/prompt, prefill and fixed-token generation
costs. Overall latency estimates add stage medians, not a measured physical network.
See [profiling scope and limits](profiling.md) for excluded media frontends,
unmodeled FLOPs, peak-memory definitions and unavailable CUDA cases.

## Activity and question-answer evaluation

Add a local prepared `manifest` to each dataset entry and replace its `config` with
the matching model configuration. Question/answer annotations must exist in training
data; answers are targets and never prompt evidence. Then run:

```bash
python -m distrixsense matrix-suite \
  --spec configs/llm_matrix.json --output runs/core-evaluation --mode train \
  --methods distrixsense dense_tokens deepconvlstm
```

Training uses `config.device` for each dataset (cpu/cuda). The `--targets` option
controls profiling only. Every modality set changes the input architecture and is
trained separately. This is a single-modality/group comparison, not an inference-only
sensor-dropout experiment on a model trained with all modalities. Existing
`evaluate --drop-modalities` supports that separate robustness experiment.

Each core/modality/method/seed gets its own checkpoint, predictions and QA answers.
`matrix_results.csv` / JSON retain activity metrics, transmission bytes and language
metrics; per-scenario summaries retain seed aggregation and participant comparisons.
Language evaluation provides exact match/token F1 on annotated answers. Semantic
reasoning quality requires additional human/reference evaluation; fixture tests do
not establish real-data QA quality.

Core `prompt_style: auto` uses the tokenizer's local native chat template when
available, otherwise the common plain prompt. Qwen thinking is disabled in templates
that support that setting. Common instruction/evidence/query content is identical;
template/tokenizer-specific wrappers can differ. `plain` and `chat` are explicit
alternatives. Local prediction methods use classifier prompts automatically.

## Custom modality sets and restored models

```json
{"modality_sets":{"all":"all","inertial":["imu","accelerometer","gyroscope"],"vision":["rgb","depth"]},"include_each_modality":true,"include_leave_one_out":false}
```

Names must identify configured input streams. Repeated devices use their separate
stream names. Enable `include_leave_one_out` for retrained models with each stream
excluded in turn. Unsupported method/modality combinations are recorded explicitly.

Cost profiling accepts `scenario_checkpoints`, keyed by the generated scenario ID
(for example `nymeria__only_imu__qwen3_4b`) and then method name. These checkpoints
must match that modality set and core. A checkpoint trained with all modalities
cannot silently stand in for a single-modality retrained model. Paths resolve
relative to the original specification. Training produces fresh models and adapters.

The integration tests use two locally constructed random GPT-2 fixtures with widths
16 and 24. They exercise the matrix, size differences, BF16 gradients, chat templates,
training, generation and checkpoint isolation without downloads. They are not the
configured 1B/4B/8B models and are not useful reasoning models.
