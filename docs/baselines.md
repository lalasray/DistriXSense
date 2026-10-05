# Baselines and controlled comparisons

Shared preprocessing, participant splits, train-only autoencoder pretraining, frozen
encoders, initial central weights, optimizer, training budget, validation macro-F1
selection and seeds apply across the main suite. Fine-tuning is an explicit ablation.
Sweep token count, bank allocation and compression dimension to compare utility
against actual serialized bytes; equal token counts do not mean equal bandwidth.

| Method | Representation and purpose |
|---|---|
| `distrixsense` | K categorical indices, reserved central bank and shared temporal Transformer |
| `dense_tokens` | Same pooling/projection and K positions, transmitting vectors; primary index-interface control |
| `dense` | Full continuous encoder grid; full-information dense control |
| `raw` | Raw normalized observations with central encoders and same reasoner |
| `early_fusion` | Central resampling, concatenation, projection and same reasoner |
| `late_fusion` | Peripheral class probabilities averaged centrally; local CE supervises every head |
| `local_only` | Peripheral predicted labels with central majority vote |
| `bottleneck` | Learned tanh compression, configurable continuous width and optional training Gaussian noise; central decompression |
| `tokenlearner` | K adaptive temporal weighted-pooling vectors transmitted continuously |
| `posthoc_vq` | Train-only k-means on the best matched dense model's full grid; inherited frozen dense reasoner, no retraining |
| `joint_vq` | Nearest-neighbor straight-through VQ on K projected features; bank also stored peripherally |
| `deepconvlstm` | Four padded temporal convolutions and two LSTM layers on centralized observations |
| `imagebind` | Optional genuine pretrained ImageBind feature export, projection and same central Transformer |

The compression, adaptive pooling and ConvLSTM implementations are **sensor
adaptations**, not exact published-architecture reproductions or reproduced scores.
The corresponding primary sources are [BottleNet++](https://arxiv.org/abs/1910.14315),
[TokenLearner](https://arxiv.org/abs/2106.11297),
[VQ-VAE](https://arxiv.org/abs/1711.00937),
[DeepConvLSTM](https://www.mdpi.com/1424-8220/16/1/115), and
[ImageBind's official implementation](https://github.com/facebookresearch/ImageBind).

Post-hoc quantization keeps the dense grid so its effect is not confounded with a
new selector. It requires the same dense configuration, seed and training fingerprint.
The proposed categorical selector uses deterministic hard argmax with a softmax
straight-through gradient in training; deployment does not read the bank. Losses
include task CE, paired cross-modal InfoNCE, commitment, usage KL-to-uniform and
training-only reconstruction. Actual frame annotations enable temporal CE. Answer
annotations enable LM CE on answer tokens only.

| Ablation | Controlled change |
|---|---|
| `unrestricted_bank` | All bank entries available to every selector; same total capacity |
| `separate_banks` | Independent per-modality matrices; same total entries and dimensions |
| `fixed_resampling` | Timestamp interpolation without learned refinement |
| `finetune_encoders` | Update the same pretrained encoder weights downstream |
| `no_alignment` | Disable paired cross-modal alignment |
| `no_usage` | Disable marginal assignment usage regularization |

Disjoint reserved regions and independent matrices have identical representational
capacity when rows never interact. Separate banks are therefore a storage/interface
control, not an expected accuracy improvement from tensor layout. The reasoner and
alignment loss provide cross-modal interaction. Unrestricted selection tests region
sharing explicitly.

For language comparisons use the same LM and annotated queries with `sensor`,
`classifier`, and `summary` modes. The sensor adapter is trained; text-only controls
use frozen prompts. Report this difference rather than claiming matched trainable
parameter counts. Classifier prompts use predicted labels, never true labels.

ImageBind comparisons require an identical supported-modality subset and official
preprocessing. Switch/UWB/quaternion/LiDAR streams are not automatically supported.
Record feature-extraction costs separately; exported-feature timings measure fusion.
Real pretrained ImageBind weights and the optional package are not included.

Dataset sources: [OPPORTUNITY++](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.792065/full),
[OpenMarcie](https://github.com/HymalaiDFKI/OpenMarcie).
Published-model reproduction, hardware energy, semantic QA assessment and privacy
claims require experiments on real datasets and deployment hardware.

Nymeria source: https://github.com/facebookresearch/nymeria_dataset/tree/nymeria_dataset_legacy
