from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


METHODS = (
    "distrixsense", "dense", "dense_tokens", "raw", "early_fusion", "late_fusion", "local_only",
    "bottleneck", "tokenlearner", "posthoc_vq", "joint_vq", "deepconvlstm",
    "unrestricted_bank", "separate_banks", "fixed_resampling", "finetune_encoders",
    "no_alignment", "no_usage", "imagebind", "tcn", "ts_transformer", "patchtst",
)


@dataclass
class Config:
    modalities: dict[str, int] = field(default_factory=lambda: {"imu": 6, "ambient": 2})
    classes: int = 3
    method: str = "distrixsense"
    resample_length: int = 32
    tokens: int = 8
    hidden: int = 32
    bank_size: int = 32  # entries PER modality; all ablations preserve total capacity
    bank_sizes: dict[str, int] = field(default_factory=dict)
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    bottleneck_dim: int = 4
    channel_noise: float = 0.0
    tcn_kernel: int = 3
    patch_length: int = 8
    patch_stride: int = 4
    modality_dropout: float = 0.15
    temperature: float = 1.0
    commitment: float = 0.1
    alignment: float = 0.05
    usage: float = 0.01
    reconstruction: float = 0.1
    language_weight: float = 1.0
    event_weight: float = 0.1
    epochs: int = 20
    pretrain_epochs: int = 5
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 5
    seed: int = 0
    device: str = "cpu"
    protocol_overhead: int = 0
    bandwidth_mbps: float = 10.0
    link_latency_ms: float = 0.0

    def validate(self):
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method}; choose from {METHODS}")
        if not self.modalities or any(not k or v < 1 for k, v in self.modalities.items()):
            raise ValueError("Provide named modalities with positive channel counts")
        if any("." in k for k in self.modalities):
            raise ValueError("Modality names cannot contain dots (PyTorch module keys)")
        for name in ("classes", "resample_length", "tokens", "hidden", "bank_size", "heads",
                     "layers", "bottleneck_dim", "epochs", "pretrain_epochs", "batch_size", "patience",
                     "tcn_kernel", "patch_length", "patch_stride"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden % self.heads or self.tokens > self.resample_length:
            raise ValueError("hidden must divide heads; tokens cannot exceed resample_length")
        if self.method == "patchtst" and (self.patch_length > self.resample_length or self.patch_stride > self.patch_length):
            raise ValueError("PatchTST requires patch_stride <= patch_length <= resample_length")
        if not 0 <= self.modality_dropout < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1)")
        if self.temperature <= 0 or self.bandwidth_mbps <= 0 or self.protocol_overhead < 0:
            raise ValueError("Invalid temperature or link settings")
        for name in ("channel_noise", "commitment", "alignment", "usage", "reconstruction", "language_weight", "event_weight", "weight_decay", "link_latency_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if set(self.bank_sizes) - set(self.modalities) or any(v < 1 for v in self.bank_sizes.values()):
            raise ValueError("Invalid bank allocation")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()
