"""Loadable runtimes registering only the modules deployed on each side."""
import torch
from torch import nn
from .benchmark import deployed_modules
from .config import Config
from .models import DistributedModel


class Runtime(nn.Module):
    def __init__(self, model, side):
        super().__init__()
        self.cfg, self.names = model.cfg, model.names
        self.offsets, self.bank_sizes = model.offsets, model.bank_sizes
        self.vocabulary = model.vocabulary
        self.register_buffer("posthoc_ready", model.posthoc_ready.detach().clone())
        modules = deployed_modules(model)[0 if side == "peripheral" else 1]
        selected = {id(module) for module in modules}
        for name, module in model.named_children():
            if id(module) in selected:
                self.add_module(name, module)
        self.eval()

    def train(self, mode=True):
        if mode:
            raise RuntimeError("Deployment runtimes are inference-only; train the research model")
        return super().train(False)

    @classmethod
    def load(cls, path, device="cpu"):
        saved = torch.load(path, map_location=device, weights_only=True)
        cfg = Config(**{**saved["config"], "device": device}).validate()
        prototype = DistributedModel(cfg).to(device)
        prototype.load_state_dict(saved["state"], strict=False)
        return cls(prototype), saved["normalization"]

    def save(self, path, normalization):
        torch.save({"config": self.cfg.to_dict(), "normalization": normalization, "state": self.state_dict()}, path)


class PeripheralRuntime(Runtime):
    def __init__(self, model):
        super().__init__(model, "peripheral")

    features = DistributedModel.features
    region = DistributedModel.region
    _mask = DistributedModel._mask

    @torch.no_grad()
    def forward(self, streams):
        messages, _ = DistributedModel.peripheral(self, streams, drop=False)
        return messages


class CentralRuntime(Runtime):
    def __init__(self, model):
        super().__init__(model, "central")
        self.register_buffer("device_anchor", torch.empty(0, device=model.cfg.device))

    _mask = DistributedModel._mask

    @torch.no_grad()
    def forward(self, messages, batch_size=1):
        return DistributedModel.central(self, messages, batch_size=batch_size)
