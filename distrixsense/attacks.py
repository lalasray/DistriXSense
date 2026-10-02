"""Train-only matched-budget empirical reconstruction attacks on wire payloads.

The attacker receives serialized packets and can learn categorical embeddings
for indices. It is not given the central token bank or untransmitted latents.
These are empirical attacks, not a privacy guarantee or hardware measurement.
"""
import copy
import numpy as np
import torch
from torch import nn
from .data import ManifestDataset, collate, to_device
from .models import Resampler
from .training import seed_everything
from .transport import roundtrip


class ReconstructionAttacker(nn.Module):
    def __init__(self, model, slots=8, budget=100000):
        super().__init__()
        self.names, self.slots = model.names, slots
        self.resample = nn.ModuleDict()
        self.categorical = nn.ModuleDict()
        self.input_width = 0
        self.widths = {}
        self.classes = model.cfg.classes
        self.offsets = {}
        method = model.cfg.method
        from .models import DISCRETE, RAW, LOCAL
        for n in self.names:
            if method in DISCRETE:
                width = 8
                size = model.vocabulary if method == "unrestricted_bank" else model.bank_sizes[n]
                self.offsets[n] = 0 if method == "unrestricted_bank" else model.offsets[n]
                self.categorical[n] = nn.Embedding(size, width)
            elif method in RAW or method == "imagebind":
                width = model.cfg.modalities[n]
            elif method == "local_only":
                width = model.cfg.classes
            elif method == "late_fusion":
                width = model.cfg.classes
            elif method == "bottleneck":
                width = model.cfg.bottleneck_dim
            else:
                width = model.cfg.hidden
            self.resample[n] = Resampler(width, slots, learnable=False)
            self.widths[n] = width
            self.input_width += slots*width + 1
        self.output_width = model.cfg.resample_length*sum(model.cfg.modalities.values())
        embedding_parameters = sum(p.numel() for p in self.categorical.parameters())
        remaining = budget-embedding_parameters-self.output_width
        hidden = remaining//(self.input_width+self.output_width+1)
        if hidden < 1:
            raise ValueError("Attack parameter budget too small for this modality configuration")
        self.net = nn.Sequential(nn.Linear(self.input_width, hidden), nn.GELU(), nn.Linear(hidden, self.output_width))
        # Fixed resamplers do not execute their learned refinement parameters.
        for r in self.resample.values():
            r.refine = nn.Identity()
        self.method = method

    def forward(self, messages):
        device = next(self.parameters()).device
        parts = []
        for n in self.names:
            if n not in messages:
                channels = self.widths[n]
                parts.append(torch.zeros(1, self.slots*channels+1, device=device))
                continue
            m = messages[n]
            if n in self.categorical:
                x = self.categorical[n](m.values-self.offsets[n])
            elif self.method == "local_only":
                x = nn.functional.one_hot(m.values[..., 0].long(), self.classes).float()
            else:
                x = m.values
            x, _, present = self.resample[n]({"x": x, "time": m.times, "mask": m.valid})
            parts.append(torch.cat([x.flatten(1), present[:, None].float()], -1))
        return self.net(torch.cat(parts, -1))


@torch.no_grad()
def observations(model, dataset):
    targets = {n: Resampler(c, model.cfg.resample_length, learnable=False).to(model.cfg.device) for n, c in model.cfg.modalities.items()}
    examples = []
    model.eval()
    for sample in dataset:
        batch = to_device(collate([sample], model.cfg.modalities), model.cfg.device)
        messages, _ = model.peripheral(batch["streams"])
        received, _ = roundtrip(messages, model.names, model.cfg.device)
        ys, masks = [], []
        for n in model.names:
            x, _, p = targets[n](batch["streams"][n])
            ys.append(x.flatten(1))
            masks.append(p[:, None].expand(-1, x.numel()))
        examples.append((received, torch.cat(ys, -1), torch.cat(masks, -1), sample["participant"]))
    return examples


def run_attacks(model, saved, manifest, epochs=20, budget=100000):
    if epochs < 1:
        raise ValueError("Attack epochs must be positive")
    seed_everything(model.cfg.seed)
    sets = {s: ManifestDataset(manifest, s, model.cfg.modalities, saved["stats"]) for s in ("train", "val", "test")}
    data = {s: observations(model, ds) for s, ds in sets.items()}
    attack = ReconstructionAttacker(model, budget=budget).to(model.cfg.device)
    optimizer = torch.optim.AdamW(attack.parameters(), lr=model.cfg.lr)
    def objective(prediction, y, mask):
        return ((prediction-y).square()*mask).sum()/mask.sum().clamp_min(1)
    best, state = float("inf"), None
    for _ in range(epochs):
        attack.train()
        for i in np.random.permutation(len(data["train"])):
            messages, target, mask, _ = data["train"][i]
            optimizer.zero_grad()
            loss = objective(attack(messages), target, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(attack.parameters(), 1.)
            optimizer.step()
        attack.eval()
        with torch.no_grad():
            val = float(np.mean([float(objective(attack(m), y, mask)) for m, y, mask, _ in data["val"]]))
        if val < best:
            best, state = val, copy.deepcopy(attack.state_dict())
    attack.load_state_dict(state)
    errors, correlations = {}, {}
    with torch.no_grad():
        for messages, y, mask, person in data["test"]:
            prediction = attack(messages)
            keep = mask[0]
            rmse = float(objective(prediction, y, mask).sqrt())
            errors.setdefault(person, []).append(rmse)
            if keep.sum() > 1:
                a, b = prediction[0, keep].cpu().numpy(), y[0, keep].cpu().numpy()
                if a.std() > 1e-8 and b.std() > 1e-8:
                    correlations.setdefault(person, []).append(float(np.corrcoef(a, b)[0, 1]))
    return {"method": model.cfg.method, "attack": "wire-only categorical/continuous MLP reconstruction",
            "epochs": epochs, "parameter_budget": budget, "actual_parameters": sum(p.numel() for p in attack.parameters()),
            "participant_mean_normalized_rmse": {p: float(np.mean(v)) for p, v in errors.items()},
            "normalized_rmse": float(np.mean([np.mean(v) for v in errors.values()])),
            "participant_mean_pearson": {p: float(np.mean(v)) for p, v in correlations.items()},
            "validation_mse": best,
            "limitations": "Inputs are training-normalized signals. Equal maximum parameter budget; actual sizes differ. No central bank access. No sensitive-attribute labels supplied, so no attribute attack is claimed. No formal privacy guarantee."}
