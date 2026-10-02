import math
import torch
from torch import nn
from torch.nn import functional as F
from .transport import Message

DISCRETE = {"distrixsense", "joint_vq", "unrestricted_bank", "separate_banks",
            "fixed_resampling", "finetune_encoders", "no_alignment", "no_usage", "posthoc_vq"}
RAW = {"raw", "early_fusion", "deepconvlstm"}
LOCAL = {"late_fusion", "local_only"}


class Resampler(nn.Module):
    """Timestamp-aware interpolation followed by a learned temporal residual."""
    def __init__(self, channels, length, learnable=True):
        super().__init__()
        self.length, self.learnable = length, learnable
        self.refine = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1, groups=channels),
                                    nn.GELU(), nn.Conv1d(channels, channels, 1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        if not learnable:
            self.requires_grad_(False)

    def forward(self, stream):
        x, t, mask = stream["x"], stream["time"], stream["mask"]
        outputs, times = [], []
        available = mask.any(1)
        for i in range(len(x)):
            v = x[i, mask[i]]
            ts = t[i, mask[i]]
            if len(v) == 0:
                outputs.append(x.new_zeros(self.length, x.shape[-1]))
                times.append(t.new_zeros(self.length))
                continue
            order = ts.argsort(dim=0, stable=True)
            ts, v = ts[order], v[order]
            target = torch.linspace(0, 1, self.length, device=x.device) * (ts[-1]-ts[0]) + ts[0]
            if len(v) == 1:
                y = v.expand(self.length, -1)
            else:
                upper = torch.searchsorted(ts.contiguous(), target).clamp(1, len(ts)-1)
                lower = upper - 1
                a = ((target-ts[lower]) / (ts[upper]-ts[lower]).clamp_min(1e-8)).unsqueeze(-1)
                y = v[lower]*(1-a) + v[upper]*a
            outputs.append(y)
            times.append(target)
        y = torch.stack(outputs)
        if self.learnable:
            y = y + self.refine(y.transpose(1, 2)).transpose(1, 2)
        y = y * available[:, None, None]
        return y, torch.stack(times), available


class SensorEncoder(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, hidden, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class AdaptiveTokens(nn.Module):
    """TokenLearner-style temporal weighted pooling (an adaptation to sensors)."""
    def __init__(self, hidden, tokens):
        super().__init__()
        self.scores = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, tokens))

    def forward(self, h, times):
        weights = self.scores(h).transpose(1, 2).softmax(-1)
        return weights @ h, (weights @ times.unsqueeze(-1)).squeeze(-1)


class TemporalReasoner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.names = list(cfg.modalities)
        d = cfg.hidden
        self.modality = nn.Embedding(len(self.names), d)
        self.time = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.availability = nn.Linear(len(self.names), d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, d*4, cfg.dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.classifier = nn.Linear(d, cfg.classes)
        self.events = nn.Linear(d, cfg.classes)

    def forward(self, tokens, times, masks, availability):
        b = availability.shape[0]
        pieces, valid, temporal = [], [], []
        for mid, n in enumerate(self.names):
            if n not in tokens:
                continue
            t = times[n]
            position = torch.stack([t, torch.sin(t*math.pi), torch.cos(t*math.pi)], -1)
            z = tokens[n] + self.modality.weight[mid] + self.time(position)
            pieces.append(z)
            valid.append(masks[n])
            # Masked slots sort last. Each sample retains its own temporal ordering.
            temporal.append(t.masked_fill(~masks[n], float("inf")))
        if pieces:
            z, mask, ts = torch.cat(pieces, 1), torch.cat(valid, 1), torch.cat(temporal, 1)
            order = ts.argsort(dim=1, stable=True)
            z = z.gather(1, order.unsqueeze(-1).expand_as(z))
            mask, ts = mask.gather(1, order), ts.gather(1, order)
        else:
            z = self.cls.new_zeros(b, 0, self.cls.shape[-1])
            mask = torch.zeros(b, 0, dtype=torch.bool, device=z.device)
            ts = z.new_zeros(b, 0)
        cls = self.cls.expand(b, -1, -1) + self.availability(availability.float()).unsqueeze(1)
        attention = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=z.device), mask], 1)
        encoded = self.norm(self.transformer(torch.cat([cls, z], 1), src_key_padding_mask=~attention))
        return {"logits": self.classifier(encoded[:, 0]), "context": encoded[:, 0],
                "sensor_tokens": encoded[:, 1:], "sensor_mask": mask,
                "event_logits": self.events(encoded[:, 1:]), "event_times": ts,
                "availability": availability}


class DistributedModel(nn.Module):
    """Edge selectors contain no token-bank weights; central lookup is explicit."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.validate()
        self.names = list(cfg.modalities)
        d = cfg.hidden
        self.resamplers = nn.ModuleDict({n: Resampler(c, cfg.resample_length, cfg.method != "fixed_resampling") for n, c in cfg.modalities.items()})
        self.encoders = nn.ModuleDict({n: SensorEncoder(c, d) for n, c in cfg.modalities.items()})
        self.reconstructors = nn.ModuleDict({n: nn.Linear(d, c) for n, c in cfg.modalities.items()})
        self.poolers = nn.ModuleDict({n: AdaptiveTokens(d, cfg.tokens) for n in self.names})
        self.bank_sizes = {n: cfg.bank_sizes.get(n, cfg.bank_size) for n in self.names}
        self.offsets, offset = {}, 0
        for n in self.names:
            self.offsets[n], offset = offset, offset + self.bank_sizes[n]
        self.vocabulary = offset
        self.bank = nn.Embedding(offset, d)
        nn.init.normal_(self.bank.weight, std=.1)
        self.selectors = nn.ModuleDict({n: nn.Linear(d, offset if cfg.method == "unrestricted_bank" else self.bank_sizes[n]) for n in self.names})
        self.projections = nn.ModuleDict({n: nn.Linear(d, d) for n in self.names})
        self.compressors = nn.ModuleDict({n: nn.Sequential(nn.Linear(d, cfg.bottleneck_dim), nn.Tanh()) for n in self.names})
        self.decompressors = nn.ModuleDict({n: nn.Linear(cfg.bottleneck_dim, d) for n in self.names})
        self.local_heads = nn.ModuleDict({n: nn.Linear(d, cfg.classes) for n in self.names})
        self.external_projections = nn.ModuleDict({n: nn.Linear(c, d) for n, c in cfg.modalities.items()})
        self.reasoner = TemporalReasoner(cfg)
        self.early_projection = nn.Linear(sum(cfg.modalities.values())+len(self.names), d)
        layers, channels = [], sum(cfg.modalities.values())
        for _ in range(4):
            layers.extend([nn.Conv1d(channels, d, 5, padding=2), nn.ReLU()])
            channels = d
        self.conv_lstm = nn.Sequential(*layers)
        self.lstm = nn.LSTM(d, d, num_layers=2, dropout=cfg.dropout, batch_first=True)
        self.lstm_head = nn.Linear(d, cfg.classes)
        if cfg.method == "separate_banks":
            self.separate_banks = nn.ParameterDict({n: nn.Parameter(self.bank.weight[self.offsets[n]:self.offsets[n]+self.bank_sizes[n]].detach().clone()) for n in self.names})
            self.bank.requires_grad_(False)
        self.register_buffer("posthoc_ready", torch.tensor(False))

    def freeze_encoders(self):
        if self.cfg.method != "finetune_encoders":
            self.encoders.requires_grad_(False)
        if self.cfg.method == "posthoc_vq":
            self.resamplers.requires_grad_(False)
            self.bank.requires_grad_(False)
        return self

    def features(self, streams):
        features, times, availability, targets = {}, {}, {}, {}
        for n in self.names:
            x, t, present = self.resamplers[n](streams[n])
            features[n] = self.encoders[n](x)
            times[n], availability[n], targets[n] = t, present, x
        return features, times, availability, targets

    def region(self, name):
        if self.cfg.method == "separate_banks":
            return self.separate_banks[name]
        if self.cfg.method == "unrestricted_bank":
            return self.bank.weight
        start = self.offsets[name]
        return self.bank.weight[start:start+self.bank_sizes[name]]

    def _mask(self, present, count):
        return present[:, None].expand(-1, count)

    def peripheral(self, streams, drop=False):
        cfg, messages, aux = self.cfg, {}, {}
        if cfg.method in RAW or cfg.method == "imagebind":
            for n, s in streams.items():
                valid = s["mask"]
                if drop:
                    valid = valid & (torch.rand(valid.shape[0], device=valid.device) >= cfg.modality_dropout)[:, None]
                messages[n] = Message(n, s["x"], s["time"], valid)
            return messages, aux
        features, times, present, targets = self.features(streams)
        if drop:
            present = {n: p & (torch.rand(len(p), device=p.device) >= cfg.modality_dropout) for n, p in present.items()}
        pooled_for_alignment = {}
        commitment, usage, reconstruction = [], [], []
        aux["assignments"] = {}
        aux["local_logits"] = {}
        for n in self.names:
            h, t, p = features[n], times[n], present[n]
            pooled_for_alignment[n] = (h.mean(1), p)
            if cfg.method in LOCAL:
                logits = self.local_heads[n](h.mean(1))
                aux["local_logits"][n] = (logits, p)
                values = logits.softmax(-1) if cfg.method == "late_fusion" else logits.argmax(-1).to(h.dtype).unsqueeze(-1)
                messages[n] = Message(n, values[:, None, :], t.mean(1, keepdim=True), self._mask(p, 1))
                continue
            if cfg.method in DISCRETE:
                if cfg.method == "posthoc_vq":
                    if not self.posthoc_ready:
                        raise RuntimeError("Fit train-only post-hoc k-means before inference")
                    # Match the dense baseline's full temporal feature grid exactly.
                    z, t = h, t
                else:
                    z, t = self.poolers[n](h, t)
                    z = self.projections[n](z)
                if not self.training and cfg.method not in ("joint_vq", "posthoc_vq"):
                    # Deployment selectors have no access to bank weights.
                    indices = self.selectors[n](z).argmax(-1)
                    offset = 0 if cfg.method == "unrestricted_bank" else self.offsets[n]
                    valid = self._mask(p, z.shape[1])
                    messages[n] = Message(n, indices+offset, t, valid, self.vocabulary)
                    aux["assignments"][n] = (indices, valid)
                    continue
                bank = self.region(n)
                if cfg.method in ("joint_vq", "posthoc_vq"):
                    indices = torch.cdist(z.float(), bank.float()).argmin(-1)
                    onehot = F.one_hot(indices, len(bank)).to(z.dtype)
                    soft = (-torch.cdist(z.float(), bank.float()).square()/cfg.temperature).softmax(-1)
                else:
                    logits = self.selectors[n](z)
                    soft = (logits/cfg.temperature).softmax(-1)
                    indices = logits.argmax(-1)
                    hard = F.one_hot(indices, len(bank)).to(soft.dtype)
                    onehot = hard - soft.detach() + soft if self.training else hard
                selected = onehot @ bank
                if cfg.method == "joint_vq" and self.training:
                    selected = z + (selected-z).detach()
                valid = self._mask(p, z.shape[1])
                # Mask all auxiliary losses: absent streams never influence training.
                if self.training and p.any() and cfg.method != "posthoc_vq":
                    commitment.append(F.mse_loss(z[p], (onehot @ bank).detach()[p]))
                    commitment.append(F.mse_loss((onehot.detach() @ bank)[p], z.detach()[p]))
                    probs = soft[p].mean((0, 1)).clamp_min(1e-8)
                    usage.append((probs * (probs.log()+math.log(len(bank)))).sum())
                    reconstructed = self.reconstructors[n](selected)
                    target = F.adaptive_avg_pool1d(targets[n].transpose(1, 2), z.shape[1]).transpose(1, 2)
                    reconstruction.append(F.mse_loss(reconstructed[p], target[p]))
                offset = 0 if cfg.method == "unrestricted_bank" else self.offsets[n]
                messages[n] = Message(n, indices+offset, t, valid, self.vocabulary)
                # The training surrogate exists centrally and is never serialized.
                if self.training:
                    aux.setdefault("surrogates", {})[n] = selected
                aux["assignments"][n] = (indices, valid)
            elif cfg.method == "tokenlearner":
                z, t = self.poolers[n](h, t)
                messages[n] = Message(n, z, t, self._mask(p, z.shape[1]))
            elif cfg.method == "dense_tokens":
                z, t = self.poolers[n](h, t)
                z = self.projections[n](z)
                messages[n] = Message(n, z, t, self._mask(p, z.shape[1]))
            elif cfg.method == "bottleneck":
                z = self.compressors[n](h)
                if self.training and cfg.channel_noise:
                    z = z + torch.randn_like(z)*cfg.channel_noise
                messages[n] = Message(n, z, t, self._mask(p, z.shape[1]))
            else:
                messages[n] = Message(n, h, t, self._mask(p, h.shape[1]))
        if not self.training:
            return messages, aux
        zero = next(self.parameters()).sum()*0
        align = []
        for i, n in enumerate(self.names):
            for m in self.names[i+1:]:
                a, pa = pooled_for_alignment[n]
                b, pb = pooled_for_alignment[m]
                keep = pa & pb
                if keep.sum() > 1:
                    logits = F.normalize(a[keep], dim=-1) @ F.normalize(b[keep], dim=-1).T / .1
                    y = torch.arange(len(logits), device=logits.device)
                    align.append((F.cross_entropy(logits, y)+F.cross_entropy(logits.T, y))/2)
        for name, values in (("commitment", commitment), ("usage", usage), ("reconstruction", reconstruction), ("alignment", align)):
            aux[name] = torch.stack(values).mean() if values else zero
        return messages, aux

    def central(self, messages, aux=None, batch_size=None):
        cfg = self.cfg
        aux = aux or {}
        if messages:
            b = next(iter(messages.values())).values.shape[0]
        else:
            b = batch_size or 1
        device = next(self.parameters(), getattr(self, "device_anchor", torch.empty(0))).device
        availability = torch.zeros(b, len(self.names), dtype=torch.bool, device=device)
        for n, m in messages.items():
            availability[:, self.names.index(n)] = m.valid.any(1)
        if cfg.method in LOCAL:
            probabilities = torch.zeros(b, cfg.classes, device=device)
            for n, m in messages.items():
                v = m.values[:, 0]
                if cfg.method == "local_only":
                    v = F.one_hot(v[:, 0].long(), cfg.classes).float()
                probabilities += v * m.valid.any(1)[:, None]
            probabilities /= availability.sum(1).clamp_min(1)[:, None]
            probabilities = torch.where(availability.any(1)[:, None], probabilities, torch.full_like(probabilities, 1/cfg.classes))
            return {"logits": probabilities.clamp_min(1e-8).log(), "availability": availability,
                    "sensor_tokens": probabilities[:, None], "sensor_mask": availability.any(1)[:, None]}
        tokens, times, masks = {}, {}, {}
        if cfg.method in RAW:
            aligned, atimes, ap = {}, {}, {}
            for n, c in cfg.modalities.items():
                if n in messages:
                    m = messages[n]
                    x, t, p = self.resamplers[n]({"x": m.values, "time": m.times, "mask": m.valid})
                else:
                    x = torch.zeros(b, cfg.resample_length, c, device=device)
                    t = torch.zeros(b, cfg.resample_length, device=device)
                    p = torch.zeros(b, dtype=torch.bool, device=device)
                aligned[n], atimes[n], ap[n] = x, t, p
            if cfg.method == "deepconvlstm":
                raw = torch.cat([aligned[n] for n in self.names], -1)
                z, _ = self.lstm(self.conv_lstm(raw.transpose(1, 2)).transpose(1, 2))
                return {"logits": self.lstm_head(z[:, -1]), "availability": availability,
                        "sensor_tokens": z, "sensor_mask": availability.any(1)[:, None].expand(-1, z.shape[1])}
            if cfg.method == "early_fusion":
                x = torch.cat([aligned[n] for n in self.names] + [availability[:, None].expand(-1, cfg.resample_length, -1).float()], -1)
                # Treat synchronized fused slots as one central sequence.
                n = self.names[0]
                tokens[n] = self.early_projection(x)
                weights = availability.float().unsqueeze(-1)
                times[n] = (torch.stack([atimes[k] for k in self.names], 1)*weights).sum(1)/weights.sum(1).clamp_min(1)
                masks[n] = availability.any(1)[:, None].expand(-1, cfg.resample_length)
            else:
                for n in self.names:
                    tokens[n], times[n] = self.encoders[n](aligned[n]), atimes[n]
                    masks[n] = self._mask(ap[n], cfg.resample_length)
        else:
            for n, m in messages.items():
                if cfg.method in DISCRETE:
                    if cfg.method != "unrestricted_bank" and m.valid.any():
                        ids = m.values[m.valid]
                        if (ids < self.offsets[n]).any() or (ids >= self.offsets[n]+self.bank_sizes[n]).any():
                            raise ValueError("Token outside modality-reserved bank")
                    if n in aux.get("surrogates", {}):
                        z = aux["surrogates"][n]
                    elif cfg.method == "separate_banks":
                        # Invalid padding indices are clamped before masked lookup.
                        z = F.embedding((m.values-self.offsets[n]).clamp(0, self.bank_sizes[n]-1), self.separate_banks[n])
                    else:
                        z = self.bank(m.values)
                elif cfg.method == "bottleneck":
                    z = self.decompressors[n](m.values)
                elif cfg.method == "imagebind":
                    z = self.external_projections[n](m.values)
                else:
                    z = m.values
                tokens[n], times[n], masks[n] = z, m.times, m.valid
        return self.reasoner(tokens, times, masks, availability)

    def forward(self, batch):
        messages, aux = self.peripheral(batch["streams"], drop=self.training)
        output = self.central(messages, aux)
        output.update(messages=messages, auxiliary=aux)
        return output

    def objective(self, output, labels, batch=None):
        cfg, aux = self.cfg, output["auxiliary"]
        if cfg.method == "local_only":
            loss = next(self.parameters()).sum()*0
        else:
            loss = F.cross_entropy(output["logits"], labels)
        local = [F.cross_entropy(logits[p], labels[p]) for logits, p in aux.get("local_logits", {}).values() if p.any()]
        if local:
            loss = loss + torch.stack(local).mean()
        for name, weight in (("commitment", cfg.commitment), ("alignment", cfg.alignment),
                             ("usage", cfg.usage), ("reconstruction", cfg.reconstruction)):
            if cfg.method == "no_alignment" and name == "alignment" or cfg.method == "no_usage" and name == "usage":
                continue
            loss = loss + weight*aux.get(name, loss.new_zeros(()))
        if batch is not None and "event_logits" in output and cfg.event_weight:
            event_losses = []
            for i, event in enumerate(batch.get("events", [])):
                if event is None or not output["sensor_mask"][i].any():
                    continue
                t = event["times"].to(labels.device)
                y = event["labels"].to(labels.device)
                keep = output["sensor_mask"][i]
                token_times = output["event_times"][i, keep]
                nearest = (token_times[:, None]-t[None]).abs().argmin(1)
                target = y[nearest]
                known = target != -100
                if known.any():
                    event_losses.append(F.cross_entropy(output["event_logits"][i, keep][known], target[known]))
            if event_losses:
                loss = loss + cfg.event_weight*torch.stack(event_losses).mean()
        return loss
