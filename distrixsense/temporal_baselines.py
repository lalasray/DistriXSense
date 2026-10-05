"""Supervised sensor adaptations of TCN, time-series Transformer and PatchTST.

These consume the same normalized feature sequences as the main suite. No
published pretrained weights, forecasting heads or reproduction scores are used.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

METHODS = {"tcn", "ts_transformer", "patchtst"}


class CausalBlock(nn.Module):
    def __init__(self, width, kernel, dilation, dropout):
        super().__init__()
        self.padding = (kernel-1)*dilation
        self.convs = nn.ModuleList([weight_norm(nn.Conv1d(width, width, kernel,
                                                       dilation=dilation)) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = x
        for conv in self.convs:
            z = self.dropout(F.relu(conv(F.pad(z, (self.padding, 0)))))
        return F.relu(x+z)


class TemporalBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.method = cfg.method
        self.patch_length, self.patch_stride = cfg.patch_length, cfg.patch_stride
        channels = sum(cfg.modalities.values())
        self.channel_counts = list(cfg.modalities.values())
        self.availability = nn.Linear(len(cfg.modalities), cfg.hidden)
        if self.method == "tcn":
            self.projection = nn.Linear(channels+len(cfg.modalities), cfg.hidden)
            self.network = nn.Sequential(*[CausalBlock(cfg.hidden, cfg.tcn_kernel, 2**i, cfg.dropout)
                                           for i in range(cfg.layers)])
        else:
            self.projection = nn.Linear(cfg.patch_length if self.method == "patchtst"
                                        else channels+len(cfg.modalities), cfg.hidden)
            layer = nn.TransformerEncoderLayer(cfg.hidden, cfg.heads, cfg.hidden*4,
                                               cfg.dropout, batch_first=True, norm_first=True)
            self.network = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.hidden)
        self.classifier = nn.Linear(cfg.hidden, cfg.classes)
        self.events = nn.Linear(cfg.hidden, cfg.classes)

    @staticmethod
    def position(count, width, reference):
        pos = torch.arange(count, device=reference.device, dtype=reference.dtype)[:, None]
        frequencies = torch.exp(torch.arange(0, width, 2, device=reference.device,
                                             dtype=reference.dtype)*(-math.log(10000.)/width))
        result = reference.new_zeros(count, width)
        result[:, 0::2] = torch.sin(pos*frequencies)
        result[:, 1::2] = torch.cos(pos*frequencies[:width//2])
        return result

    def forward(self, x, times, availability):
        b, length, channels = x.shape
        present = availability.any(1)
        flags = availability[:, None].expand(-1, length, -1).to(x.dtype)
        if self.method == "patchtst":
            # One shared patch projection and Transformer for every univariate channel.
            remainder = (length-self.patch_length) % self.patch_stride
            padding = (self.patch_stride-remainder) % self.patch_stride
            series = F.pad(x.transpose(1, 2), (0, padding), mode="replicate")
            patches = series.unfold(-1, self.patch_length, self.patch_stride)
            count = patches.shape[2]
            z = self.projection(patches.reshape(b*channels, count, self.patch_length))
            z = self.network(z+self.position(count, z.shape[-1], z))
            z = self.norm(z).reshape(b, channels, count, -1)
            channel_mask = torch.cat([availability[:, i:i+1].expand(-1, c)
                                      for i, c in enumerate(self.channel_counts)], 1)
            weights = channel_mask[:, :, None, None].to(z.dtype)
            z = (z*weights).sum(1)/weights.sum(1).clamp_min(1)
            event_times = F.pad(times[:, None], (0, padding), mode="replicate").unfold(
                -1, self.patch_length, self.patch_stride).mean(-1).squeeze(1)
        else:
            z = self.projection(torch.cat([x, flags], -1))
            if self.method == "tcn":
                z = self.network(z.transpose(1, 2)).transpose(1, 2)
            else:
                z = self.network(z+self.position(length, z.shape[-1], z))
            z = self.norm(z)
            event_times = times
        mask = present[:, None].expand(-1, z.shape[1])
        z = z*mask[:, :, None]
        context = z.mean(1)+self.availability(availability.to(x.dtype))
        return {"logits": self.classifier(context), "context": context,
                "sensor_tokens": z, "sensor_mask": mask, "availability": availability,
                "event_logits": self.events(z), "event_times": event_times}
