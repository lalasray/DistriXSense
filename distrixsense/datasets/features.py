"""Opt-in deterministic feature adapters; these are not pretrained encoders."""
import numpy as np
import torch
from torch.nn import functional as F


def feature_sequence(stream, policy=None):
    kind, t = stream["kind"], stream["time"].numpy()
    if kind in ("signal", "features") and policy is None:
        policy = {"method": "identity"}
    if isinstance(policy, str):
        policy = {"method": policy}
    if not policy:
        raise ValueError(f"{kind} requires an explicit feature policy or a frozen-encoder feature source")
    method = policy["method"]
    if method == "identity" and kind in ("signal", "features"):
        x = stream["x"].numpy()
    elif method == "flatten_pose" and kind == "pose":
        x = stream["x"].flatten(1).numpy()
    elif method == "spatial_pool" and kind == "image":
        grid = int(policy.get("grid", 8))
        if grid < 1:
            raise ValueError("Pooling grid must be positive")
        x = F.adaptive_avg_pool2d(stream["x"], (grid, grid)).flatten(1).numpy()
    elif method == "point_moments" and kind == "points":
        rows = []
        for frame in stream["x"]:
            frame = frame[torch.isfinite(frame).all(-1)]
            if not len(frame):
                rows.append(torch.full((stream["x"][0].shape[-1]*4,), float("nan")))
            else:
                rows.append(torch.cat([frame.mean(0), frame.std(0, unbiased=False), frame.min(0).values, frame.max(0).values]))
        x = torch.stack(rows).numpy()
    elif method == "log_spectrum" and kind == "audio":
        samples = stream["x"]
        n_fft, hop = int(policy.get("n_fft", 512)), int(policy.get("hop", 256))
        if n_fft < 2 or not 1 <= hop <= n_fft:
            raise ValueError("Audio requires n_fft >= 2 and hop in [1,n_fft]")
        if len(t) > 2 and not np.allclose(np.diff(t), np.median(np.diff(t)), rtol=.01, atol=1e-9):
            raise ValueError("Audio spectrum requires a regular sample clock")
        if len(samples) < n_fft:
            samples = F.pad(samples.T, (0, n_fft-len(samples))).T
        spec = torch.stft(samples.T, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), center=False, return_complex=True)
        x = torch.log1p(spec.abs()).permute(2, 0, 1).flatten(1).numpy()
        positions = np.minimum(np.arange(len(x))*hop+(n_fft-1)/2, len(t)-1)
        t = np.interp(positions, np.arange(len(t)), t)
    else:
        raise ValueError(f"Feature method {method} does not support kind {kind}")
    if x.ndim != 2 or len(x) != len(t) or not len(x):
        raise ValueError("Feature adapter must return nonempty (time, channels) values with timestamps")
    return x, t
