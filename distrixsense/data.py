"""Participant-isolated manifests, training-only normalization and masked collation.

Each JSONL row identifies one synchronized observation window. NPZ modality
arrays have shape (time, channels); optional `<name>__time` arrays are seconds
relative to the SAME window origin. Missing modalities are omitted from NPZ.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, manifest, split, modalities, stats=None):
        self.path = Path(manifest)
        all_rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        if not all_rows:
            raise ValueError("Manifest is empty")
        owners, files = {}, {}
        for row in all_rows:
            if row["split"] not in ("train", "val", "test"):
                raise ValueError("Every row requires train/val/test split")
            participant = str(row["participant"])
            if participant in owners and owners[participant] != row["split"]:
                raise ValueError(f"Participant leakage: {participant}")
            owners[participant] = row["split"]
            file = str((self.path.parent / row["path"]).resolve())
            if file in files:
                raise ValueError(f"Duplicate window path: {file}")
            files[file] = row["split"]
        self.rows = [r for r in all_rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} observations")
        self.modalities, self.stats = modalities, stats

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        streams = {}
        with np.load(self.path.parent / row["path"], allow_pickle=False) as arrays:
            for name, channels in self.modalities.items():
                if name not in arrays:
                    continue
                x = np.asarray(arrays[name], dtype=np.float32).copy()
                if x.ndim != 2 or x.shape[1] != channels or not len(x):
                    raise ValueError(f"Invalid {name} shape at {row['path']}")
                t = np.asarray(arrays[f"{name}__time"], dtype=np.float64) if f"{name}__time" in arrays else np.linspace(0, 1, len(x), dtype=np.float64)
                if t.shape != (len(x),) or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
                    raise ValueError("Timestamps must be finite and strictly increasing")
                valid = np.isfinite(x).any(axis=1)
                if not valid.any():
                    continue
                # Interpolate only inside this window. Never cross partition boundaries.
                for c in range(channels):
                    finite = np.isfinite(x[:, c])
                    if finite.any():
                        x[:, c] = np.interp(t, t[finite], x[finite, c])
                    else:
                        x[:, c] = self.stats[name]["mean"][c] if self.stats else 0
                if self.stats:
                    x = (x - np.array(self.stats[name]["mean"])) / np.array(self.stats[name]["std"])
                streams[name] = {"x": torch.tensor(x, dtype=torch.float32),
                                 "time": torch.tensor(t, dtype=torch.float32), "valid": torch.tensor(valid)}
        return {"streams": streams, "label": int(row["label"]),
                "participant": str(row["participant"]), "query": row.get("query"),
                "answer": row.get("answer"), "path": row["path"], "events": row.get("events"),
                "annotations": row.get("annotations", []), "dataset": row.get("dataset")}


def fit_stats(dataset):
    """Streaming moments after window-local imputation; training split only."""
    if any(r["split"] != "train" for r in dataset.rows):
        raise ValueError("Normalization may only use training observations")
    sums = {n: np.zeros(c, dtype=np.float64) for n, c in dataset.modalities.items()}
    squares = {n: np.zeros(c, dtype=np.float64) for n, c in dataset.modalities.items()}
    counts = dict.fromkeys(dataset.modalities, 0)
    for sample in dataset:
        for n, s in sample["streams"].items():
            x = s["x"][s["valid"]].numpy().astype(np.float64)
            sums[n] += x.sum(0)
            squares[n] += (x*x).sum(0)
            counts[n] += len(x)
    result = {}
    for n in dataset.modalities:
        if counts[n] == 0:
            raise ValueError(f"No training data for modality {n}")
        mean = sums[n] / counts[n]
        std = np.sqrt(np.maximum(squares[n] / counts[n] - mean*mean, 1e-6))
        result[n] = {"mean": mean.tolist(), "std": std.tolist()}
    return result


def collate(samples, modalities):
    b = len(samples)
    streams = {}
    for n, c in modalities.items():
        length = max([len(s["streams"][n]["x"]) for s in samples if n in s["streams"]] or [1])
        x, t = torch.zeros(b, length, c), torch.zeros(b, length)
        mask = torch.zeros(b, length, dtype=torch.bool)
        for i, sample in enumerate(samples):
            if n in sample["streams"]:
                s = sample["streams"][n]
                l = len(s["x"])
                x[i, :l], t[i, :l], mask[i, :l] = s["x"], s["time"], s["valid"]
        streams[n] = {"x": x, "time": t, "mask": mask}
    return {"streams": streams, "labels": torch.tensor([s["label"] for s in samples]),
            "participants": [s["participant"] for s in samples],
            "queries": [s["query"] for s in samples], "answers": [s["answer"] for s in samples],
            "annotations": [s.get("annotations", []) for s in samples],
            "events": [{"times": torch.tensor(s["events"]["times"], dtype=torch.float32),
                        "labels": torch.tensor(s["events"]["labels"], dtype=torch.long)} if s.get("events") else None for s in samples]}


def to_device(batch, device):
    return {**batch, "streams": {n: {k: v.to(device) for k, v in s.items()}
                               for n, s in batch["streams"].items()}, "labels": batch["labels"].to(device)}


def make_synthetic(output, samples=18, seed=0):
    """Small deterministic fixture; never a scientific benchmark."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rng, rows = np.random.default_rng(seed), []
    for split, people in (("train", range(3)), ("val", range(3, 5)), ("test", range(5, 8))):
        for person in people:
            for j in range(samples):
                label = j % 3
                arrays = {}
                for n, c in (("imu", 6), ("ambient", 2)):
                    if n == "ambient" and j % 7 == 0:
                        continue
                    t = np.cumsum(rng.uniform(.02, .05, rng.integers(25, 45))).astype(np.float32)
                    t -= t[0]
                    arrays[n] = (np.sin(t[:, None] * (label+1)*5) + label*.7 + rng.normal(0, .1, (len(t), c))).astype(np.float32)
                    arrays[n][3, 0] = np.nan
                    arrays[f"{n}__time"] = t
                path = f"p{person}_{j}.npz"
                np.savez_compressed(root / path, **arrays)
                rows.append({"path": path, "participant": f"p{person}", "split": split, "label": label})
    path = root / "manifest.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    return path
