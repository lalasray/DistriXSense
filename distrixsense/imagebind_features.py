"""Optional extraction using the official ImageBind model and local weights."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .config import Config

SUPPORTED = {"vision", "audio", "depth", "thermal", "imu"}


@torch.no_grad()
def extract(manifest, weights, output, device="cpu"):
    try:
        from imagebind.models import imagebind_model
    except ImportError as exc:
        raise RuntimeError("Install the official facebookresearch/ImageBind package in a compatible environment") from exc
    path, root = Path(manifest), Path(output)
    if (root/"manifest.jsonl").exists():
        raise ValueError("Output already exists")
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    owners = {}
    for row in rows:
        person, split = str(row["participant"]), row["split"]
        if split not in ("train", "val", "test") or person in owners and owners[person] != split:
            raise ValueError("Invalid split or participant leakage")
        owners[person] = split
    model = imagebind_model.imagebind_huge(pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    root.mkdir(parents=True, exist_ok=True)
    exported, widths, timings = [], {}, []
    for i, row in enumerate(rows):
        with np.load(path.parent/row["path"], allow_pickle=False) as arrays:
            unsupported = set(arrays.files)-SUPPORTED-{f"{n}__time" for n in SUPPORTED}
            if unsupported:
                raise ValueError(f"Unsupported or annotation input keys: {unsupported}")
            inputs, times = {}, {}
            for n in SUPPORTED:
                if n not in arrays:
                    continue
                x = torch.tensor(arrays[n], dtype=torch.float32, device=device)
                if x.shape[0] != 1 or not torch.isfinite(x).all():
                    raise ValueError("Official preprocessed inputs must be finite with batch size one")
                inputs[n] = x
                times[n] = float(np.asarray(arrays[f"{n}__time"]).reshape(-1).mean()) if f"{n}__time" in arrays else .5
            if not inputs:
                raise ValueError("No supported sensor input")
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        embeddings = model(inputs)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter()-start)*1000)
        features = {}
        for n, embedding in embeddings.items():
            x = embedding.cpu().numpy().reshape(1, -1)
            features[n], features[f"{n}__time"] = x, np.asarray([times[n]], dtype=np.float32)
            widths[n] = x.shape[1]
        name = f"window-{i:08d}.npz"
        np.savez_compressed(root/name, **features)
        exported.append({**row, "path": name})
    labels = sorted(set(int(r["label"]) for r in rows if r["split"] == "train"))
    if labels != list(range(len(labels))) or any(int(r["label"]) not in labels for r in rows):
        raise ValueError("Input labels must be a contiguous training-derived vocabulary")
    (root/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in exported))
    (root/"config.json").write_text(json.dumps(Config(modalities=widths, classes=len(labels), method="imagebind").to_dict(), indent=2))
    h = hashlib.sha256()
    with Path(weights).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    (root/"feature_provenance.json").write_text(json.dumps({"encoder": "official imagebind_huge", "weights": str(Path(weights).resolve()),
        "sha256": h.hexdigest(), "supported_modalities": sorted(widths), "device": device,
        "extraction_mean_ms": float(np.mean(timings)), "extraction_p95_ms": float(np.quantile(timings, .95)),
        "note": "Preprocessed input tensors; input transforms and first-call warmup are not removed from this timing description. Record preprocessing separately."}, indent=2))
    return root/"manifest.jsonl"
