"""Export explicitly selected features from the three local multimodal datasets."""
import json
from pathlib import Path
import numpy as np
from .config import Config
from .datasets import RecordingDataset
from .datasets.features import feature_sequence


def prepare_records(records_path, output, dataset_name, window_seconds=3., stride_seconds=1.5,
                    feature_policies=None, target_track="activity", target_policy="strict"):
    if target_policy not in ("strict", "majority"):
        raise ValueError("target_policy must be strict or majority")
    data = RecordingDataset(records_path, dataset_name, window_seconds=window_seconds, stride_seconds=stride_seconds)
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output directory must be empty")
    policies, rows, widths, skipped = feature_policies or {}, [], {}, 0
    for index, sample in enumerate(data):
        candidates = [a for a in sample["annotations"] if a.get("track", "activity") == target_track and "label" in a]
        if target_policy == "strict":
            full = [a for a in candidates if a["start"] <= 1e-8 and a["end"] >= window_seconds-1e-8]
            chosen = full[0] if len(candidates) == 1 and len(full) == 1 else None
        else:
            durations = {}
            for a in candidates:
                label = str(a["label"])
                durations[label] = durations.get(label, 0.)+a["end"]-a["start"]
            chosen_label = max(durations, key=lambda k: (durations[k], k)) if durations else None
            chosen = next((a for a in candidates if str(a["label"]) == chosen_label), None)
        if chosen is None:
            skipped += 1
            continue
        arrays = {}
        for name, stream in sample["streams"].items():
            x, t = feature_sequence(stream, policies.get(name, policies.get(stream["kind"])))
            if name in widths and widths[name] != x.shape[1]:
                raise ValueError(f"Feature width changes for {name}")
            widths[name] = x.shape[1]
            arrays[name], arrays[f"{name}__time"] = x.astype(np.float32), t.astype(np.float64)
        if not arrays:
            skipped += 1
            continue
        destination.mkdir(parents=True, exist_ok=True)
        name = f"window-{index:08d}.npz"
        np.savez_compressed(destination/name, **arrays)
        rows.append({"path": name, "dataset": dataset_name, "participant": sample["participant"],
                     "recording": sample["recording"], "split": sample["split"], "raw_label": str(chosen["label"]),
                     "query": chosen.get("query"), "answer": chosen.get("answer"),
                     "origin_seconds": sample["start"], "annotations": sample["annotations"]})
    labels = sorted({r["raw_label"] for r in rows if r["split"] == "train"})
    mapping = {label: i for i, label in enumerate(labels)}
    if not labels or any(not any(r["split"] == s for r in rows) for s in ("train", "val", "test")):
        raise ValueError("Need labelled train/val/test windows; raw loading also supports unlabelled/concurrent windows")
    for row in rows:
        if row["raw_label"] not in mapping:
            raise ValueError(f"Evaluation label absent from training: {row['raw_label']}")
        row["label"] = mapping[row["raw_label"]]
    (destination/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (destination/"labels.json").write_text(json.dumps(mapping, indent=2))
    (destination/"class_names.json").write_text(json.dumps(labels, indent=2))
    (destination/"config.json").write_text(json.dumps(Config(modalities=widths, classes=len(labels)).to_dict(), indent=2))
    (destination/"preparation.json").write_text(json.dumps({"dataset": dataset_name,
        "window_seconds": window_seconds, "stride_seconds": stride_seconds, "target_track": target_track,
        "target_policy": target_policy, "skipped_unlabelled_or_ambiguous_windows": skipped,
        "feature_policies": policies, "source_manifest": str(Path(records_path).resolve()),
        "modalities": data.modalities}, indent=2))
    return destination/"manifest.jsonl"
