"""Explicit preparation adapters; no annotation or timestamp column is a feature."""
from collections import Counter
import json
from pathlib import Path
import re
import urllib.request
import zipfile
import numpy as np
from .config import Config

UCI_URL = "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip"
LABEL_COLUMNS = {"locomotion": 243, "activity": 244, "gestures": 249}


def download_opportunity(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "opportunity.zip"
    if not archive.exists():
        urllib.request.urlretrieve(UCI_URL, archive)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            destination = (root / item.filename).resolve()
            if not destination.is_relative_to(root.resolve()):
                raise ValueError("Archive path escapes output directory")
        z.extractall(root)
    return root


def metadata_groups(path):
    """Group channels sharing a sensor descriptor, removing only terminal axes.

    Indices are zero-based RAW-file indices. Only the 242 sensor columns are
    considered; all seven annotation columns and MILLISEC are excluded.
    """
    groups = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r"\s*Column:\s*(\d+)\s+(.+?)\s*$", line)
        if not match:
            continue
        index, descriptor = int(match[1])-1, match[2]
        if not 1 <= index < 243:
            continue
        descriptor = re.sub(r"(?:[XYZxyz]|[1-4])$", "", descriptor)
        name = re.sub(r"[^a-zA-Z0-9_-]+", "_", descriptor).strip("_")
        groups.setdefault(name, []).append(index)
    if sorted(i for cols in groups.values() for i in cols) != list(range(1, 243)):
        raise ValueError("column_names.txt did not describe all 242 sensor columns; use --group-map")
    return groups


def prepare_opportunity(root, output, task="gestures", window=96, stride=48,
                        train_people=("S1", "S2"), val_people=("S3",), test_people=("S4",),
                        group_map=None, exclude_null=False, max_windows=None):
    if window < 2 or stride < 1:
        raise ValueError("Invalid window or stride")
    sets = [set(train_people), set(val_people), set(test_people)]
    if any(not s for s in sets) or any(sets[i]&sets[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Train, validation and test participants must be nonempty and disjoint")
    source, destination = Path(root), Path(output)
    if (destination / "manifest.jsonl").exists():
        raise ValueError("Prepared data already exists; choose a new output directory")
    files = sorted(source.rglob("*.dat"))
    if not files:
        raise ValueError(f"No OPPORTUNITY .dat files under {source}")
    metadata = sorted(source.rglob("column_names.txt"))
    if group_map:
        groups = json.loads(Path(group_map).read_text())
    elif metadata:
        groups = metadata_groups(metadata[0])
    else:
        # Dataset's published body/object/ambient partitions, not guessed axes.
        groups = {"body_worn": list(range(1, 146)), "objects": list(range(146, 206)), "ambient": list(range(206, 243))}
    if not groups or any(not cols or len(set(cols)) != len(cols) or any(not isinstance(i, int) or not 1 <= i < 243 for i in cols) for cols in groups.values()):
        raise ValueError("Group map must contain sensor indices 1..242 in RAW 0-based coordinates")
    if len(set(i for cols in groups.values() for i in cols)) != sum(map(len, groups.values())):
        raise ValueError("A sensor channel cannot occur in multiple peripheral groups")
    destination.mkdir(parents=True, exist_ok=True)
    rows, counts, observed_train_labels = [], Counter(), set()
    for file in files:
        match = re.match(r"(S\d+)-", file.name)
        if not match:
            continue
        person = match[1]
        split = next((s for s, people in zip(("train", "val", "test"), sets) if person in people), None)
        if split is None:
            continue
        data = np.loadtxt(file, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 250:
            raise ValueError(f"Expected 250 OPPORTUNITY columns in {file}")
        for start in range(0, len(data)-window+1, stride):
            if max_windows is not None and counts[(split, person)] >= max_windows:
                break
            x = data[start:start+window]
            labels = x[:, LABEL_COLUMNS[task]]
            if not np.isfinite(labels).all():
                raise ValueError(f"Missing label in {file}")
            raw_label = int(Counter(labels.astype(int)).most_common(1)[0][0])
            if exclude_null and raw_label == 0:
                continue
            t = (x[:, 0]-x[0, 0])/1000
            if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
                continue  # retain only windows with an unambiguous time axis
            if split == "train":
                observed_train_labels.update(int(y) for y in labels if not exclude_null or y != 0)
            arrays = {}
            for n, cols in groups.items():
                arrays[n] = x[:, cols]
                arrays[f"{n}__time"] = t
            name = f"{file.stem}-{start:08d}.npz"
            np.savez_compressed(destination/name, **arrays)
            rows.append({"path": name, "participant": person, "split": split, "raw_label": raw_label,
                         "source": file.name, "start_row": start, "end_row": start+window,
                         "events": {"times": t.tolist(), "labels": labels.astype(int).tolist()}})
            counts[(split, person)] += 1
    train_labels = sorted(observed_train_labels)
    if not train_labels:
        raise ValueError("No training windows retained")
    mapping = {label: i for i, label in enumerate(train_labels)}
    for row in rows:
        if row["raw_label"] not in mapping:
            raise ValueError(f"Evaluation label {row['raw_label']} absent from training; revise the split or window size")
        row["label"] = mapping[row["raw_label"]]
        row["events"]["labels"] = [mapping.get(y, -100) for y in row["events"]["labels"]]
    for split in ("train", "val", "test"):
        if not any(r["split"] == split for r in rows):
            raise ValueError(f"No {split} windows retained")
    (destination/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (destination/"labels.json").write_text(json.dumps({str(k): v for k, v in mapping.items()}, indent=2))
    (destination/"groups.json").write_text(json.dumps(groups, indent=2))
    config = Config(modalities={n: len(cols) for n, cols in groups.items()}, classes=len(mapping))
    (destination/"config.json").write_text(json.dumps(config.to_dict(), indent=2))
    (destination/"preparation.json").write_text(json.dumps({"dataset": "original OPPORTUNITY", "task": task,
        "window": window, "stride": stride, "null_excluded": exclude_null,
        "participants": {s: sorted(p) for s, p in zip(("train", "val", "test"), sets)},
        "max_windows_per_participant": max_windows, "grouping": "explicit" if group_map else "metadata" if metadata else "coarse sensor families"}, indent=2))
    return destination/"manifest.jsonl"


def prepare_records(records_path, output, dataset_name, window_seconds=3., stride_seconds=1.5):
    """Timestamp-based adapter for OPPORTUNITY++ / OpenMarcie numeric streams.

    JSONL recording rows: participant, split, streams {name: {values: npy path,
    timestamps: npy path}}, annotations [{start, end, label, query?, answer?}].
    Times and annotations use the same seconds origin. Videos/audio may be
    converted into frozen-encoder feature arrays before using this adapter.
    No fabricated labels, captions, queries or sensor modality mappings.
    """
    if dataset_name not in ("opportunity++", "openmarcie"):
        raise ValueError("Use opportunity++ or openmarcie")
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("Window and stride must be positive")
    records_path, destination = Path(records_path), Path(output)
    if (destination/"manifest.jsonl").exists():
        raise ValueError("Prepared data already exists")
    records = [json.loads(s) for s in records_path.read_text().splitlines() if s.strip()]
    owners, widths, rows = {}, {}, []
    for recording_index, record in enumerate(records):
        participant, split = str(record["participant"]), record["split"]
        if split not in ("train", "val", "test") or participant in owners and owners[participant] != split:
            raise ValueError("Invalid split or participant leakage")
        owners[participant] = split
        streams = {}
        for n, spec in record["streams"].items():
            values = np.load(records_path.parent/spec["values"], allow_pickle=False)
            times = np.load(records_path.parent/spec["timestamps"], allow_pickle=False)
            if values.ndim != 2 or times.shape != (len(values),) or not len(times) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                raise ValueError("Each stream requires (T,C) values and ordered timestamps")
            if n in widths and widths[n] != values.shape[1]:
                raise ValueError("Channel dimensions change between recordings")
            widths[n] = values.shape[1]
            streams[n] = (values, times)
        if not streams:
            continue
        start = min(t[0] for _, t in streams.values())
        end = max(t[-1] for _, t in streams.values())
        for j, origin in enumerate(np.arange(start, end-window_seconds+1e-8, stride_seconds)):
            annotations = [a for a in record["annotations"] if a["start"] <= origin and a["end"] >= origin+window_seconds]
            if len(annotations) != 1:
                continue  # do not invent a target for ambiguous or concurrent annotations
            annotation = annotations[0]
            arrays = {}
            for n, (x, t) in streams.items():
                keep = (t >= origin)&(t < origin+window_seconds)
                if keep.any():
                    arrays[n], arrays[f"{n}__time"] = x[keep].astype(np.float32), (t[keep]-origin).astype(np.float32)
            if not arrays:
                continue
            destination.mkdir(parents=True, exist_ok=True)
            name = f"recording-{recording_index}-{j}.npz"
            np.savez_compressed(destination/name, **arrays)
            rows.append({"path": name, "participant": participant, "split": split, "raw_label": str(annotation["label"]),
                         "query": annotation.get("query"), "answer": annotation.get("answer"), "origin_seconds": float(origin),
                         "events": {"times": [0., window_seconds], "labels": [str(annotation["label"])]*2}})
    labels = sorted(set(r["raw_label"] for r in rows if r["split"] == "train"))
    mapping = {label: i for i, label in enumerate(labels)}
    if not labels or any(not any(r["split"] == s for r in rows) for s in ("train", "val", "test")):
        raise ValueError("Adapter requires train, validation and test windows")
    for row in rows:
        if row["raw_label"] not in mapping:
            raise ValueError("Evaluation label absent from training")
        row["label"] = mapping[row["raw_label"]]
        row["events"]["labels"] = [mapping[y] for y in row["events"]["labels"]]
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (destination/"labels.json").write_text(json.dumps(mapping, indent=2))
    (destination/"config.json").write_text(json.dumps(Config(modalities=widths, classes=len(mapping)).to_dict(), indent=2))
    (destination/"preparation.json").write_text(json.dumps({"dataset": dataset_name, "window_seconds": window_seconds,
        "stride_seconds": stride_seconds, "target_rule": "single annotation covering complete window"}, indent=2))
    return destination/"manifest.jsonl"
