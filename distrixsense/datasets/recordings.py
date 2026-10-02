"""Shape-preserving, windowed multimodal datasets with explicit local sources."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from .catalog import DATASETS, stream_kind
from .readers import local_path, read_annotations, read_stream


class RecordingDataset(Dataset):
    def __init__(self, manifest, dataset, split=None, window_seconds=3., stride_seconds=1.5,
                 modalities=None, transforms=None):
        if dataset not in DATASETS or window_seconds <= 0 or stride_seconds <= 0:
            raise ValueError("Choose a supported dataset and positive window/stride")
        self.path, self.dataset = Path(manifest), dataset
        self.transforms = transforms or {}
        self.records = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        if not self.records:
            raise ValueError("Recording manifest is empty")
        self.windows, self.stream_specs = [], {}
        self.partitions = {}
        identifiers, ownership = {}, {}
        selected = set(modalities) if modalities is not None else None
        for index, record in enumerate(self.records):
            if record.get("dataset", dataset) != dataset:
                raise ValueError("Mixed datasets require separate manifests")
            partition, participant = record["split"], str(record["participant"])
            if partition not in ("train", "val", "test"):
                raise ValueError("Each recording needs a train/val/test split")
            # Multi-person/observer recordings can declare every participant to avoid leakage.
            people = set(map(str, record.get("participants", [participant]))) | {participant}
            for person in people:
                if person in self.partitions and self.partitions[person] != partition:
                    raise ValueError(f"Participant leakage: {person}")
                self.partitions[person] = partition
            recording_id = str(record.get("recording", index))
            if recording_id in identifiers:
                raise ValueError(f"Duplicate recording ID: {recording_id}")
            identifiers[recording_id] = partition
            start, end = float(record["start"]), float(record["end"])
            if not np.isfinite([start, end]).all() or end <= start:
                raise ValueError("Recording start/end must be finite seconds on a common recording clock")
            annotations = list(record.get("annotations", []))
            for spec in record.get("annotation_files", []):
                annotations.extend(read_annotations(self.path.parent, spec))
            for name, spec in record["streams"].items():
                if not name or "." in name:
                    raise ValueError("Stream names must be nonempty and cannot contain dots")
                kind = stream_kind(dataset, name, spec)
                if spec.get("role", "target" if kind == "text" else "input") == "target":
                    annotations.extend(read_annotations(self.path.parent, spec))
                    continue
                if kind == "text" and spec.get("role") != "input":
                    raise ValueError("Text streams require an explicit input role")
                signature = (kind, spec.get("modality", name), spec.get("device"), spec.get("coordinate_frame"), spec.get("units"))
                if name in self.stream_specs and self.stream_specs[name]["signature"] != signature:
                    raise ValueError(f"Stream {name} changes modality, location, units or coordinate frame")
                self.stream_specs[name] = {"kind": kind, "signature": signature, "spec": spec}
                for file in [spec.get("path", spec.get("values")), *spec.get("files", [])]:
                    if file is None:
                        continue
                    source = str(local_path(self.path.parent, file).resolve())
                    if source in ownership and ownership[source] != partition:
                        raise ValueError(f"Source recording leakage: {source}")
                    ownership[source] = partition
            for a in annotations:
                if not np.isfinite([float(a["start"]), float(a["end"])]).all() or float(a["end"]) <= float(a["start"]):
                    raise ValueError("Annotations require positive finite time intervals")
            record["annotations"] = annotations
            if split is not None and partition != split:
                continue
            count = max(0, int(math.floor((end-start-window_seconds)/stride_seconds+1e-8))+1)
            self.windows.extend((index, start+i*stride_seconds, start+i*stride_seconds+window_seconds) for i in range(count))
        if selected is not None and selected-set(self.stream_specs):
            raise ValueError(f"Unknown requested streams: {selected-set(self.stream_specs)}")
        self.modalities = sorted(selected if selected is not None else self.stream_specs)
        if not self.windows:
            raise ValueError("No observation windows; check split, spans and window length")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        recording_index, start, end = self.windows[index]
        record = self.records[recording_index]
        streams = {}
        for name in self.modalities:
            if name not in record["streams"]:
                continue
            spec, kind = record["streams"][name], self.stream_specs[name]["kind"]
            if spec.get("role", "target" if kind == "text" else "input") != "input":
                continue
            if spec.get("available", True) is False:
                continue
            stream = read_stream(self.path.parent, spec, kind, start, end)
            if not len(stream["timestamps"]):
                continue
            if name in self.transforms:
                stream = self.transforms[name](stream)
                kind = stream["kind"]
            stream["time"] = torch.as_tensor(stream.pop("timestamps")-start, dtype=torch.float64)
            values = stream.pop("values")
            if kind == "text":
                stream["text"] = list(values)
                stream["valid"] = torch.ones(len(values), dtype=torch.bool)
            elif kind == "points":
                frames = [torch.tensor(np.asarray(frame), dtype=torch.float32) for frame in values]
                if any(frame.ndim != 2 for frame in frames):
                    raise ValueError("Point-cloud frames must be (points, channels)")
                stream["x"] = frames
                stream["valid"] = torch.tensor([bool(torch.isfinite(frame).all(-1).any()) for frame in frames])
            else:
                x = torch.tensor(np.asarray(values), dtype=torch.float32)
                stream["x"] = x
                stream["valid"] = torch.isfinite(x).flatten(1).any(1)
            streams[name] = stream
        annotations = [{**a, "start": max(float(a["start"]), start)-start,
                        "end": min(float(a["end"]), end)-start,
                        "recording_start": float(a["start"]), "recording_end": float(a["end"])}
                       for a in record["annotations"] if float(a["start"]) < end and float(a["end"]) > start]
        return {"dataset": self.dataset, "recording": str(record.get("recording", recording_index)),
                "participant": str(record["participant"]), "split": record["split"],
                "start": start, "end": end, "streams": streams, "annotations": annotations,
                "metadata": record.get("metadata", {})}


class OpportunityPlusPlusDataset(RecordingDataset):
    def __init__(self, manifest, **kwargs):
        super().__init__(manifest, "opportunity++", **kwargs)


class OpenMarcieDataset(RecordingDataset):
    def __init__(self, manifest, **kwargs):
        super().__init__(manifest, "openmarcie", **kwargs)


class NymeriaDataset(RecordingDataset):
    def __init__(self, manifest, **kwargs):
        super().__init__(manifest, "nymeria", **kwargs)


def multimodal_collate(samples, modalities=None):
    """Pad time without flattening images/poses; pad point counts with masks.

    finite_mask distinguishes unavailable scalar measurements from real zeros;
    mask distinguishes missing time steps and sensors. Text remains a list.
    """
    names = list(modalities) if modalities is not None else sorted({n for sample in samples for n in sample["streams"]})
    streams, availability = {}, torch.zeros(len(samples), len(names), dtype=torch.bool)
    for mid, name in enumerate(names):
        present = [(i, sample["streams"][name]) for i, sample in enumerate(samples) if name in sample["streams"]]
        if not present:
            continue
        kinds = {s["kind"] for _, s in present}
        if len(kinds) != 1:
            raise ValueError(f"Cannot batch mixed kinds for {name}")
        kind, length = next(iter(kinds)), max(len(s["time"]) for _, s in present)
        time = torch.zeros(len(samples), length, dtype=torch.float64)
        mask = torch.zeros(len(samples), length, dtype=torch.bool)
        output = {"kind": kind, "time": time, "mask": mask, "metadata": [None]*len(samples)}
        if kind == "text":
            output["text"] = [[] for _ in samples]
        elif kind == "points":
            frames = [frame for _, s in present for frame in s["x"]]
            channels = {frame.shape[-1] for frame in frames}
            if len(channels) != 1:
                raise ValueError("Point-cloud feature dimensions change within a batch")
            points = max((len(frame) for frame in frames), default=0)
            output["x"] = torch.zeros(len(samples), length, points, next(iter(channels)))
            output["point_mask"] = torch.zeros(len(samples), length, points, dtype=torch.bool)
            output["finite_mask"] = torch.zeros_like(output["x"], dtype=torch.bool)
        else:
            shapes = {tuple(s["x"].shape[1:]) for _, s in present}
            if len(shapes) != 1:
                raise ValueError(f"{name} has changing frame shape; supply an explicit resize/pose transform")
            output["x"] = torch.zeros(len(samples), length, *next(iter(shapes)))
            output["finite_mask"] = torch.zeros_like(output["x"], dtype=torch.bool)
        for i, stream in present:
            count = len(stream["time"])
            time[i, :count], mask[i, :count] = stream["time"], stream["valid"]
            availability[i, mid] = stream["valid"].any()
            output["metadata"][i] = stream["metadata"]
            if kind == "text":
                output["text"][i] = stream["text"]
            elif kind == "points":
                for j, frame in enumerate(stream["x"]):
                    n = len(frame)
                    finite = torch.isfinite(frame)
                    output["x"][i, j, :n] = torch.nan_to_num(frame, nan=0., posinf=0., neginf=0.)
                    output["finite_mask"][i, j, :n] = finite
                    output["point_mask"][i, j, :n] = finite.all(-1)
            else:
                output["x"][i, :count] = torch.nan_to_num(stream["x"], nan=0., posinf=0., neginf=0.)
                output["finite_mask"][i, :count] = torch.isfinite(stream["x"])
        streams[name] = output
    return {"streams": streams, "modalities": names, "availability": availability,
            "annotations": [s["annotations"] for s in samples],
            "participants": [s["participant"] for s in samples],
            "recordings": [s["recording"] for s in samples],
            "window_start": torch.tensor([s["start"] for s in samples], dtype=torch.float64),
            "window_end": torch.tensor([s["end"] for s in samples], dtype=torch.float64)}
