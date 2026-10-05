"""Local-only readers. Time is float64 until subtracting the observation origin.

Readers preserve native axes, physical values and missing-value masks. No image
is flattened and no unseen modality is silently discarded. Optional codecs are
imported only when that source format is requested.
"""
import csv
import gzip
import json
from pathlib import Path
import re
import wave
import numpy as np

UNITS = {"s": 1., "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def local_path(root, path):
    if "://" in str(path):
        raise ValueError("Dataset readers accept local paths only; URLs/downloads are disabled")
    return Path(root) / path


def array(root, path, key=None):
    p = local_path(root, path)
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            if key is None:
                raise ValueError(f"An NPZ key is required for {p}")
            return z[key].copy()
    return np.load(p, allow_pickle=False, mmap_mode="r")


def seconds(raw, spec, root=None):
    unit = spec.get("time_unit", "s")
    if unit not in UNITS:
        raise ValueError("time_unit must be s, ms, us or ns")
    # Integer subtraction before conversion protects epoch-scale nanoseconds.
    origin = spec.get("time_origin", 0)
    raw = np.asarray(raw)
    if "clock_vrs" in spec:
        if root is None or "timecode_origin_ns" not in spec:
            raise ValueError("clock_vrs requires timecode_origin_ns and a local source root")
        from .vrs import provider_for
        import os
        provider = provider_for(str(local_path(root, spec["clock_vrs"]).resolve()), os.getpid())
        ns_per_unit = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1000, "ns": 1}[unit]
        raw_ns = (raw*ns_per_unit).astype(np.int64)
        mapped = np.asarray([provider.convert_from_device_time_to_timecode_ns(int(v)) for v in raw_ns], dtype=np.int64)
        return seconds(mapped, {k: v for k, v in {**spec, "time_unit": "ns", "time_origin": spec["timecode_origin_ns"]}.items() if k != "clock_vrs"})
    if np.issubdtype(raw.dtype, np.integer) and isinstance(origin, int):
        raw = raw.astype(np.int64)-origin
    else:
        raw = raw.astype(np.float64)-origin
    scale = float(spec.get("clock_scale", 1.))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("clock_scale must be positive")
    return raw.astype(np.float64)*UNITS[unit]*scale + float(spec.get("offset_seconds", 0.))


def timestamps(root, spec, count):
    if "timestamps" in spec:
        source = spec["timestamps"]
        raw = array(root, source["path"], source.get("key")) if isinstance(source, dict) else array(root, source)
        t = seconds(raw, spec, root)
    elif "timestamp_key" in spec:
        t = seconds(array(root, spec.get("path", spec.get("values")), spec["timestamp_key"]), spec, root)
    elif "sample_rate" in spec:
        rate = float(spec["sample_rate"])
        if rate <= 0 or not np.isfinite(rate):
            raise ValueError("sample_rate must be finite and positive")
        t = seconds(np.arange(count)/rate, {**spec, "time_unit": "s"}, root)
    else:
        raise ValueError("Provide timestamps, timestamp_key or sample_rate; rates are never guessed")
    validate_times(t, count)
    return t


def validate_times(t, count):
    if t.shape != (count,) or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("Timestamps must match frames and be finite, strictly increasing on the recording clock")


def select(t, start, end):
    return slice(int(np.searchsorted(t, start, side="left")), int(np.searchsorted(t, end, side="left")))


def canonical(values, kind, spec):
    x = np.asarray(values)
    if kind == "image":
        layout = spec.get("layout")
        if layout == "THWC":
            x = x.transpose(0, 3, 1, 2)
        elif layout == "THW":
            x = x[:, None]
        elif layout != "TCHW":
            raise ValueError("Image streams require layout THWC, THW or TCHW")
        if x.ndim != 4:
            raise ValueError("Images must have time, channel, height and width axes")
    elif kind in ("signal", "audio", "features"):
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError(f"{kind} expects (time, channels); use pose or image for structured inputs")
    elif kind == "pose":
        if "frame_shape" in spec:
            x = x.reshape((len(x), *spec["frame_shape"]))
        if x.ndim < 2:
            raise ValueError("Pose requires time and coordinate axes")
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("Numeric streams cannot contain object or string arrays")
    if "value_scale" in spec or "value_offset" in spec:
        x = x.astype(np.float32)*float(spec.get("value_scale", 1.)) + float(spec.get("value_offset", 0.))
    return np.array(x, copy=True)


def numeric_table(root, spec):
    path = local_path(root, spec["path"])
    if "columns" not in spec or "time_column" not in spec:
        raise ValueError("Table sources require explicit columns and time_column")
    if spec["time_column"] in spec["columns"]:
        raise ValueError("Timestamp column cannot be a sensor feature")
    if spec.get("header", False):
        handle = gzip.open(path, "rt") if path.suffix == ".gz" else path.open()
        with handle:
            rows = list(csv.DictReader(handle, delimiter=spec.get("delimiter", ",")))
        columns = spec["columns"]
        x = np.asarray([[float(row[c]) if row[c].strip() else np.nan for c in columns] for row in rows])
        raw = np.asarray([row[spec["time_column"]] for row in rows], dtype=np.int64 if spec.get("integer_timestamps") else np.float64)
    else:
        table = np.loadtxt(path, delimiter=spec.get("delimiter"), skiprows=spec.get("skiprows", 0), ndmin=2)
        if "columns" not in spec or "time_column" not in spec:
            raise ValueError("Table sources require explicit columns and time_column")
        if spec["time_column"] in spec["columns"]:
            raise ValueError("Timestamp column cannot be a sensor feature")
        x, raw = table[:, spec["columns"]], table[:, spec["time_column"]]
    return x, seconds(raw, spec, root)


def audio_slice(root, spec, rate, count, start, end):
    if "timestamps" in spec or "timestamp_key" in spec or "clock_vrs" in spec:
        t = timestamps(root, {**spec, "sample_rate": rate}, count)
        keep = select(t, start, end)
        return keep, t[keep]
    origin = seconds(np.asarray([0.]), {**spec, "time_unit": "s"})[0]
    dt = float(spec.get("clock_scale", 1.))/rate
    first = min(count, max(0, int(np.ceil((start-origin)/dt-1e-8))))
    last = min(count, max(first, int(np.ceil((end-origin)/dt-1e-8))))
    return slice(first, last), origin+np.arange(first, last)*dt


def frame_paths(root, spec):
    if "files" in spec:
        paths = [local_path(root, p) for p in spec["files"]]
    else:
        paths = sorted(local_path(root, spec["path"]).glob(spec.get("pattern", "*")))
    return paths


def read_points(path):
    if path.suffix == ".npy":
        return np.array(np.load(path, allow_pickle=False), copy=True)
    if path.suffix in (".txt", ".xyz", ".csv"):
        return np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None, ndmin=2)
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("PLY/PCD requires the optional open3d package; NPY/XYZ/CSV do not") from exc
    cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(cloud.points)
    return np.concatenate([xyz, np.asarray(cloud.colors)], -1) if cloud.has_colors() else xyz


def read_stream(root, spec, kind, start, end):
    """Return native values and recording-relative timestamps in [start, end)."""
    fmt = spec.get("format", "array")
    metadata = {k: spec[k] for k in ("modality", "device", "units", "coordinate_frame", "calibration") if k in spec}
    if kind == "text":
        rows = [r for r in read_annotations(root, spec) if r["start"] < end and r["end"] > start]
        # Concurrent text annotations remain separate; do not merge their content.
        return {"values": [r.get("text", r.get("caption", "")) for r in rows],
                "timestamps": np.asarray([max(start, r["start"]) for r in rows]), "kind": kind, "metadata": metadata}
    if fmt == "array":
        path = spec.get("path", spec.get("values"))
        x = array(root, path, spec.get("key"))
        t = timestamps(root, spec, len(x))
        keep = select(t, start, end)
        x, t = canonical(x[keep], kind, spec), t[keep]
    elif fmt in ("csv", "table", "opportunity_dat"):
        if fmt == "opportunity_dat" and any(not isinstance(c, int) or not 1 <= c < 243 for c in spec["columns"]):
            raise ValueError("OPPORTUNITY++ sensor columns must be RAW zero-based indices 1..242; exclude all annotations")
        x, t = numeric_table(root, spec)
        validate_times(t, len(x))
        keep = select(t, start, end)
        x, t = canonical(x[keep], kind, spec), t[keep]
    elif fmt in ("images", "point_frames", "openpose"):
        paths = frame_paths(root, spec)
        t = timestamps(root, spec, len(paths))
        keep = select(t, start, end)
        frames = []
        for path in paths[keep]:
            if fmt == "point_frames":
                frames.append(read_points(path))
            elif fmt == "openpose":
                people = json.loads(path.read_text()).get("people", [])
                person = spec.get("person_index", 0)
                # Missing detections stay missing, not confident zero coordinates.
                key = spec.get("pose_key", "pose_keypoints_2d")
                shape = tuple(spec.get("frame_shape", [25, 3]))
                frames.append(np.asarray(people[person][key]).reshape(shape) if len(people) > person else np.full(shape, np.nan))
            else:
                try:
                    from PIL import Image
                except ImportError as exc:
                    raise RuntimeError("Image files require Pillow: install the media extra") from exc
                with Image.open(path) as im:
                    frames.append(np.asarray(im).copy())
        t = t[keep]
        if fmt == "point_frames":
            x = frames
        elif frames:
            layout = spec.get("layout", "THWC" if frames[0].ndim == 3 else "THW")
            x = canonical(np.stack(frames), kind, {**spec, "layout": layout})
        else:
            x = []
    elif fmt == "video":
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("Video decoding requires PyAV: install the media extra") from exc
        frames, clock = [], []
        external_times = None
        if "timestamps" in spec:
            source = spec["timestamps"]
            external_times = seconds(array(root, source["path"], source.get("key")) if isinstance(source, dict) else array(root, source), spec)
            validate_times(external_times, len(external_times))
        with av.open(str(local_path(root, spec["path"]))) as container:
            if external_times is None:
                native_start = (start-float(spec.get("offset_seconds", 0.)))/float(spec.get("clock_scale", 1.))+float(spec.get("time_origin", 0.))
                if native_start > 0:
                    container.seek(int(native_start*1e6), backward=True)
            for i, frame in enumerate(container.decode(video=spec.get("video_stream", 0))):
                if external_times is not None:
                    if i >= len(external_times):
                        raise ValueError("Video has more frames than its timestamp sidecar")
                    timestamp = external_times[i]
                else:
                    if frame.pts is None:
                        raise ValueError("Video needs presentation timestamps or an explicit sidecar")
                    timestamp = seconds([float(frame.pts*frame.time_base)], {**spec, "time_unit": "s"})[0]
                if timestamp >= end:
                    break
                if timestamp >= start:
                    frames.append(frame.to_ndarray(format=spec.get("pixel_format", "rgb24")))
                    clock.append(timestamp)
        x = canonical(np.stack(frames), kind, {**spec, "layout": "THWC" if frames[0].ndim == 3 else "THW"}) if frames else []
        t = np.asarray(clock, dtype=np.float64)
    elif fmt == "audio":
        path = local_path(root, spec["path"])
        try:
            import soundfile as sf
            with sf.SoundFile(path) as audio:
                rate, count = audio.samplerate, len(audio)
                # Seek a local slice rather than retaining a whole recording waveform.
                keep, t = audio_slice(root, spec, rate, count, start, end)
                audio.seek(keep.start)
                x = audio.read(keep.stop-keep.start, dtype="float32", always_2d=True)
        except ImportError:
            with wave.open(str(path), "rb") as audio:
                if audio.getsampwidth() not in (1, 2, 4):
                    raise RuntimeError("This PCM width requires soundfile from the media extra")
                keep, t = audio_slice(root, spec, audio.getframerate(), audio.getnframes(), start, end)
                audio.setpos(keep.start)
                width = audio.getsampwidth()
                dtype = {1: "u1", 2: "<i2", 4: "<i4"}[width]
                x = np.frombuffer(audio.readframes(keep.stop-keep.start), dtype=dtype).astype(np.float32).reshape(-1, audio.getnchannels())
                x = (x-128)/128 if width == 1 else x/(2**(width*8-1))
    elif fmt == "vrs":
        from .vrs import read_vrs
        x, t = read_vrs(root, spec, kind, start, end)
    elif fmt == "static_points":
        # A mapped scene is static context, not a time series of independent measurements.
        x, t = [read_points(local_path(root, spec["path"]))], np.asarray([start], dtype=np.float64)
        metadata["static"] = True
    else:
        raise ValueError(f"Unknown local source format: {fmt}")
    validate_times(t, len(x))
    return {"values": x, "timestamps": t, "kind": kind, "metadata": metadata}


def read_annotations(root, spec):
    """Annotations are kept as targets with their original fields and tracks."""
    fmt, path = spec.get("format", "json"), local_path(root, spec["path"])
    if fmt == "srt":
        def parse_time(value):
            h, m, s, ms = map(int, re.split(r"[:,.]", value.strip()))
            return h*3600+m*60+s+ms/1000
        rows = []
        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
            lines = block.splitlines()
            line = next((i for i, s in enumerate(lines) if "-->" in s), None)
            if line is None:
                continue
            a, b = lines[line].split("-->")
            rows.append({"start": parse_time(a), "end": parse_time(b), "text": "\n".join(lines[line+1:])})
    elif fmt == "csv":
        with path.open() as handle:
            rows = list(csv.DictReader(handle, delimiter=spec.get("delimiter", ",")))
    elif fmt == "jsonl":
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    else:
        rows = json.loads(path.read_text())
        if "key" in spec:
            rows = rows[spec["key"]]
    result = []
    for row in rows:
        mapping = spec.get("fields", {})
        row = {**row, **{new: row[old] for new, old in mapping.items()}}
        # Strings in CSV may represent large integer nanosecond clocks.
        raw = [int(row[k]) if spec.get("integer_timestamps") else float(row[k]) for k in ("start", "end")]
        start, end = seconds(raw, spec, root)
        result.append({**row, "start": float(start), "end": float(end), "track": spec.get("track", row.get("track", "activity"))})
    return result
