"""Helpers for constructing source descriptors from local official metadata."""
from pathlib import Path
import re
import numpy as np


def opportunity_sensor_streams(dat_path, column_names, clock_origin_ms=0):
    """All 242 sensor channels, grouped by metadata; never include label tracks."""
    groups = {}
    for line in Path(column_names).read_text().splitlines():
        match = re.match(r"\s*Column:\s*(\d+)\s+(.+?)\s*$", line)
        if not match or not 1 <= int(match[1])-1 < 243:
            continue
        index, description = int(match[1])-1, match[2]
        name = re.sub(r"[^a-zA-Z0-9_-]+", "_", re.sub(r"[XYZxyz1-4]$", "", description)).strip("_")
        lower = description.lower()
        modality = ("switch" if "reed" in lower or "switch" in lower else
                    "uwb" if "location" in lower or "uwb" in lower else
                    "orientation" if "quat" in lower else
                    "magnetometer" if "mag" in lower or "compass" in lower else
                    "gyroscope" if "gyro" in lower or "angvel" in lower else "accelerometer")
        groups.setdefault(name, {"modality": modality, "kind": "signal", "format": "opportunity_dat",
                                "path": str(dat_path), "time_column": 0, "columns": [],
                                "time_unit": "ms", "time_origin": clock_origin_ms})["columns"].append(index)
    if sorted(c for spec in groups.values() for c in spec["columns"]) != list(range(1, 243)):
        raise ValueError("Metadata must describe each of the 242 sensor columns once")
    return groups


def nymeria_xsens_streams(path, clock_origin_us=0):
    """Expose every time-aligned numeric XSens array, including release additions.

    Identity/T-pose models and static metadata are deliberately not observations.
    Coordinates are retained in their source frame; no guessed Aria alignment.
    """
    streams = {}
    with np.load(path, allow_pickle=False) as arrays:
        times = arrays["timestamps_us"]
        for key in arrays.files:
            x = arrays[key]
            if key == "timestamps_us" or key.startswith(("identity_", "tpose_")):
                continue
            if x.ndim < 1 or len(x) != len(times) or not np.issubdtype(x.dtype, np.number):
                continue
            spec = {"path": str(Path(path).resolve()), "key": key, "timestamp_key": "timestamps_us",
                    "format": "array", "modality": "xsens", "kind": "pose", "device": "xsens_body",
                    "time_unit": "us", "time_origin": clock_origin_us, "coordinate_frame": "xsens_world"}
            if key in ("segment_tXYZ", "segment_qWXYZ", "sensor_tXYZ", "sensor_qWXYZ"):
                coords = 4 if "qWXYZ" in key else 3
                spec["frame_shape"] = [int(np.prod(x.shape[1:]))//coords, coords]
            streams[f"xsens_{key}"] = spec
    return streams
