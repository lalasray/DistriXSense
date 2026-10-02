"""Optional Aria reader: preserve per-sample clocks; no dataset download APIs."""
from functools import lru_cache
import os
import numpy as np
from .readers import local_path, canonical, seconds, select, validate_times


@lru_cache(maxsize=8)
def provider_for(path, pid):
    # pid prevents sharing a native VRS provider after a DataLoader fork.
    try:
        from projectaria_tools.core import data_provider
    except ImportError as exc:
        raise RuntimeError("VRS requires projectaria-tools in a supported Python environment") from exc
    provider = data_provider.create_vrs_data_provider(path)
    if provider is None:
        raise ValueError(f"Cannot open local VRS: {path}")
    return provider


def read_vrs(root, spec, kind, start, end):
    provider = provider_for(str(local_path(root, spec["path"]).resolve()), os.getpid())
    from projectaria_tools.core.sensor_data import TimeDomain
    from projectaria_tools.core.stream_id import StreamId
    stream = StreamId(spec["stream_id"]) if "stream_id" in spec else provider.get_stream_id_from_label(spec["stream_label"])
    domain_name = spec.get("time_domain", "TIME_CODE")
    if domain_name not in ("TIME_CODE", "DEVICE_TIME"):
        raise ValueError("Use TIME_CODE for synchronized devices or explicit DEVICE_TIME clock mappings")
    domain = getattr(TimeDomain, domain_name)
    raw_times = np.asarray(provider.get_timestamps_ns(stream, domain), dtype=np.int64)
    clock_spec = {**spec, "time_unit": "ns"}
    t = seconds(raw_times, clock_spec)
    validate_times(t, len(t))
    keep = select(t, start, end)
    indices = range(max(0, keep.start-1) if kind == "audio" else keep.start, keep.stop)
    values, times = [], []
    for index in indices:
        sample = provider.get_sensor_data_by_index(stream, index)
        if kind == "image":
            image, _ = sample.image_data_and_record()
            values.append(image.to_numpy_array())
            times.append(t[index])
        elif kind == "audio":
            audio, record = sample.audio_data_and_record()
            raw = np.asarray(record.capture_timestamps_ns, dtype=np.int64)
            if domain_name == "TIME_CODE":
                raw = np.asarray([provider.convert_from_device_time_to_timecode_ns(int(v)) for v in raw], dtype=np.int64)
            stamps = seconds(raw, clock_spec)
            channels = int(spec.get("channels", provider.get_audio_configuration(stream).num_channels))
            waveform = np.asarray(audio.data).reshape(-1, channels)
            if len(waveform) != len(stamps):
                raise ValueError("VRS audio sample timestamps do not match waveform")
            chosen = (stamps >= start)&(stamps < end)
            values.extend(waveform[chosen])
            times.extend(stamps[chosen])
        else:
            modality = spec.get("modality", "imu")
            if modality == "barometer":
                record = sample.barometer_data()
                fields = spec.get("fields", ["pressure"])
            elif modality == "magnetometer":
                record = sample.magnetometer_data()
                fields = spec.get("fields", ["mag_tesla"])
            else:
                record = sample.imu_data()
                fields = spec.get("fields", ["accel_msec2", "gyro_radsec"])
            values.append(np.concatenate([np.asarray(getattr(record, f)).reshape(-1) for f in fields]))
            times.append(t[index])
    if not values:
        return [], np.empty(0, dtype=np.float64)
    layout = "THWC" if kind == "image" and np.asarray(values[0]).ndim == 3 else "THW"
    return canonical(np.stack(values), kind, {**spec, "layout": spec.get("layout", layout)}), np.asarray(times)
