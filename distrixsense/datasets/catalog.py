"""Modality families, not a claim that every recording contains every sensor.

Device/view identifiers are arbitrary stream names, so multiple cameras, body
locations, and Aria devices remain independent. Annotation tracks are targets.
"""
DATASETS = ("opportunity++", "openmarcie", "nymeria")
KINDS = {"signal", "image", "audio", "pose", "points", "text", "features"}

COMMON_MOTION = {"imu": "signal", "accelerometer": "signal", "gyroscope": "signal",
                 "magnetometer": "signal", "orientation": "signal"}
CATALOG = {
    "opportunity++": {
        "modalities": {**COMMON_MOTION, "shoe_imu": "signal", "uwb": "signal",
                       "switch": "signal", "ambient_accelerometer": "signal",
                       "object_imu": "signal", "rgb": "image", "body25": "pose"},
        "annotation_tracks": ["activity", "locomotion", "left_action", "right_action",
                              "left_object", "right_object", "both_arms", "subtitles"],
        "source": "https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.792065/full",
    },
    "openmarcie": {
        "modalities": {**COMMON_MOTION, "barometer": "signal", "temperature": "signal",
                       "spectrometer": "signal", "thermal": "image", "rgb": "image",
                       "depth": "image", "lidar_depth": "image", "lidar_points": "points",
                       "audio": "audio", "pose2d": "pose", "pose3d": "pose",
                       "object_tracks": "signal", "position": "signal"},
        "annotation_tracks": ["activity", "object", "tool", "concurrent_action",
                              "caption", "transcript", "remarks"],
        "source": "https://arxiv.org/html/2603.02390v1",
    },
    "nymeria": {
        "modalities": {**COMMON_MOTION, "rgb": "image", "slam": "image", "eye_images": "image",
                       "gaze": "signal", "barometer": "signal", "audio": "audio",
                       "trajectory": "signal", "pointcloud": "points", "xsens": "pose",
                       "body_position": "pose", "body_orientation": "pose",
                       "body_velocity": "pose", "body_acceleration": "pose",
                       "joint_angles": "pose", "foot_contacts": "signal"},
        "annotation_tracks": ["activity_summarization", "atomic_action", "motion_narration"],
        "source": "https://github.com/facebookresearch/nymeria_dataset/tree/nymeria_dataset_legacy",
    },
}


def stream_kind(dataset, name, spec):
    if dataset not in DATASETS:
        raise ValueError(f"Dataset must be one of {DATASETS}")
    modality = spec.get("modality", name)
    kind = spec.get("kind", CATALOG[dataset]["modalities"].get(modality))
    if kind not in KINDS:
        raise ValueError(f"Stream {name}: specify a catalog modality or explicit kind from {sorted(KINDS)}")
    return kind
