import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import numpy as np
import torch
from distrixsense.datasets import (CATALOG, RecordingDataset, OpportunityPlusPlusDataset,
                                   OpenMarcieDataset, NymeriaDataset, multimodal_collate)
from distrixsense.datasets.features import feature_sequence
from distrixsense.datasets.native import opportunity_sensor_streams, nymeria_xsens_streams
from distrixsense.datasets.readers import read_stream, read_annotations, seconds
from distrixsense.prepare import prepare_records
from distrixsense.config import Config
from distrixsense.data import ManifestDataset, collate, fit_stats
from distrixsense.models import DistributedModel


class DatasetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, dataset, all_families=True):
        rows = []
        for participant, split in enumerate(("train", "val", "test")):
            streams = {}
            for name, kind in CATALOG[dataset]["modalities"].items():
                t = np.arange(8)/4
                if kind == "image":
                    x = np.full((8, 4, 5, 3 if name == "rgb" else 1), 128, dtype=np.uint8)
                elif kind == "pose":
                    x = np.ones((8, 5, 3), dtype=np.float32)
                elif kind == "points":
                    x = np.ones((8, 6, 3), dtype=np.float32)
                elif kind == "audio":
                    t = np.arange(64)/32
                    x = np.ones((64, 2), dtype=np.float32)
                else:
                    x = np.ones((8, 3), dtype=np.float32)
                path = f"p{participant}-{name}.npz"
                np.savez(self.root/path, values=x, time=t)
                streams[name] = {"modality": name, "format": "array", "path": path,
                                 "key": "values", "timestamp_key": "time", "layout": "THWC"}
            rows.append({"dataset": dataset, "recording": f"record-{participant}", "participant": f"p{participant}",
                         "split": split, "start": 0., "end": 2., "streams": streams,
                         "annotations": [{"track": "activity", "start": 0., "end": 2., "label": "assembly"}]})
        path = self.root/f"{dataset}.jsonl"
        self.write_rows(path, rows)
        return path, rows

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))

    def test_every_catalog_modality_in_all_three_datasets(self):
        classes = {"opportunity++": OpportunityPlusPlusDataset, "openmarcie": OpenMarcieDataset, "nymeria": NymeriaDataset}
        for dataset, cls in classes.items():
            with self.subTest(dataset=dataset):
                path, _ = self.fixture(dataset)
                data = cls(path, split="train", window_seconds=1., stride_seconds=1.)
                self.assertEqual(len(data), 2)
                sample = data[0]
                self.assertEqual(set(sample["streams"]), set(CATALOG[dataset]["modalities"]))
                batch = multimodal_collate([data[0], data[1]], data.modalities)
                self.assertTrue(batch["availability"].all())
                self.assertEqual(batch["streams"]["rgb"]["x"].shape, (2, 4, 3, 4, 5))
                self.assertEqual(batch["streams"]["rgb"]["time"].dtype, torch.float64)

    def test_missing_sensor_and_missing_values_masks(self):
        path, rows = self.fixture("openmarcie")
        rows[1]["streams"].pop("rgb")
        self.write_rows(path, rows)
        data = OpenMarcieDataset(path, window_seconds=1., stride_seconds=1.)
        first, second = data[0], data[2]
        first["streams"]["imu"]["x"][0, 1] = float("nan")
        batch = multimodal_collate([first, second], data.modalities)
        self.assertFalse(batch["availability"][1, data.modalities.index("rgb")])
        self.assertFalse(batch["streams"]["rgb"]["mask"][1].any())
        self.assertFalse(batch["streams"]["imu"]["finite_mask"][0, 0, 1])
        self.assertTrue(torch.isfinite(batch["streams"]["imu"]["x"]).all())

    def test_nanosecond_clock_preserves_resolution_and_boundaries(self):
        origin = 1_700_000_000_000_000_000
        raw = np.asarray([origin, origin+1, origin+2], dtype=np.int64)
        np.testing.assert_allclose(seconds(raw, {"time_unit": "ns", "time_origin": origin}), [0, 1e-9, 2e-9], atol=1e-20)
        np.savez(self.root/"clock.npz", x=np.arange(4)[:, None], t=np.array([1000, 1100, 1200, 1300]))
        s = read_stream(self.root, {"path": "clock.npz", "key": "x", "timestamp_key": "t", "time_unit": "ms", "time_origin": 1000}, "signal", .1, .3)
        np.testing.assert_array_equal(s["values"][:, 0], [1, 2])

    def test_overlap_annotations_kept_and_never_input(self):
        path, rows = self.fixture("nymeria")
        rows[0]["annotations"].append({"track": "motion_narration", "start": .2, "end": .8, "text": "moving"})
        rows[0]["annotations"].append({"track": "activity", "start": .4, "end": 1.5, "label": "walking"})
        self.write_rows(path, rows)
        data = NymeriaDataset(path, split="train", window_seconds=1.)
        self.assertEqual(len(data[0]["annotations"]), 3)
        self.assertNotIn("motion_narration", data[0]["streams"])

    def test_participant_and_source_leakage(self):
        path, rows = self.fixture("opportunity++")
        rows[1]["participants"] = ["p0", "p1"]
        self.write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "Participant leakage"):
            OpportunityPlusPlusDataset(path, window_seconds=1.)
        rows[1].pop("participants")
        rows[1]["streams"]["imu"]["path"] = rows[0]["streams"]["imu"]["path"]
        self.write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "Source recording leakage"):
            OpportunityPlusPlusDataset(path, window_seconds=1.)

    def test_variable_point_counts_and_empty_cloud(self):
        files = []
        for i, count in enumerate((2, 0, 5)):
            path = self.root/f"cloud{i}.npy"
            np.save(path, np.ones((count, 3)))
            files.append(path.name)
        stream = read_stream(self.root, {"format": "point_frames", "files": files, "sample_rate": 2}, "points", 0, 2)
        self.assertEqual([len(v) for v in stream["values"]], [2, 0, 5])
        path, rows = self.fixture("nymeria")
        rows[0]["streams"]["pointcloud"] = {"modality": "pointcloud", "format": "point_frames", "files": files, "sample_rate": 2}
        self.write_rows(path, rows)
        sample = NymeriaDataset(path, split="train", window_seconds=2.)[0]
        batch = multimodal_collate([sample])
        self.assertEqual(batch["streams"]["pointcloud"]["point_mask"].sum().item(), 7)
        self.assertFalse(batch["streams"]["pointcloud"]["mask"][0, 1])

    def test_local_audio_window_preserves_channels_and_rate(self):
        path = self.root/"audio.wav"
        x = np.arange(80, dtype=np.int16).reshape(40, 2)
        with wave.open(str(path), "wb") as f:
            f.setnchannels(2)
            f.setsampwidth(2)
            f.setframerate(20)
            f.writeframes(x.tobytes())
        s = read_stream(self.root, {"format": "audio", "path": path.name}, "audio", .5, 1.)
        self.assertEqual(s["values"].shape, (10, 2))
        np.testing.assert_allclose(s["timestamps"], np.arange(10, 20)/20)
        np.testing.assert_allclose(s["values"], x[10:20]/32768)

    def test_csv_columns_and_annotation_exclusion(self):
        data = np.zeros((4, 250))
        data[:, 0] = np.arange(4)*100
        data[:, 1:4] = [1, 2, 3]
        data[:, 249] = 999
        np.savetxt(self.root/"session.dat", data)
        spec = {"format": "opportunity_dat", "path": "session.dat", "time_column": 0, "columns": [1, 2, 3], "time_unit": "ms"}
        s = read_stream(self.root, spec, "signal", 0, .4)
        self.assertEqual(s["values"].shape, (4, 3))
        with self.assertRaises(ValueError):
            read_stream(self.root, {**spec, "columns": [249]}, "signal", 0, .4)
        metadata = self.root/"columns.txt"
        metadata.write_text("\n".join(f"Column: {i+1} sensor{i} accX" for i in range(250)))
        specs = opportunity_sensor_streams("session.dat", metadata)
        self.assertEqual(sorted(c for s in specs.values() for c in s["columns"]), list(range(1, 243)))

    def test_nymeria_native_npz_all_time_aligned_arrays(self):
        path = self.root/"xdata.npz"
        np.savez(path, timestamps_us=np.array([100, 200, 300]), segment_tXYZ=np.zeros((3, 69)),
                 segment_qWXYZ=np.ones((3, 92)), foot_contacts=np.ones((3, 4)),
                 extra_measurement=np.ones((3, 7)), frameCount=3, identity_segment_tXYZ=np.zeros((3, 69)))
        streams = nymeria_xsens_streams(path, clock_origin_us=100)
        self.assertEqual(len(streams), 4)
        self.assertEqual(streams["xsens_segment_qWXYZ"]["frame_shape"], [23, 4])
        s = read_stream(self.root, streams["xsens_segment_tXYZ"], "pose", 0, .001)
        self.assertEqual(s["values"].shape, (3, 23, 3))

    def test_srt_and_csv_annotation_tracks(self):
        (self.root/"caption.srt").write_text("1\n00:00:00,250 --> 00:00:01,000\nA caption.\n")
        rows = read_annotations(self.root, {"format": "srt", "path": "caption.srt", "track": "subtitles"})
        self.assertEqual(rows[0]["start"], .25)
        self.assertEqual(rows[0]["track"], "subtitles")
        (self.root/"annotations.csv").write_text("begin,finish,narration\n1000,2000,walking\n")
        rows = read_annotations(self.root, {"format": "csv", "path": "annotations.csv", "fields": {"start": "begin", "end": "finish", "text": "narration"}, "time_unit": "ms", "time_origin": 1000})
        self.assertEqual(rows[0]["end"], 1.)

    def test_all_modalities_feature_export_reaches_existing_model(self):
        policies = {"image": {"method": "spatial_pool", "grid": 2}, "pose": "flatten_pose",
                    "points": "point_moments", "audio": {"method": "log_spectrum", "n_fft": 8, "hop": 4}}
        for dataset in CATALOG:
            with self.subTest(dataset=dataset):
                path, _ = self.fixture(dataset)
                manifest = prepare_records(path, self.root/f"prepared-{dataset}", dataset, 1., 1., policies)
                cfg = Config.load(manifest.parent/"config.json")
                self.assertEqual(set(cfg.modalities), set(CATALOG[dataset]["modalities"]))
                ds = ManifestDataset(manifest, "train", cfg.modalities)
                ds.stats = fit_stats(ds)
                batch = collate([ds[0]], cfg.modalities)
                model = DistributedModel(cfg).eval()
                with torch.no_grad():
                    out = model(batch)
                self.assertTrue(torch.isfinite(out["logits"]).all())

    def test_raw_images_require_explicit_feature_conversion(self):
        path, _ = self.fixture("opportunity++")
        sample = OpportunityPlusPlusDataset(path, split="train", window_seconds=1.)[0]
        with self.assertRaisesRegex(ValueError, "explicit feature"):
            feature_sequence(sample["streams"]["rgb"])

    def test_local_only_sources_and_clock_validation(self):
        with self.assertRaisesRegex(ValueError, "local paths"):
            read_stream(self.root, {"path": "https://example.com/data.npy", "sample_rate": 1}, "signal", 0, 1)
        with self.assertRaises(ValueError):
            seconds([1, 2], {"clock_scale": -1})

    def test_local_image_depth_and_video_decoding(self):
        try:
            import av
            from PIL import Image
        except ImportError:
            self.skipTest("Optional media dependencies are not installed")
        depth = np.arange(16, dtype=np.uint16).reshape(4, 4)*100
        Image.fromarray(depth).save(self.root/"depth.png")
        result = read_stream(self.root, {"format": "images", "files": ["depth.png"], "sample_rate": 1}, "image", 0, 1)
        np.testing.assert_array_equal(result["values"][0, 0], depth)
        with av.open(str(self.root/"video.mkv"), "w") as container:
            stream = container.add_stream("ffv1", rate=4)
            stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
            for i in range(4):
                frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i*40, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        result = read_stream(self.root, {"format": "video", "path": "video.mkv"}, "image", .25, .75)
        self.assertEqual(result["values"].shape, (2, 3, 16, 16))
        np.testing.assert_allclose(result["timestamps"], [.25, .5])

    def test_openpose_missing_detections_and_csv_time_exclusion(self):
        for i, people in enumerate(([{"pose_keypoints_2d": list(range(75))}], [])):
            (self.root/f"pose{i}.json").write_text(json.dumps({"people": people}))
        result = read_stream(self.root, {"format": "openpose", "files": ["pose0.json", "pose1.json"], "sample_rate": 2}, "pose", 0, 1)
        self.assertEqual(result["values"].shape, (2, 25, 3))
        self.assertTrue(np.isnan(result["values"][1]).all())
        (self.root/"sensor.csv").write_text("time,value\n0,1\n1,2\n")
        with self.assertRaisesRegex(ValueError, "Timestamp column"):
            read_stream(self.root, {"format": "csv", "path": "sensor.csv", "header": True, "columns": ["time", "value"], "time_column": "time"}, "signal", 0, 2)

    def test_explicit_device_to_timecode_mapping(self):
        class Provider:
            def convert_from_device_time_to_timecode_ns(self, value):
                return value+1000
        with patch("distrixsense.datasets.vrs.provider_for", return_value=Provider()):
            t = seconds(np.array([1, 2]), {"time_unit": "us", "clock_vrs": "clock.vrs", "timecode_origin_ns": 1000}, self.root)
        np.testing.assert_allclose(t, [1e-6, 2e-6])


if __name__ == "__main__":
    unittest.main()
