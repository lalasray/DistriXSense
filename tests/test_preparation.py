import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from distrixsense.config import Config
from distrixsense.data import ManifestDataset, fit_stats
from distrixsense.prepare import metadata_groups, prepare_opportunity, prepare_records


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def original_files(self):
        source = self.root/"raw"
        source.mkdir()
        for p in range(1, 5):
            data = np.ones((12, 250), dtype=np.float32)*p
            data[:, 0] = np.arange(12)*33.333
            data[:, 249] = np.repeat([0, 406516], 6)
            data[:, 244] = 9999  # ensure annotations never become features
            data[2, 7] = np.nan
            np.savetxt(source/f"S{p}-ADL1.dat", data)
        return source

    def test_opportunity_boundaries_labels_and_feature_exclusion(self):
        path = prepare_opportunity(self.original_files(), self.root/"prepared", window=4, stride=2)
        cfg = Config.load(path.parent/"config.json")
        self.assertEqual(cfg.modalities, {"body_worn": 145, "objects": 60, "ambient": 37})
        self.assertEqual(cfg.classes, 2)
        train = ManifestDataset(path, "train", cfg.modalities)
        for sample in train:
            self.assertEqual(sum(s["x"].shape[1] for s in sample["streams"].values()), 242)
            self.assertFalse(any((s["x"] == 9999).any() for s in sample["streams"].values()))
            self.assertIsNotNone(sample["events"])
        fit_stats(train)
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        self.assertTrue(all(r["end_row"] <= 12 for r in rows))
        self.assertEqual(set(r["participant"] for r in rows if r["split"] == "test"), {"S4"})

    def test_reject_annotation_group_and_participant_overlap(self):
        source = self.original_files()
        group = self.root/"groups.json"
        group.write_text(json.dumps({"leak": [249]}))
        with self.assertRaises(ValueError):
            prepare_opportunity(source, self.root/"bad1", group_map=group)
        with self.assertRaises(ValueError):
            prepare_opportunity(source, self.root/"bad2", train_people=("S1", "S3"))

    def test_metadata_sensor_groups(self):
        path = self.root/"columns.txt"
        path.write_text("Column: 1 MILLISEC\n"+"\n".join(f"Column: {i+1} sensor{i//3} acc{'XYZ'[i%3]}" for i in range(1, 243))+"\nColumn: 250 label_gesture")
        groups = metadata_groups(path)
        self.assertEqual(sorted(i for v in groups.values() for i in v), list(range(1, 243)))
        self.assertFalse(any("label" in k for k in groups))

    def test_recording_adapter_preserves_asynchronous_streams(self):
        rows = []
        for i, split in enumerate(("train", "val", "test")):
            streams = {}
            for n, step in (("imu", .1), ("vision", .25)):
                t = np.arange(0, 4, step)
                np.save(self.root/f"{i}-{n}-time.npy", t)
                np.save(self.root/f"{i}-{n}.npy", np.ones((len(t), 3)))
                streams[n] = {"values": f"{i}-{n}.npy", "timestamps": f"{i}-{n}-time.npy"}
            rows.append({"participant": f"p{i}", "split": split, "streams": streams,
                         "annotations": [{"start": 0, "end": 4, "label": "assembly", "query": "What?", "answer": "Assembly"}]})
        records = self.root/"recordings.jsonl"
        records.write_text("".join(json.dumps(r)+"\n" for r in rows))
        path = prepare_records(records, self.root/"windows", "openmarcie", 1., .5)
        cfg = Config.load(path.parent/"config.json")
        ds = ManifestDataset(path, "train", cfg.modalities)
        sample = ds[0]
        self.assertNotEqual(len(sample["streams"]["imu"]["x"]), len(sample["streams"]["vision"]["x"]))
        self.assertEqual(sample["answer"], "Assembly")
        self.assertEqual(sample["events"]["labels"], [0, 0])


if __name__ == "__main__":
    unittest.main()
