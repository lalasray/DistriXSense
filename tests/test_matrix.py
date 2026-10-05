import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import torch
from distrixsense.config import Config
from distrixsense.data import make_synthetic
from distrixsense.matrix import expand_matrix, run_matrix
from distrixsense.models import DistributedModel
from distrixsense.profiling import profile_language, synthetic_batch
from test_language import tiny_local_lm, HAS_TRANSFORMERS


class MatrixTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = Config(hidden=8, heads=2, layers=1, tokens=2, resample_length=8,
                          epochs=1, pretrain_epochs=1, batch_size=3, dropout=0, modality_dropout=0)
        self.spec = {"cores": [{"name": "a", "checkpoint": "lm-a", "decode_tokens": 2},
                               {"name": "b", "checkpoint": "lm-b", "decode_tokens": 2}],
                     "methods": ["distrixsense"], "seeds": [0],
                     "datasets": [{"dataset": "nymeria", "config": self.cfg.to_dict(),
                                   "inputs": {"imu": {"frames": 5}, "ambient": {"frames": 7}},
                                   "window_seconds": 3, "stride_seconds": 1.5,
                                   "include_each_modality": True}]}
        self.path = self.root/"spec.json"

    def save_spec(self):
        self.path.write_text(json.dumps(self.spec))

    def test_expansion_isolates_each_modality_core_and_physical_node(self):
        self.spec["datasets"][0]["topology"] = {"stream_devices": {"imu": "wrist", "ambient": "room"}}
        self.save_spec()
        plan = expand_matrix(self.path)
        self.assertEqual(len(plan["datasets"]), 6)
        self.assertEqual(len(set(d["scenario_id"] for d in plan["datasets"])), 6)
        for d in plan["datasets"]:
            self.assertEqual(set(d["config"]["modalities"]), set(d["inputs"]))
            self.assertEqual(set(d["config"]["modalities"]), set(d["topology"]["stream_devices"]))
            self.assertTrue(Path(d["language"]["checkpoint"]).is_absolute())

    def test_unsupported_method_subset_is_explicit(self):
        d = self.spec["datasets"][0]
        d["method_profiles"] = {"imagebind": {"config": replace(self.cfg, modalities={"imu": 1024}).to_dict(),
                                              "inputs": {"imu": {"frames": 1}}}}
        self.save_spec()
        plan = expand_matrix(self.path)
        excluded = [d for d in plan["datasets"] if d["modality_set"] == "only_ambient"]
        self.assertTrue(all("imagebind" in d["excluded_methods"] for d in excluded))

    def test_dry_run_requires_no_local_weights_or_data(self):
        self.save_spec()
        result = run_matrix(self.path, self.root/"dry", dry_run=True)
        self.assertTrue(result.exists())

    @unittest.skipUnless(HAS_TRANSFORMERS, "Optional language dependencies not installed")
    def test_each_modality_profiles_two_different_llm_architectures(self):
        tiny_local_lm(self.root/"lm-a", width=16)
        tiny_local_lm(self.root/"lm-b", width=24)
        self.save_spec()
        result = run_matrix(self.path, self.root/"profile", targets=("cpu",), iterations=1, warmup=0)
        with (result.parent/"summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["modality_set"] for r in rows}, {"all", "only_imu", "only_ambient"})
        a = next(r for r in rows if r["core_name"] == "a" and r["modality_set"] == "all")
        b = next(r for r in rows if r["core_name"] == "b" and r["modality_set"] == "all")
        self.assertGreater(int(b["full_parameters"]), int(a["full_parameters"]))
        self.assertGreater(int(a["full_parameters"]), int(a["parameters"]))
        self.assertEqual(a["wire_bytes"], b["wire_bytes"])
        self.assertEqual(len(list(result.parent.glob('*-cpu-distrixsense.json'))), 6)

    @unittest.skipUnless(HAS_TRANSFORMERS, "Optional language dependencies not installed")
    def test_core_shape_reuse_is_explicit_and_decode_budget_invalidates_it(self):
        tiny_local_lm(self.root/"lm-a", width=16)
        batch = synthetic_batch(self.cfg, self.spec['datasets'][0]['inputs'], 3)
        model = DistributedModel(self.cfg).eval()
        spec = dict(checkpoint='lm-a', decode_tokens=2)
        cores, shapes = {}, {}
        first = profile_language(model, batch, spec, self.root, 'cpu', 1, 0,
                                 core_cache=cores, shape_cache=shapes)
        second = profile_language(model, batch, spec, self.root, 'cpu', 1, 0,
                                  core_cache=cores, shape_cache=shapes)
        self.assertEqual(first['language_measurement'], 'measured_this_case')
        self.assertEqual(second['language_measurement'], 'shape_matched_reuse')
        self.assertEqual(first['language_shape_id'], second['language_shape_id'])
        self.assertEqual(first['generation_arithmetic'], second['generation_arithmetic'])
        self.assertEqual(first['generated_tokens'], 2)
        third = profile_language(model, batch, {**spec,'decode_tokens':3}, self.root, 'cpu', 1, 0,
                                 core_cache=cores, shape_cache=shapes)
        self.assertEqual(third['language_measurement'], 'measured_this_case')
        self.assertNotEqual(first['language_shape_id'], third['language_shape_id'])
        self.assertEqual(third['generated_tokens'], 3)

    @unittest.skipUnless(HAS_TRANSFORMERS, "Optional language dependencies not installed")
    def test_training_matrix_retains_core_specific_checkpoints_and_answers(self):
        tiny_local_lm(self.root/"lm-a", width=16)
        tiny_local_lm(self.root/"lm-b", width=24)
        manifest = make_synthetic(self.root/"data", samples=3)
        rows = [json.loads(s) for s in manifest.read_text().splitlines()]
        for row in rows:
            row.update(query="What happened?", answer="fixture answer")
        manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
        self.spec["datasets"][0].update(manifest=str(manifest), include_each_modality=False)
        self.save_spec()
        result = run_matrix(self.path, self.root/"train", mode="train")
        with result.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["core"] for r in rows}, {"a", "b"})
        for row in rows:
            self.assertTrue(Path(row["checkpoint"]).exists())
            self.assertTrue((Path(row["checkpoint"]).parent/"answers.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
