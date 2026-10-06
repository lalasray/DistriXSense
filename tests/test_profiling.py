from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from distrixsense.config import Config, METHODS
from distrixsense.models import DistributedModel
from distrixsense.profiling import (arithmetic, edge_modules, link_cost, profile_model,
                                   run_suite, storage, synthetic_batch)


class ProfilingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = Config(modalities={"imu": 3, "audio": 2}, hidden=8, heads=2, layers=1,
                          tokens=2, resample_length=8, dropout=0, modality_dropout=0)
        self.inputs = {"imu": {"frames": 12}, "audio": {"frames": 15}}
        self.batch = synthetic_batch(self.cfg, self.inputs, 3)
        self.topology = {"stream_devices": {"imu": "wrist", "audio": "microphone"},
                         "edges": {"wrist": {"bandwidth_mbps": 1, "latency_ms": 2},
                                   "microphone": {"bandwidth_mbps": 10, "latency_ms": 3}},
                         "core_device": "cpu"}

    def test_flops_match_known_linear_and_convolution(self):
        linear = torch.nn.Linear(3, 7)
        self.assertEqual(arithmetic(lambda: linear(torch.ones(2, 5, 3)))["counted_flops"], 2*2*5*3*7)
        conv = torch.nn.Conv1d(3, 7, 3)
        self.assertEqual(arithmetic(lambda: conv(torch.ones(2, 3, 5)))["counted_flops"], 2*2*3*7*3*3)
        q = torch.randn(1, 2, 3, 4)
        self.assertEqual(arithmetic(lambda: torch.nn.functional.scaled_dot_product_attention(q, q, q))["counted_flops"], 288)

    def test_network_fragmentation_and_empty_links(self):
        size = {"packets": 2, "application_bytes": 3000, "application_packet_sizes": [2000, 1000]}
        link = link_cost(size, {"bandwidth_mbps": 2, "latency_ms": 5,
                                "overhead_bytes_per_packet": 28, "mtu_payload_bytes": 1000}, 1)
        self.assertEqual(link["wire_bytes"], 3084)
        self.assertAlmostEqual(link["transfer_ms"], 17.336)
        empty = link_cost({"packets": 0, "application_bytes": 0}, {"latency_ms": 5}, 1)
        self.assertEqual(empty["transfer_ms"], 0)

    def test_physical_device_bank_replication(self):
        model = DistributedModel(replace(self.cfg, method="joint_vq"))
        a = storage(edge_modules(model, ["imu"]))
        b = storage(edge_modules(model, ["audio"]))
        together = storage(edge_modules(model, ["imu", "audio"]))
        self.assertEqual(a["parameters"]+b["parameters"]-together["parameters"], model.bank.weight.numel())
        proposed = DistributedModel(self.cfg)
        self.assertNotIn(proposed.bank, edge_modules(proposed, ["imu"]))

    def test_all_methods_profile_independent_devices(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = DistributedModel(replace(self.cfg, method=method)).eval()
                model.posthoc_ready.fill_(True)
                result = profile_model(model, self.batch, self.topology, 1, 0)
                self.assertLess(result["packet_logit_max_difference"], 1e-5)
                self.assertEqual(result["total_deployed"]["parameters"],
                                 result["core"]["storage"]["parameters"]+sum(e["storage"]["parameters"] for e in result["edges"].values()))
                for edge in result["edges"].values():
                    net = edge["network"]
                    self.assertEqual(net["application_bytes"], net["header_bytes"]+net["timestamp_bytes"]+net["representation_bytes"])
                if method in ("raw", "deepconvlstm", "imagebind", "tcn"):
                    self.assertTrue(all(e["storage"]["parameters"] == 0 for e in result["edges"].values()))

    def test_missing_modality_and_gpu_availability_report(self):
        batch = synthetic_batch(self.cfg, {"imu": {"frames": 3, "available": False}, "audio": {"frames": 3}}, 3)
        result = profile_model(DistributedModel(self.cfg), batch, self.topology, 1, 0)
        self.assertEqual(result["edges"]["wrist"]["network"]["wire_bytes"], 0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            spec = {"datasets": [{"dataset": "nymeria", "config": self.cfg.to_dict(),
                                   "inputs": self.inputs, "window_seconds": 3, "stride_seconds": 1.5}]}
            (path/"spec.json").write_text(json.dumps(spec))
            with patch("torch.cuda.is_available", return_value=False):
                run_suite(path/"spec.json", path/"out", ("cuda", "mixed"), ("distrixsense",), 1, 0)
            self.assertEqual(len(json.loads((path/"out/availability.json").read_text())), 2)

    def test_resume_preserves_completed_measurements_and_rejects_setting_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            spec = {"datasets": [{"dataset": "nymeria", "config": self.cfg.to_dict(),
                                   "inputs": self.inputs, "window_seconds": 3, "stride_seconds": 1.5}]}
            (path/"spec.json").write_text(json.dumps(spec))
            run_suite(path/"spec.json", path/"out", ("cpu",), ("distrixsense",), 1, 0)
            report = path/"out/nymeria-cpu-distrixsense.json"
            original = report.read_bytes()
            with patch("distrixsense.profiling.profile_model", side_effect=AssertionError("Must not rerun")):
                run_suite(path/"spec.json", path/"out", ("cpu",), ("distrixsense",), 1, 0, resume=True)
            self.assertEqual(report.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "matching timing"):
                run_suite(path/"spec.json", path/"out", ("cpu",), ("distrixsense",), 2, 0, resume=True)
            spec['datasets'][0]['window_seconds'] = 4
            (path/"spec.json").write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "different input"):
                run_suite(path/"spec.json", path/"out", ("cpu",), ("distrixsense",), 1, 0, resume=True)

    def test_transport_validation_separates_reencoding_differences(self):
        model = DistributedModel(replace(self.cfg, method="dense")).eval()
        peripheral = model.peripheral
        def different_reference(streams):
            messages, aux = peripheral(streams)
            if len(streams) == len(self.cfg.modalities):
                name = next(iter(messages))
                values = messages[name].values.clone()
                values[..., 0] += 5
                messages[name] = replace(messages[name], values=values)
            return messages, aux
        with patch.object(model, "peripheral", side_effect=different_reference):
            result = profile_model(model, self.batch, self.topology, 1, 0)
        self.assertLess(result['packet_logit_max_difference'], 1e-6)
        self.assertGreater(result['colocated_reference_logit_max_difference'], 1e-5)


if __name__ == "__main__":
    unittest.main()
