from dataclasses import replace
import unittest
import torch
from distrixsense.config import Config
from distrixsense.models import DistributedModel
from distrixsense.temporal_baselines import CausalBlock, TemporalBaseline, METHODS


class TemporalBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.cfg = Config(modalities={"imu": 3, "audio": 2}, hidden=8, heads=2,
                          layers=2, resample_length=15, tokens=3, patch_length=6,
                          patch_stride=4, dropout=0, modality_dropout=0)
        self.streams = {n: {"x": torch.randn(2, 10, c), "time": torch.linspace(0, 2, 10).repeat(2, 1),
                            "mask": torch.ones(2, 10, dtype=torch.bool)} for n, c in self.cfg.modalities.items()}

    def test_gradients_masked_inputs_and_checkpoint(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = DistributedModel(replace(self.cfg, method=method)).freeze_encoders()
                self.streams["audio"]["mask"][0].zero_()
                batch = {"streams": self.streams, "labels": torch.tensor([0, 1])}
                result = model(batch)
                model.objective(result, batch["labels"]).backward()
                self.assertGreater(float(model.temporal_baseline.projection.weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.temporal_baseline.classifier.weight.grad.abs().sum()), 0)
                self.assertTrue(all(p.grad is None for p in model.encoders.parameters()))
                model.eval()
                with torch.no_grad():
                    expected = model(batch)["logits"]
                    self.streams["audio"]["x"][0].fill_(99999)
                    self.streams["audio"]["time"][0].fill_(-99999)
                    torch.testing.assert_close(model(batch)["logits"], expected)
                    restored = DistributedModel(replace(self.cfg, method=method)).eval()
                    restored.load_state_dict(model.state_dict())
                    torch.testing.assert_close(restored(batch)["logits"], expected)

    def test_tcn_is_causal(self):
        block = CausalBlock(4, 3, 4, 0).eval()
        x = torch.randn(2, 4, 20)
        expected = block(x)
        x[:, :, 10:] += 100
        torch.testing.assert_close(block(x)[:, :, :10], expected[:, :, :10])

    def test_patch_tail_coverage_and_missing_channel_exclusion(self):
        cfg = replace(self.cfg, method="patchtst")
        model = TemporalBaseline(cfg).eval()
        x = torch.randn(2, 15, 5)
        availability = torch.tensor([[True, False], [False, False]])
        times = torch.arange(15).float().repeat(2, 1)
        with torch.no_grad():
            first = model(x, times, availability)
            self.assertEqual(first["sensor_tokens"].shape, (2, 4, 8))
            self.assertFalse(first["sensor_mask"][1].any())
            x[:, :, 3:] += 999
            torch.testing.assert_close(model(x, times, availability)["logits"], first["logits"])
            x[0, -1, :3] += 100
            self.assertGreater(float((model(x, times, availability)["sensor_tokens"][0]-first["sensor_tokens"][0]).abs().max()), 1e-5)

    def test_invalid_patch_configuration(self):
        with self.assertRaisesRegex(ValueError, "patch_stride"):
            replace(self.cfg, method="patchtst", patch_length=16).validate()
        with self.assertRaisesRegex(ValueError, "patch_stride"):
            replace(self.cfg, method="patchtst", patch_stride=7).validate()


if __name__ == "__main__":
    unittest.main()
