from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
from distrixsense.config import Config, METHODS
from distrixsense.data import ManifestDataset, collate, fit_stats, make_synthetic
from distrixsense.models import DistributedModel, Resampler
from distrixsense.training import fit_posthoc, pretrain_encoders, save_checkpoint, load_checkpoint
from distrixsense.transport import Message, HEADER, deserialize, payload_bytes, roundtrip, serialize
from distrixsense.metrics import classification, participant_bootstrap, answer_metrics, holm


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = make_synthetic(self.root, samples=3)
        self.cfg = Config(hidden=16, heads=2, layers=1, resample_length=16, tokens=4, batch_size=3,
                          pretrain_epochs=1, epochs=1, dropout=0, modality_dropout=0)
        self.dataset = ManifestDataset(self.manifest, "train", self.cfg.modalities)
        self.batch = collate([self.dataset[i] for i in range(3)], self.cfg.modalities)

    def test_all_methods_deploy_from_real_packets(self):
        one = collate([self.dataset[0]], self.cfg.modalities)
        for method in METHODS:
            with self.subTest(method=method):
                model = DistributedModel(replace(self.cfg, method=method)).freeze_encoders().eval()
                if method == "posthoc_vq":
                    dl = DataLoader(self.dataset, batch_size=3, collate_fn=partial(collate, modalities=self.cfg.modalities))
                    fit_posthoc(model, dl, iterations=2)
                with torch.no_grad():
                    messages, _ = model.peripheral(one["streams"])
                    expected = model.central(messages)["logits"]
                    decoded, packets = roundtrip(messages, model.names, "cpu")
                    actual = model.central(decoded)["logits"]
                torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)
                self.assertEqual(sum(map(len, packets)), payload_bytes(messages, model.names)[0])

    def test_gradients_freeze_encoder_but_train_selector_and_bank(self):
        model = DistributedModel(self.cfg).freeze_encoders().train()
        output = model(self.batch)
        model.objective(output, self.batch["labels"]).backward()
        self.assertTrue(all(p.grad is None for p in model.encoders.parameters()))
        self.assertGreater(float(model.selectors["imu"].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.bank.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.resamplers["imu"].refine[-1].weight.grad.abs().sum()), 0)
        for n, message in output["messages"].items():
            self.assertEqual(message.values.dtype, torch.int64)
            self.assertTrue((message.values >= model.offsets[n]).all())
            self.assertTrue((message.values < model.offsets[n]+model.bank_sizes[n]).all())

    def test_deployed_categorical_selector_never_reads_bank(self):
        for method in ("distrixsense", "unrestricted_bank", "separate_banks", "fixed_resampling"):
            model = DistributedModel(replace(self.cfg, method=method)).eval()
            def forbidden(*args):
                raise AssertionError("Peripheral accessed central bank")
            model.region = forbidden
            messages, aux = model.peripheral(self.batch["streams"])
            self.assertNotIn("surrogates", aux)
            self.assertTrue(all(m.values.ndim == 2 for m in messages.values()))

    def test_event_loss_trains_temporal_head(self):
        model = DistributedModel(self.cfg).freeze_encoders()
        self.batch["events"] = [{"times": torch.tensor([0., 2.]), "labels": torch.tensor([0, 1])}]*3
        output = model(self.batch)
        model.objective(output, self.batch["labels"], self.batch).backward()
        self.assertGreater(float(model.reasoner.events.weight.grad.abs().sum()), 0)

    def test_all_missing_sensors_produce_finite_output(self):
        for stream in self.batch["streams"].values():
            stream["mask"].zero_()
        for method in METHODS:
            if method == "posthoc_vq":
                continue
            with self.subTest(method=method):
                model = DistributedModel(replace(self.cfg, method=method)).freeze_encoders()
                result = model(self.batch)
                self.assertTrue(torch.isfinite(result["logits"]).all())
                self.assertEqual(payload_bytes(result["messages"], model.names), [0, 0, 0])
                loss = model.objective(result, self.batch["labels"])
                loss.backward()
                self.assertTrue(torch.isfinite(loss))

    def test_no_padding_or_absent_sensor_influence(self):
        model = DistributedModel(self.cfg).freeze_encoders().eval()
        with torch.no_grad():
            first = model(self.batch)["logits"]
            for s in self.batch["streams"].values():
                s["x"][~s["mask"]] = 123456
                s["time"][~s["mask"]] = -123456
            second = model(self.batch)["logits"]
        torch.testing.assert_close(first, second)

    def test_timestamp_interpolation_and_gradients(self):
        r = Resampler(1, 3, learnable=False)
        s = {"x": torch.tensor([[[0.], [1.], [10.]]], requires_grad=True),
             "time": torch.tensor([[0., 1., 10.]]), "mask": torch.ones(1, 3, dtype=torch.bool)}
        output, time, _ = r(s)
        torch.testing.assert_close(output, torch.tensor([[[0.], [5.], [10.]]]))
        output.sum().backward()
        self.assertIsNotNone(s["x"].grad)
        torch.testing.assert_close(time, torch.tensor([[0., 5., 10.]]))

    def test_packet_index_widths_and_rejection(self):
        for vocab in (1, 2, 3, 256, 257, 65537):
            message = Message("imu", torch.tensor([[0, vocab-1]]), torch.tensor([[0., 1.]]),
                              torch.ones(1, 2, dtype=torch.bool), vocab)
            packet = serialize(message, 0, 0)
            parsed = deserialize(packet, ["imu"], vocab)
            torch.testing.assert_close(parsed.values, message.values)
            with self.assertRaises(ValueError):
                deserialize(packet[:-1], ["imu"], vocab)
            with self.assertRaises(ValueError):
                deserialize(b"BAD!"+packet[4:], ["imu"], vocab)

    def test_reserved_bank_rejects_wrong_modality_indices(self):
        model = DistributedModel(self.cfg).eval()
        one = collate([self.dataset[1]], self.cfg.modalities)
        messages, _ = model.peripheral(one["streams"])
        messages["ambient"].values.zero_()
        with self.assertRaisesRegex(ValueError, "reserved"):
            model.central(messages)

    def test_normalization_train_only_and_leakage_guard(self):
        stats = fit_stats(self.dataset)
        val = ManifestDataset(self.manifest, "val", self.cfg.modalities)
        with self.assertRaises(ValueError):
            fit_stats(val)
        rows = [json.loads(s) for s in self.manifest.read_text().splitlines()]
        rows[-1]["participant"] = "p0"
        self.manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "leakage"):
            ManifestDataset(self.manifest, "train", self.cfg.modalities)
        self.assertTrue(all(np.asarray(s["std"]).min() > 0 for s in stats.values()))

    def test_roundtrip_checkpoint_and_pretraining(self):
        model = DistributedModel(self.cfg)
        before = model.encoders["imu"].net[0].weight.detach().clone()
        dl = DataLoader(self.dataset, batch_size=3, collate_fn=partial(collate, modalities=self.cfg.modalities))
        pretrain_encoders(model, dl, 1, .001)
        self.assertFalse(torch.equal(before, model.encoders["imu"].net[0].weight))
        model.freeze_encoders().eval()
        path = self.root/"model.pt"
        save_checkpoint(path, model, fit_stats(self.dataset), {"fixture": True})
        restored, saved = load_checkpoint(path)
        with torch.no_grad():
            torch.testing.assert_close(model(self.batch)["logits"], restored(self.batch)["logits"])
        self.assertTrue(saved["provenance"]["fixture"])

    def test_metrics_fixed_vocabulary_and_grouping(self):
        result = classification([0, 0, 1], [0, 1, 1], 3)
        self.assertAlmostEqual(result["macro_f1"], 4/9)
        self.assertIsNone(participant_bootstrap([0], [0], ["p0"], 3)["macro_f1_95ci"])
        self.assertEqual(answer_metrics("Turn screw", "turn screw."), {"exact_match": 1., "token_f1": 1.})
        self.assertEqual(holm([.01, .04, .03]), [.03, .06, .06])

    def test_separate_bank_capacity_and_unrestricted_selection(self):
        separate = DistributedModel(replace(self.cfg, method="separate_banks"))
        self.assertEqual(sum(v.shape[0] for v in separate.separate_banks.values()), separate.vocabulary)
        unrestricted = DistributedModel(replace(self.cfg, method="unrestricted_bank"))
        self.assertEqual(unrestricted.selectors["imu"].out_features, unrestricted.vocabulary)

    def test_standalone_edge_and_central_exports(self):
        from distrixsense.benchmark import export_deployment
        from distrixsense.deployment import PeripheralRuntime, CentralRuntime
        one = collate([self.dataset[1]], self.cfg.modalities)
        for method in METHODS:
            with self.subTest(method=method):
                model = DistributedModel(replace(self.cfg, method=method)).freeze_encoders().eval()
                if method == "posthoc_vq":
                    dl = DataLoader(self.dataset, batch_size=3, collate_fn=partial(collate, modalities=self.cfg.modalities))
                    fit_posthoc(model, dl, iterations=2)
                export_deployment(model, self.root/method, fit_stats(self.dataset))
                edge, _ = PeripheralRuntime.load(self.root/method/"peripheral.pt")
                central, _ = CentralRuntime.load(self.root/method/"central.pt")
                messages = edge(one["streams"])
                received, _ = roundtrip(messages, edge.names, "cpu")
                with torch.no_grad():
                    torch.testing.assert_close(central(received)["logits"], model(one)["logits"], atol=1e-6, rtol=1e-5)
                if method == "distrixsense":
                    self.assertFalse(hasattr(edge, "bank"))
                    self.assertFalse(hasattr(edge, "reasoner"))
                    self.assertFalse(hasattr(central, "encoders"))
                    with self.assertRaises(RuntimeError):
                        edge.train()


if __name__ == "__main__":
    unittest.main()
