import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch
from distrixsense.config import Config
from distrixsense.data import ManifestDataset, collate, make_synthetic
from distrixsense.models import DistributedModel

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def tiny_local_lm(path, width=16):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "fixture": 3, "answer": 4,
             "What": 5, "happened": 6, "?": 7, "class": 8, "0": 9, "1": 10, "2": 11}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.save_pretrained(path)
    lm = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_embd=width, n_layer=1, n_head=2,
                                   n_positions=256, bos_token_id=2, eos_token_id=2, pad_token_id=1))
    lm.save_pretrained(path)


@unittest.skipUnless(HAS_TRANSFORMERS, "Optional language dependencies not installed")
class LanguageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lm_path = self.root/"lm"
        tiny_local_lm(self.lm_path)
        self.manifest = make_synthetic(self.root/"data", samples=3)
        rows = [json.loads(s) for s in self.manifest.read_text().splitlines()]
        for row in rows:
            row.update(query="What happened?", answer=f"fixture class {row['label']}")
        self.manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
        self.cfg = Config(hidden=16, heads=2, layers=1, resample_length=8, tokens=2, epochs=1,
                          pretrain_epochs=1, batch_size=3, modality_dropout=0, dropout=0)
        dataset = ManifestDataset(self.manifest, "train", self.cfg.modalities)
        self.batch = collate([dataset[1]], self.cfg.modalities)

    def test_local_lm_loss_mask_gradient_and_generation(self):
        from distrixsense.language import SensorLanguageModel
        sensor = DistributedModel(self.cfg).freeze_encoders()
        output = sensor(self.batch)
        language = SensorLanguageModel.from_local(self.lm_path, self.cfg.hidden)
        embeddings, labels = language.embeddings(output, self.batch, 0, self.batch["answers"][0])
        self.assertEqual(int((labels != -100).sum()), 4)  # fixture, class, number, EOS
        self.assertEqual(labels.shape, embeddings.shape[:2])
        self.assertNotIn("fixture", language.prompt(output, self.batch, 0))
        loss = language.loss(output, self.batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in language.lm.parameters()))
        self.assertGreater(float(language.adapter[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(sensor.selectors["imu"].weight.grad.abs().sum()), 0)
        self.assertIsInstance(language.answer(output, self.batch, max_new_tokens=2), str)

    def test_classifier_and_summary_receive_no_true_answer(self):
        from distrixsense.language import SensorLanguageModel
        output = DistributedModel(self.cfg).eval()(self.batch)
        for mode in ("classifier", "summary"):
            language = SensorLanguageModel.from_local(self.lm_path, self.cfg.hidden, mode=mode)
            prompt = language.prompt(output, self.batch, 0)
            self.assertNotIn("fixture class", prompt)
            self.assertIn("Predicted window activity" if mode == "classifier" else "normalized mean", prompt)

    def test_full_language_training_and_checkpoint_restore(self):
        from distrixsense.training import load_checkpoint, train
        from distrixsense.cli import restore_language
        result = train(self.manifest, self.cfg, self.root/"run", language_checkpoint=self.lm_path)
        self.assertEqual(result["language"]["examples"], 9)
        model, saved = load_checkpoint(self.root/"run/best.pt")
        restored = restore_language(saved, model)
        self.assertTrue(all(not p.requires_grad for p in restored.lm.parameters()))
        self.assertTrue((self.root/"run/answers.jsonl").exists())

    def test_language_deployment_cost_profile(self):
        from distrixsense.profiling import profile_language
        report = profile_language(DistributedModel(self.cfg).eval(), self.batch,
                                  {"checkpoint": "lm", "decode_tokens": 2}, self.root, "cpu", 1, 0)
        self.assertEqual(report["decode_tokens"], 2)
        self.assertGreater(report["storage"]["parameters"], 0)
        self.assertGreater(report["prefill_arithmetic"]["counted_flops"], 0)
        self.assertGreater(report["generation_arithmetic"]["counted_flops"], report["prefill_arithmetic"]["counted_flops"])

    def test_bfloat16_core_preserves_float32_adapter_and_gradients(self):
        from distrixsense.language import SensorLanguageModel
        language = SensorLanguageModel.from_local(self.lm_path, self.cfg.hidden, dtype="bfloat16")
        self.assertEqual(language.lm.get_input_embeddings().weight.dtype, torch.bfloat16)
        self.assertEqual(language.adapter[0].weight.dtype, torch.float32)
        model = DistributedModel(self.cfg)
        loss = language.loss(model(self.batch), self.batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(language.adapter[0].weight.grad.abs().sum()), 0)

    def test_native_chat_template_uses_the_same_sensor_prefix(self):
        from distrixsense.language import SensorLanguageModel
        language = SensorLanguageModel.from_local(self.lm_path, self.cfg.hidden)
        language.tokenizer.chat_template = "{% for message in messages %}{{ 'What ' + message['content'] }}{% endfor %}{{ ' answer' if add_generation_prompt else '' }}"
        output = DistributedModel(self.cfg).eval()(self.batch)
        auto, _ = language.embeddings(output, self.batch, 0)
        language.prompt_style = "plain"
        plain, _ = language.embeddings(output, self.batch, 0)
        self.assertGreater(auto.shape[1], plain.shape[1])
        count = int(output["sensor_mask"].sum())
        torch.testing.assert_close(auto[:, :count], plain[:, :count])


if __name__ == "__main__":
    unittest.main()
