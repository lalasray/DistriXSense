"""Optional frozen local causal LM with an answer-supervised sensor prefix.

No implicit downloads; pass a local Hugging Face checkpoint. Classifier-to-LM
and raw statistical-summary modes are separate controls using predicted labels
or measured signals. Ground-truth answers are used only as training targets.
"""
import torch
from torch import nn

SYSTEM = "Interpret the supplied sensor evidence. Answer the query concisely. If the evidence is insufficient, say 'Insufficient evidence'."


class SensorLanguageModel(nn.Module):
    def __init__(self, lm, tokenizer, sensor_dim, mode="sensor", class_names=None, prompt_style="auto"):
        super().__init__()
        if mode not in ("sensor", "classifier", "summary"):
            raise ValueError("Invalid language control")
        self.lm, self.tokenizer, self.mode = lm, tokenizer, mode
        self.class_names = class_names
        if prompt_style not in ("auto", "plain", "chat"):
            raise ValueError("prompt_style must be auto, plain or chat")
        self.prompt_style = prompt_style
        self.lm.requires_grad_(False)
        self.lm.eval()
        width = lm.get_input_embeddings().weight.shape[1]
        self.adapter = nn.Sequential(nn.Linear(sensor_dim, width), nn.GELU(), nn.Linear(width, width))
        if tokenizer.eos_token_id is None:
            raise ValueError("The local LM tokenizer must define an EOS token")
        if class_names is not None and (not isinstance(class_names, list) or not all(isinstance(s, str) for s in class_names)):
            raise ValueError("Class names must be a JSON list of strings")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    @classmethod
    def from_local(cls, checkpoint, sensor_dim, device="cpu", dtype="float32", **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the language extra: pip install -e '.[language]'") from exc
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
        if dtype not in dtypes:
            raise ValueError("LM dtype must be float32, float16, bfloat16 or auto")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        lm = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, torch_dtype=dtypes[dtype])
        return cls(lm, tokenizer, sensor_dim, **kwargs).to(device)

    def train(self, mode=True):
        super().train(mode)
        self.lm.eval()
        return self

    def prompt(self, output, batch, i):
        availability = output["availability"][i].detach().cpu().tolist()
        evidence = f"Modality availability: {availability}."
        if self.mode == "classifier":
            p = output["logits"][i].detach().softmax(-1)
            label = int(p.argmax())
            name = self.class_names[label] if self.class_names else str(label)
            evidence += f" Predicted window activity: {name}; confidence: {float(p.max()):.3f}."
        elif self.mode == "summary":
            for n, s in batch["streams"].items():
                x = s["x"][i, s["mask"][i]]
                if len(x):
                    evidence += f" {n}: normalized mean {x.mean(0).detach().cpu().tolist()}, std {x.std(0, unbiased=False).detach().cpu().tolist()}, duration {float(s['time'][i, s['mask'][i]][-1]-s['time'][i, s['mask'][i]][0]):.3f}s."
        return f"{SYSTEM}\nEvidence: {evidence}\nQuery: {batch['queries'][i]}\nAnswer:"

    def embeddings(self, output, batch, i, answer=None):
        device = self.adapter[0].weight.device
        text = self.prompt(output, batch, i)
        chat = self.prompt_style == "chat" or (self.prompt_style == "auto" and bool(self.tokenizer.chat_template))
        if chat:
            if not self.tokenizer.chat_template:
                raise ValueError("Chat prompting requires a local tokenizer chat template")
            prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=True, return_tensors="pt", enable_thinking=False).to(device)
        else:
            prompt = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
        prompt_emb = self.lm.get_input_embeddings()(prompt)
        if self.mode == "sensor":
            prefix = self.adapter(output["sensor_tokens"][i, output["sensor_mask"][i]].to(self.adapter[0].weight.dtype))[None].to(prompt_emb.dtype)
        else:
            prefix = prompt_emb[:, :0]
        inputs = torch.cat([prefix, prompt_emb], 1)
        labels = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=device)
        if answer is not None:
            target = self.tokenizer(" "+answer+self.tokenizer.eos_token, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            inputs = torch.cat([inputs, self.lm.get_input_embeddings()(target)], 1)
            labels = torch.cat([labels, target], 1)
        limit = getattr(self.lm.config, "max_position_embeddings", None)
        if limit and inputs.shape[1] > limit:
            raise ValueError("Sensor prefix and prompt exceed LM context; reduce token budget")
        return inputs, labels

    def loss(self, output, batch):
        losses = []
        for i, (query, answer) in enumerate(zip(batch["queries"], batch["answers"])):
            if query is None or answer is None:
                continue
            inputs, labels = self.embeddings(output, batch, i, answer)
            losses.append(self.lm(inputs_embeds=inputs, attention_mask=torch.ones_like(labels), labels=labels).loss)
        return torch.stack(losses).mean() if losses else self.adapter[0].weight.sum()*0

    @torch.no_grad()
    def answer(self, output, batch, i=0, max_new_tokens=64):
        self.eval()
        inputs, labels = self.embeddings(output, batch, i)
        limit = getattr(self.lm.config, "max_position_embeddings", None)
        if limit:
            max_new_tokens = min(max_new_tokens, limit-inputs.shape[1])
        if max_new_tokens < 1:
            raise ValueError("No room for an answer in the LM context")
        ids = self.lm.generate(inputs_embeds=inputs, attention_mask=torch.ones_like(labels),
                               max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=self.tokenizer.pad_token_id,
                               eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
