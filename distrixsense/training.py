import hashlib
import json
import platform
import random
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .config import Config
from .data import ManifestDataset, collate, fit_stats, to_device
from .metrics import classification, participant_bootstrap
from .models import DistributedModel
from .transport import payload_bytes


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def loader(dataset, cfg, shuffle=False):
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(cfg.seed),
                      collate_fn=partial(collate, modalities=cfg.modalities))


def fingerprint(manifest, cfg):
    relevant = {k: cfg.to_dict()[k] for k in ("modalities", "classes", "hidden", "resample_length", "tokens", "seed", "pretrain_epochs", "lr", "batch_size", "weight_decay", "heads", "layers", "bank_size", "bank_sizes", "bottleneck_dim", "dropout")}
    implementation = b"".join((Path(__file__).parent/name).read_bytes() for name in ("training.py", "models.py", "temporal_baselines.py", "data.py"))
    h = hashlib.sha256(implementation + Path(manifest).read_bytes() + json.dumps(relevant, sort_keys=True).encode())
    # Include source bytes so changing a window cannot reuse stale normalization/encoders.
    rows = [json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
    for row in sorted(rows, key=lambda r: r["path"]):
        if row["split"] == "train":
            with (Path(manifest).parent / row["path"]).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b""):
                    h.update(chunk)
    return h.hexdigest()


def pretrain_encoders(model, training_loader, epochs, lr):
    """Train-only reconstruction pretraining; resampling is fixed during this stage."""
    params = list(model.encoders.parameters())+list(model.reconstructors.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    model.train()
    for _ in range(epochs):
        for batch in training_loader:
            batch = to_device(batch, model.cfg.device)
            losses = []
            for n, s in batch["streams"].items():
                # Preserve the encoder pretraining target; no learned target drift.
                x, _, available = model.resamplers[n](s)
                x = x.detach()
                if available.any():
                    prediction = model.reconstructors[n](model.encoders[n](x))
                    losses.append(nn.functional.mse_loss(prediction[available], x[available]))
            if losses:
                optimizer.zero_grad()
                torch.stack(losses).mean().backward()
                nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()


def shared_initialization(manifest, cfg, cache_dir):
    key = fingerprint(manifest, cfg)
    cache = Path(cache_dir) / f"pretrain-{key[:16]}.pt"
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != key:
            raise ValueError("Pretraining cache mismatch")
        return saved
    train = ManifestDataset(manifest, "train", cfg.modalities)
    stats = fit_stats(train)
    train.stats = stats
    seed_everything(cfg.seed)
    # Always use one common architecture and initial central weights across methods.
    base_cfg = Config(**{**cfg.to_dict(), "method": "dense"})
    model = DistributedModel(base_cfg).to(cfg.device)
    pretrain_encoders(model, loader(train, cfg, True), cfg.pretrain_epochs, cfg.lr)
    saved = {"fingerprint": key, "stats": stats, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, cache)
    return saved


def initialize(model, state):
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    if model.cfg.method == "separate_banks":
        with torch.no_grad():
            for n in model.names:
                model.separate_banks[n].copy_(model.bank.weight[model.offsets[n]:model.offsets[n]+model.bank_sizes[n]])
    model.freeze_encoders()


@torch.no_grad()
def fit_posthoc(model, training_loader, max_points=20000, iterations=25):
    model.eval()
    points = {n: [] for n in model.names}
    counts = dict.fromkeys(model.names, 0)
    for batch in training_loader:
        batch = to_device(batch, model.cfg.device)
        features, _, present, _ = model.features(batch["streams"])
        for n, h in features.items():
            if counts[n] < max_points:
                p = h[present[n]].flatten(0, 1)[:max_points-counts[n]]
                points[n].append(p)
                counts[n] += len(p)
        if all(c >= max_points for c in counts.values()):
            break
    generator = torch.Generator(device=model.cfg.device).manual_seed(model.cfg.seed)
    for n, chunks in points.items():
        x = torch.cat(chunks)
        if not len(x):
            raise ValueError(f"No training embeddings for {n}")
        count = model.bank_sizes[n]
        # Sampling with replacement handles small smoke datasets deterministically.
        centers = x[torch.randint(len(x), (count,), generator=generator, device=x.device)].clone()
        for _ in range(iterations):
            assignment = torch.cdist(x, centers).argmin(1)
            sums, weights = torch.zeros_like(centers), x.new_zeros(count)
            sums.index_add_(0, assignment, x)
            weights.index_add_(0, assignment, torch.ones(len(x), device=x.device))
            occupied = weights > 0
            centers[occupied] = sums[occupied] / weights[occupied, None]
        start = model.offsets[n]
        model.bank.weight[start:start+count].copy_(centers)
    model.posthoc_ready.fill_(True)
    model.requires_grad_(False)  # pure post-hoc control, including the inherited dense reasoner


@torch.no_grad()
def evaluate(model, data_loader, language=None, drop_modalities=(), bootstrap=True):
    model.eval()
    labels, predictions, participants, bytes_per_window = [], [], [], []
    records, code_counts, language_records = [], {}, []
    for batch in data_loader:
        batch = to_device(batch, model.cfg.device)
        for n in drop_modalities:
            batch["streams"][n]["mask"].zero_()
        output = model(batch)
        p = output["logits"].argmax(-1).cpu().tolist()
        y = batch["labels"].cpu().tolist()
        labels.extend(y)
        predictions.extend(p)
        participants.extend(batch["participants"])
        sizes = payload_bytes(output["messages"], model.names, model.cfg.protocol_overhead)
        bytes_per_window.extend(sizes)
        for truth, pred, person, size in zip(y, p, batch["participants"], sizes):
            records.append({"label": truth, "prediction": pred, "participant": person, "bytes": size})
        for n, (indices, mask) in output["auxiliary"].get("assignments", {}).items():
            count = len(model.region(n))
            counts = torch.bincount(indices[mask].cpu(), minlength=count).numpy()
            code_counts[n] = code_counts.get(n, np.zeros(count)) + counts
        if language:
            from .metrics import answer_metrics
            for i, (query, reference) in enumerate(zip(batch["queries"], batch["answers"])):
                if query is not None and reference is not None:
                    answer = language.answer(output, batch, i)
                    language_records.append({"participant": batch["participants"][i], "query": query,
                                             "reference": reference, "prediction": answer,
                                             **answer_metrics(reference, answer)})
    result = classification(labels, predictions, model.cfg.classes)
    if bootstrap:
        result.update(participant_bootstrap(labels, predictions, participants, model.cfg.classes, model.cfg.seed))
    result["mean_bytes_per_window"] = float(np.mean(bytes_per_window))
    result["code_usage"] = {}
    for n, counts in code_counts.items():
        probs = counts/counts.sum() if counts.sum() else counts
        result["code_usage"][n] = {"used": int((counts > 0).sum()), "entries": len(counts),
                                  "perplexity": float(np.exp(-(probs[probs > 0]*np.log(probs[probs > 0])).sum()))}
    if language_records:
        result["language"] = {k: float(np.mean([r[k] for r in language_records])) for k in ("exact_match", "token_f1")}
        result["language"]["examples"] = len(language_records)
    return result, records, language_records


def save_checkpoint(path, model, stats, provenance, language=None):
    saved = {"config": model.cfg.to_dict(), "state": model.state_dict(), "stats": stats, "provenance": provenance}
    if language:
        # Store trainable adapter only; the local LM checkpoint is referenced in provenance.
        saved["language_adapter"] = language.adapter.state_dict()
    torch.save(saved, path)


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    cfg = Config(**{**saved["config"], "device": device}).validate()
    model = DistributedModel(cfg).to(device)
    model.load_state_dict(saved["state"])
    model.freeze_encoders().eval()
    return model, saved


def train(manifest, cfg, output, cache_dir=None, dense_checkpoint=None, language_checkpoint=None,
          language_mode="sensor", class_names=None, language_dtype="float32", language_prompt_style="auto"):
    cfg.validate()
    seed_everything(cfg.seed)
    root = Path(output)
    if (root / "best.pt").exists():
        raise ValueError(f"Run already exists: {root}; choose a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    shared = shared_initialization(manifest, cfg, cache_dir or root.parent / "pretraining")
    seed_everything(cfg.seed)
    model = DistributedModel(cfg).to(cfg.device)
    initialize(model, shared["state"])
    datasets = {s: ManifestDataset(manifest, s, cfg.modalities, shared["stats"]) for s in ("train", "val", "test")}
    for s, dataset in datasets.items():
        if any(r["label"] < 0 or r["label"] >= cfg.classes for r in dataset.rows):
            raise ValueError(f"Labels in {s} must be within configured classes")
    loaders = {s: loader(ds, cfg, s == "train") for s, ds in datasets.items()}
    provenance = {"manifest": str(Path(manifest).resolve()), "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                  "pretrain_fingerprint": shared["fingerprint"], "python": platform.python_version(),
                  "torch": str(torch.__version__), "numpy": str(np.__version__),
                  "language_checkpoint": str(Path(language_checkpoint).resolve()) if language_checkpoint else None,
                  "language_dtype": language_dtype, "language_prompt_style": language_prompt_style,
                  "language_mode": language_mode, "class_names": class_names,
                  "split_windows": {s: len(ds) for s, ds in datasets.items()}}
    (root / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (root / "norm_stats.json").write_text(json.dumps(shared["stats"], indent=2))
    language = None
    if language_checkpoint:
        if not any(r.get("query") is not None and r.get("answer") is not None for r in datasets["train"].rows):
            raise ValueError("Language training requires real query/answer annotations in training data")
        if cfg.method in ("local_only", "late_fusion") and language_mode == "sensor":
            raise ValueError("Use classifier language mode for local prediction baselines")
        from .language import SensorLanguageModel
        language = SensorLanguageModel.from_local(language_checkpoint, cfg.hidden, cfg.device,
                                                   mode=language_mode, class_names=class_names,
                                                   dtype=language_dtype, prompt_style=language_prompt_style)
    if cfg.method == "posthoc_vq":
        if not dense_checkpoint:
            raise ValueError("posthoc_vq requires --dense-checkpoint from the matched dense run")
        dense, dense_saved = load_checkpoint(dense_checkpoint, cfg.device)
        matched = ("modalities", "classes", "hidden", "resample_length", "heads", "layers", "seed", "dropout",
                   "batch_size", "lr", "epochs", "patience", "pretrain_epochs")
        if dense.cfg.method != "dense" or any(getattr(dense.cfg, k) != getattr(cfg, k) for k in matched):
            raise ValueError("Post-hoc baseline must match the dense model configuration and seed")
        if dense_saved["provenance"]["pretrain_fingerprint"] != shared["fingerprint"]:
            raise ValueError("Dense checkpoint was trained on different data")
        initialize(model, dense_saved["state"])
        fit_posthoc(model, loaders["train"])
        provenance["dense_checkpoint"] = str(Path(dense_checkpoint).resolve())
        if language:
            if "language_adapter" not in dense_saved or dense_saved["provenance"]["language_checkpoint"] != provenance["language_checkpoint"]:
                raise ValueError("Post-hoc language mode requires the matching dense language adapter")
            if any(dense_saved["provenance"].get(k, default) != provenance[k] for k, default in
                   (("language_dtype", "float32"), ("language_prompt_style", "auto"), ("language_mode", "sensor"))):
                raise ValueError("Post-hoc language dtype and prompt mode must match the dense run")
            language.adapter.load_state_dict(dense_saved["language_adapter"])
        save_checkpoint(root / "best.pt", model, shared["stats"], provenance, language)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        if language and language_mode == "sensor":
            params += list(language.adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
        best, stale, history = -1., 0, []
        for epoch in range(cfg.epochs):
            model.train()
            if language:
                language.train()
            losses = []
            for batch in loaders["train"]:
                batch = to_device(batch, cfg.device)
                optimizer.zero_grad()
                output_batch = model(batch)
                loss = model.objective(output_batch, batch["labels"], batch)
                if language:
                    loss = loss + cfg.language_weight*language.loss(output_batch, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            val, _, _ = evaluate(model, loaders["val"], bootstrap=False)
            history.append({"epoch": epoch+1, "loss": float(np.mean(losses)), "val_macro_f1": val["macro_f1"]})
            print(f"{cfg.method} seed={cfg.seed} epoch={epoch+1} loss={np.mean(losses):.4f} val_f1={val['macro_f1']:.4f}", flush=True)
            if val["macro_f1"] > best:
                best, stale = val["macro_f1"], 0
                save_checkpoint(root / "best.pt", model, shared["stats"], provenance, language)
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
        (root / "history.json").write_text(json.dumps(history, indent=2))
    model, saved = load_checkpoint(root / "best.pt", cfg.device)
    if language:
        language.adapter.load_state_dict(saved["language_adapter"])
    results, records, answers = evaluate(model, loaders["test"], language)
    results.update(method=cfg.method, seed=cfg.seed, provenance=provenance)
    (root / "metrics.json").write_text(json.dumps(results, indent=2))
    (root / "predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    if answers:
        (root / "answers.jsonl").write_text("".join(json.dumps(r)+"\n" for r in answers))
    return results
