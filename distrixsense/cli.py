import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from .config import Config, METHODS
from .data import ManifestDataset, collate, make_synthetic, to_device
from .training import evaluate, load_checkpoint, loader, train

DEFAULT_METHODS = [m for m in METHODS if m != "imagebind"]


def main(argv=None):
    p = argparse.ArgumentParser(description="DistriXSense controlled sensor-token experiments")
    p.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads, recorded by benchmarks")
    sub = p.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download-opportunity")
    download.add_argument("--output", default="dataset/Opportunity")
    prepare = sub.add_parser("prepare-opportunity")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--task", choices=("gestures", "locomotion", "activity"), default="gestures")
    prepare.add_argument("--window", type=int, default=96)
    prepare.add_argument("--stride", type=int, default=48)
    for name, default in (("train", ["S1", "S2"]), ("val", ["S3"]), ("test", ["S4"])):
        prepare.add_argument(f"--{name}-people", nargs="+", default=default)
    prepare.add_argument("--group-map")
    prepare.add_argument("--exclude-null", action="store_true")
    prepare.add_argument("--max-windows", type=int)
    adapter = sub.add_parser("prepare-records")
    adapter.add_argument("--records", required=True)
    adapter.add_argument("--dataset", choices=("opportunity++", "openmarcie"), required=True)
    adapter.add_argument("--output", required=True)
    adapter.add_argument("--window-seconds", type=float, default=3.)
    adapter.add_argument("--stride-seconds", type=float, default=1.5)
    synthetic = sub.add_parser("synthetic")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--samples", type=int, default=18)
    imagebind = sub.add_parser("extract-imagebind")
    imagebind.add_argument("--manifest", required=True)
    imagebind.add_argument("--weights", required=True)
    imagebind.add_argument("--output", required=True)
    imagebind.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    for command in ("train", "suite"):
        item = sub.add_parser(command)
        item.add_argument("--manifest", required=True)
        item.add_argument("--config", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--device", choices=("cpu", "cuda"))
        item.add_argument("--epochs", type=int)
        item.add_argument("--language-model", help="Existing local Hugging Face checkpoint directory")
        item.add_argument("--language-mode", choices=("sensor", "classifier", "summary"), default="sensor")
        item.add_argument("--class-names", help="JSON list of class names, in encoded label order")
        if command == "train":
            item.add_argument("--method", choices=METHODS)
            item.add_argument("--dense-checkpoint")
            item.add_argument("--seed", type=int)
        else:
            item.add_argument("--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS)
            item.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", default="runs/smoke")
    for command in ("evaluate", "benchmark", "infer", "export", "attack"):
        item = sub.add_parser(command)
        item.add_argument("--checkpoint", required=True)
        item.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
        item.add_argument("--output", required=True)
        if command != "export":
            item.add_argument("--manifest", required=True)
            item.add_argument("--split", choices=("train", "val", "test"), default="test")
        if command == "evaluate":
            item.add_argument("--drop-modalities", nargs="*", default=[])
        if command == "benchmark":
            item.add_argument("--iterations", type=int, default=30)
        if command == "infer":
            item.add_argument("--index", type=int, default=0)
        if command == "attack":
            item.add_argument("--epochs", type=int, default=20)
            item.add_argument("--budget", type=int, default=100000, help="Maximum attacker parameter count, matched across methods")
    args = p.parse_args(argv)
    if args.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "download-opportunity":
        from .prepare import download_opportunity
        print(download_opportunity(args.output))
    elif args.command == "prepare-opportunity":
        from .prepare import prepare_opportunity
        print(prepare_opportunity(args.root, args.output, args.task, args.window, args.stride,
            args.train_people, args.val_people, args.test_people, args.group_map, args.exclude_null, args.max_windows))
    elif args.command == "prepare-records":
        from .prepare import prepare_records
        print(prepare_records(args.records, args.output, args.dataset, args.window_seconds, args.stride_seconds))
    elif args.command == "synthetic":
        print(make_synthetic(args.output, args.samples))
    elif args.command == "extract-imagebind":
        from .imagebind_features import extract
        print(extract(args.manifest, args.weights, args.output, args.device))
    elif args.command in ("train", "suite", "smoke"):
        if args.command == "smoke":
            manifest = make_synthetic(Path(args.output)/"data", samples=3)
            cfg = Config(epochs=1, pretrain_epochs=1, batch_size=3, hidden=16, heads=2, layers=1, resample_length=16, tokens=4)
            methods, seeds = DEFAULT_METHODS, [0]
            language_options = {}
        else:
            manifest, cfg = args.manifest, Config.load(args.config)
            cfg = replace(cfg, **{n: getattr(args, n) for n in ("device", "epochs") if getattr(args, n) is not None})
            language_options = {"language_checkpoint": args.language_model, "language_mode": args.language_mode,
                                "class_names": json.loads(Path(args.class_names).read_text()) if args.class_names else None}
            if args.command == "train":
                cfg = replace(cfg, **{n: getattr(args, n) for n in ("method", "seed") if getattr(args, n) is not None})
                result = train(manifest, cfg, args.output, dense_checkpoint=args.dense_checkpoint, **language_options)
                print(json.dumps({k: result[k] for k in ("method", "macro_f1", "mean_bytes_per_window")}, indent=2))
                return
            methods, seeds = args.methods, args.seeds
        if "imagebind" in methods and len(methods) > 1:
            p.error("Run ImageBind separately on a matched supported-modality feature manifest")
        if "posthoc_vq" in methods and "dense" not in methods:
            p.error("Suite posthoc_vq requires dense in --methods")
        methods = sorted(dict.fromkeys(methods), key=lambda m: (m != "dense", m == "posthoc_vq"))
        results = []
        for seed in dict.fromkeys(seeds):
            for method in methods:
                run_cfg = replace(cfg, seed=seed, method=method)
                root = Path(args.output)/method/f"seed-{seed}"
                dense = Path(args.output)/"dense"/f"seed-{seed}"/"best.pt" if method == "posthoc_vq" else None
                options = dict(language_options)
                if options.get("language_checkpoint") and method in ("local_only", "late_fusion"):
                    options["language_mode"] = "classifier"
                results.append(train(manifest, run_cfg, root, Path(args.output)/"pretraining", dense, **options))
                if args.command == "smoke":
                    from .benchmark import benchmark
                    model, saved = load_checkpoint(root/"best.pt")
                    ds = ManifestDataset(manifest, "test", cfg.modalities, saved["stats"])
                    batch = to_device(collate([ds[0]], cfg.modalities), cfg.device)
                    measured = benchmark(model, batch, iterations=2, warmup=1)
                    if measured["packet_logit_max_difference"] > 1e-5:
                        raise AssertionError(f"Packet deployment mismatch for {method}")
                    (root/"benchmark.json").write_text(json.dumps(measured, indent=2))
        from .report import write_report
        write_report(args.output, results)
        print(f"Results: {Path(args.output)/'summary.md'}")
    else:
        model, saved = load_checkpoint(args.checkpoint, args.device)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "export":
            from .benchmark import export_deployment
            export_deployment(model, args.output, saved["stats"])
            return
        dataset = ManifestDataset(args.manifest, args.split, model.cfg.modalities, saved["stats"])
        if args.command == "evaluate":
            if set(args.drop_modalities)-set(model.names):
                p.error("Unknown dropped modality")
            language = restore_language(saved, model) if saved.get("language_adapter") else None
            metrics, _, _ = evaluate(model, loader(dataset, model.cfg), language, args.drop_modalities)
            output_path.write_text(json.dumps(metrics, indent=2))
        elif args.command == "attack":
            from .attacks import run_attacks
            output_path.write_text(json.dumps(run_attacks(model, saved, args.manifest, args.epochs, args.budget), indent=2))
        else:
            index = args.index if args.command == "infer" else 0
            batch = to_device(collate([dataset[index]], model.cfg.modalities), args.device)
            if args.command == "benchmark":
                from .benchmark import benchmark
                result = benchmark(model, batch, args.iterations)
            else:
                from .transport import roundtrip
                model.eval()
                with torch.no_grad():
                    messages, _ = model.peripheral(batch["streams"])
                    received, packets = roundtrip(messages, model.names, args.device)
                    prediction = model.central(received)
                result = {"prediction": int(prediction["logits"].argmax(-1)),
                          "probabilities": prediction["logits"].softmax(-1).tolist()[0],
                          "bytes": sum(len(x)+model.cfg.protocol_overhead for x in packets),
                          "availability": prediction["availability"].tolist()[0]}
                if saved.get("language_adapter") and batch["queries"][0] is not None:
                    result["answer"] = restore_language(saved, model).answer(prediction, batch)
            output_path.write_text(json.dumps(result, indent=2))
        print(args.output)


def restore_language(saved, model):
    from .language import SensorLanguageModel
    provenance = saved["provenance"]
    lm = SensorLanguageModel.from_local(provenance["language_checkpoint"], model.cfg.hidden, model.cfg.device,
        mode=provenance["language_mode"], class_names=provenance["class_names"])
    lm.adapter.load_state_dict(saved["language_adapter"])
    return lm
