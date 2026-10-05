"""Matched sensing-modality and local-LLM experiments and deployment profiles."""
import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
import re
from .config import Config, METHODS
from .models import LOCAL
from .training import train


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_+-]+", value):
        raise ValueError("Scenario/core names must contain only letters, digits, underscore, plus or hyphen")
    return value


def select_config(config, streams):
    result = copy.deepcopy(config)
    result["modalities"] = {n: config["modalities"][n] for n in streams}
    result["bank_sizes"] = {n: v for n, v in config.get("bank_sizes", {}).items() if n in streams}
    Config(**result).validate()
    return result


def select_topology(topology, streams):
    result = copy.deepcopy(topology)
    mapping = {n: topology.get("stream_devices", {}).get(n, n) for n in streams}
    result["stream_devices"] = mapping
    result["edges"] = {n: v for n, v in topology.get("edges", {}).items() if n in set(mapping.values())}
    return result


def expand_matrix(spec_path):
    path = Path(spec_path).resolve()
    source = json.loads(path.read_text())
    cores = source.get("cores", [{"name": "sensor_only"}])
    core_names = [identifier(c["name"]) for c in cores]
    if len(set(core_names)) != len(core_names):
        raise ValueError("Core names must be unique")
    scenarios, skipped, identifiers = [], [], set()
    for entry in source["datasets"]:
        channels = entry["config"]["modalities"]
        groups = copy.deepcopy(entry.get("modality_sets", {"all": list(channels)}))
        if entry.get("include_each_modality", False):
            groups.update({f"only_{n}": [n] for n in channels})
        if entry.get("include_leave_one_out", False) and len(channels) > 1:
            groups.update({f"without_{n}": [k for k in channels if k != n] for n in channels})
        for name, streams in groups.items():
            identifier(name)
            if streams == "all":
                streams = list(channels)
            if not streams or len(streams) != len(set(streams)) or set(streams)-set(channels):
                raise ValueError(f"Invalid modality set {name}: {streams}")
            for core in cores:
                scenario = copy.deepcopy(entry)
                scenario_id = identifier(f"{entry['dataset']}__{name}__{core['name']}")
                if scenario_id in identifiers:
                    raise ValueError(f"Duplicate scenario: {scenario_id}")
                identifiers.add(scenario_id)
                scenario.update(scenario_id=scenario_id, core_name=core["name"], modality_set=name)
                scenario["config"] = select_config(entry["config"], streams)
                scenario["topology"] = select_topology(entry.get("topology", {}), streams)
                if "inputs" in entry:
                    scenario["inputs"] = {n: entry["inputs"][n] for n in streams}
                for key in ("manifest",):
                    if entry.get(key):
                        scenario[key] = str(path.parent/entry[key])
                # Checkpoints must be explicitly supplied for this core/modality scenario.
                scenario["checkpoints"] = {m: str(path.parent/p) for m, p in entry.get("scenario_checkpoints", {}).get(scenario_id, {}).items()}
                if core.get("checkpoint"):
                    scenario["language"] = {k: v for k, v in core.items() if k != "name"}
                    scenario["language"]["checkpoint"] = str(path.parent/core["checkpoint"])
                else:
                    scenario.pop("language", None)
                profiles = {}
                excluded = set()
                for method, profile in entry.get("method_profiles", {}).items():
                    shared = [n for n in streams if n in profile.get("config", entry["config"])["modalities"]]
                    if not shared:
                        excluded.add(method)
                        skipped.append({"scenario_id": scenario_id, "method": method,
                                        "reason": "No supported sensing modality in this method's input subset"})
                        continue
                    profiles[method] = copy.deepcopy(profile)
                    profiles[method]["config"] = select_config(profile.get("config", entry["config"]), shared)
                    if "inputs" in profile:
                        profiles[method]["inputs"] = {n: profile["inputs"][n] for n in shared}
                    profiles[method]["topology"] = select_topology(profile.get("topology", entry.get("topology", {})), shared)
                scenario["method_profiles"] = profiles
                scenario["excluded_methods"] = sorted(excluded)
                scenarios.append(scenario)
    return {"datasets": scenarios, "unsupported_combinations": skipped,
            "seeds": source.get("seeds", [0, 1, 2]), "methods": source.get("methods", list(METHODS))}


def run_matrix(spec_path, output, mode="profile", targets=("cpu", "cuda", "mixed"),
               methods=None, iterations=20, warmup=5, dry_run=False):
    plan = expand_matrix(spec_path)
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty matrix output directory")
    root.mkdir(parents=True, exist_ok=True)
    (root/"matrix_plan.json").write_text(json.dumps(plan, indent=2))
    requested = list(methods or plan["methods"])
    if set(requested)-set(METHODS):
        raise ValueError("Unknown matrix method")
    if dry_run:
        return root/"matrix_plan.json"
    if mode == "profile":
        from .profiling import run_suite
        return run_suite(root/"matrix_plan.json", root/"profiles", targets, requested, iterations, warmup)
    if mode != "train":
        raise ValueError("Matrix mode must be profile or train")
    if "posthoc_vq" in requested and "dense" not in requested:
        raise ValueError("posthoc_vq matrix training requires dense in methods")
    # Validate all dependent inputs before any training starts.
    for entry in plan["datasets"]:
        if not entry.get("manifest") or not Path(entry["manifest"]).is_file():
            raise ValueError("Training matrix requires a local prepared manifest for every dataset")
        if entry.get("language") and not Path(entry["language"]["checkpoint"]).is_dir():
            raise ValueError(f"Local LLM checkpoint is missing: {entry['language']['checkpoint']}")
    from .report import write_report
    rows = []
    for entry in plan["datasets"]:
        cfg = Config(**entry["config"]).validate()
        allowed = [m for m in requested if m not in entry["excluded_methods"]]
        scenario_results = []
        if "imagebind" in allowed and "imagebind" in entry.get("method_profiles", {}):
            raise ValueError("ImageBind training needs a matched pre-extracted feature manifest in a separate matrix")
        for seed in plan["seeds"]:
            results = []
            for method in sorted(dict.fromkeys(allowed), key=lambda m: (m != "dense", m == "posthoc_vq")):
                model_cfg = replace(cfg, seed=seed, method=method)
                directory = root/entry["scenario_id"]/method/f"seed-{seed}"
                language = entry.get("language", {})
                language_mode = language.get("mode", "sensor")
                if method in LOCAL and language_mode == "sensor":
                    language_mode = "classifier"
                dense = root/entry["scenario_id"]/"dense"/f"seed-{seed}"/"best.pt" if method == "posthoc_vq" else None
                result = train(entry["manifest"], model_cfg, directory, root/"pretraining", dense,
                               language_checkpoint=language.get("checkpoint"), language_mode=language_mode,
                               language_dtype=language.get("dtype", "float32"),
                               language_prompt_style=language.get("prompt_style", "auto"))
                results.append(result)
                scenario_results.append(result)
                row = {"dataset": entry["dataset"], "modality_set": entry["modality_set"],
                       "core": entry["core_name"], "method": method, "seed": seed,
                       "macro_f1": result["macro_f1"], "mean_bytes_per_window": result["mean_bytes_per_window"],
                       "language": result.get("language"), "checkpoint": str(directory/"best.pt")}
                rows.append(row)
        if scenario_results:
            write_report(root/entry["scenario_id"], scenario_results)
    (root/"matrix_results.json").write_text(json.dumps(rows, indent=2))
    fields = ["dataset", "modality_set", "core", "method", "seed", "macro_f1", "mean_bytes_per_window", "language", "checkpoint"]
    with (root/"matrix_results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "language": json.dumps(row["language"])})
    return root/"matrix_results.csv"
