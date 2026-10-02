import csv
import json
from pathlib import Path
import numpy as np
from .metrics import classification, holm


def write_report(output, results):
    root = Path(output)
    grouped = {}
    for result in results:
        grouped.setdefault(result["method"], []).append(result)
    rows = []
    for method, runs in grouped.items():
        values = [r["macro_f1"] for r in runs]
        rows.append({"method": method, "seeds": len(runs), "macro_f1_mean": float(np.mean(values)),
                     "macro_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                     "accuracy_mean": float(np.mean([r["accuracy"] for r in runs])),
                     "bytes_per_window_mean": float(np.mean([r["mean_bytes_per_window"] for r in runs]))})
    with (root/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = []
    for baseline in grouped:
        if baseline == "distrixsense" or "distrixsense" not in grouped:
            continue
        per_person = {}
        for result in grouped["distrixsense"]:
            seed = result["seed"]
            main_path, base_path = root/f"distrixsense/seed-{seed}/predictions.jsonl", root/f"{baseline}/seed-{seed}/predictions.jsonl"
            if not base_path.exists():
                continue
            a = [json.loads(s) for s in main_path.read_text().splitlines()]
            b = [json.loads(s) for s in base_path.read_text().splitlines()]
            if len(a) != len(b) or any((x["participant"], x["label"]) != (y["participant"], y["label"]) for x, y in zip(a, b)):
                raise ValueError("Paired report requires identical test observation ordering")
            classes = len(result["per_class"])
            for person in sorted(set(r["participant"] for r in a)):
                indices = [i for i, r in enumerate(a) if r["participant"] == person]
                labels = [a[i]["label"] for i in indices]
                delta = classification(labels, [a[i]["prediction"] for i in indices], classes)["macro_f1"] - classification(labels, [b[i]["prediction"] for i in indices], classes)["macro_f1"]
                per_person.setdefault(person, []).append(delta)
        deltas = np.asarray([np.mean(v) for v in per_person.values()])
        if len(deltas) > 1:
            rng = np.random.default_rng(0)
            simulated = (rng.choice([-1, 1], (10000, len(deltas)))*deltas).mean(1)
            p = float((1+(np.abs(simulated) >= abs(deltas.mean())-1e-12).sum())/10001)
        else:
            p = None
        comparisons.append({"baseline": baseline, "participants": len(deltas), "p_value": p,
                            "mean_participant_macro_f1_delta": float(deltas.mean()) if len(deltas) else None})
    tested = [r for r in comparisons if r["p_value"] is not None]
    for r, p in zip(tested, holm([r["p_value"] for r in tested])):
        r["holm_p_value"] = p
    (root/"comparisons.json").write_text(json.dumps({"unit": "participant; seed differences averaged within participant",
        "test": "two-sided paired sign permutation, 10000 draws", "comparisons": comparisons}, indent=2))
    table = ["| Method | Seeds | Macro F1 (mean ± SD) | Bytes/window |", "|---|---:|---:|---:|"]
    for row in rows:
        sd = f" ± {row['macro_f1_std']:.4f}" if row["macro_f1_std"] is not None else ""
        table.append(f"| {row['method']} | {row['seeds']} | {row['macro_f1_mean']:.4f}{sd} | {row['bytes_per_window_mean']:.1f} |")
    (root/"summary.md").write_text("\n".join(table)+"\n\nSee comparisons.json for participant-level paired tests. Synthetic data is a software test, not evidence of method superiority.\n")
