"""Predefined group_map helpers for Opportunity dataset.

This module provides `official_opportunity_group_map()` which attempts to load
an official mapping file at `dataset/Opportunity/group_map_official.json`.
If that file is missing, it returns a conservative default grouping that
assumes triaxial sensors arranged sequentially (groups of 3 columns).

To use the true official column ordering, create `dataset/Opportunity/group_map_official.json`
with a JSON object mapping group names to lists of 0-based column indices.

Example `group_map_official.json`:
{
  "torso_acc": [0,1,2],
  "left_ankle_acc": [3,4,5],
  ...
}
"""
from pathlib import Path
import json
from typing import Dict, List, Optional


def official_opportunity_group_map(root: str = 'dataset/Opportunity', preferred_axis: int = 3) -> Dict[str, List[int]]:
    """Return the official group_map if present, else a default per-axis grouping.

    The function looks for `dataset/Opportunity/group_map_official.json` and
    returns its contents if found. If not found, it inspects the first data
    file to determine column count and groups columns into `preferred_axis`
    sized groups (e.g., 3 for triaxial sensors).
    """
    p = Path(root)
    json_path = p / 'group_map_official.json'
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            # ensure keys map to lists of ints
            return {k: [int(i) for i in v] for k, v in data.items()}
        except Exception:
            pass

    # fallback: infer by counting columns in first file
    files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in ('.dat', '.csv')]) if p.exists() else []
    if not files:
        return {}
    first = files[0]
    n_cols = None
    with open(first, 'r', errors='ignore') as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.strip().split()
            n_cols = len(parts)
            break
    if n_cols is None or n_cols <= 0:
        return {}

    num_features = n_cols
    # default grouping: every `preferred_axis` columns
    if num_features % preferred_axis == 0:
        groups = {}
        n_groups = num_features // preferred_axis
        for i in range(n_groups):
            start = i * preferred_axis
            groups[f'sensor_{i}'] = list(range(start, start + preferred_axis))
        return groups

    # fallback single-column groups
    return {f'sensor_{i}': [i] for i in range(num_features)}


def save_example_official_json(path: Optional[str] = None) -> None:
    """Write a small example `group_map_official.json` into dataset folder for editing.
    """
    root = Path(path) if path else Path('dataset/Opportunity')
    root.mkdir(parents=True, exist_ok=True)
    example = {
        "torso_acc": [0, 1, 2],
        "left_ankle_acc": [3, 4, 5],
        "right_ankle_acc": [6, 7, 8]
    }
    with open(root / 'group_map_official.json', 'w', encoding='utf-8') as fh:
        json.dump(example, fh, indent=2)
