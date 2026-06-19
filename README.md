# DistriXSense

This repository provides tooling to work with multimodal sensor datasets.
Primary focus: the Opportunity Activity Recognition dataset and a GPU-capable
PyTorch loader.

**Key files**
- `dataloaders/opportunity_pytorch_dataset.py` — PyTorch `Dataset` with on-the-fly
  sliding-window support and optional time-based windows (`window_ms`).
- `dataset/Opportunity/group_map_official.json` — official column → sensor group mapping.
- `scripts/prepare_opportunity_official_map.py` — build `group_map_official.json` from the original archive.
- `scripts/download_opportunity.py` — best-effort downloader for Opportunity (may require manual download).
- `scripts/setup_venv.ps1` — Windows PowerShell helper to create a venv and install base requirements.
- `scripts/run_opportunity_sanity_one.py` — small sanity script that loads one file and prints a sample batch.

## Quick setup

1. Create & activate venv (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_venv.ps1 -EnvName .venv
. .venv\Scripts\Activate.ps1
```

2. Install requirements inside the venv:

```powershell
pip install -r requirements.txt
# Optionally install CUDA-enabled PyTorch from the official index
```

## Prepare Opportunity dataset

1. Place the original Opportunity ZIP under `dataset/Opportunity/` and extract
   so the `.dat` files are under
   `dataset/Opportunity/extracted/OpportunityUCIDataset/dataset/`.
2. Generate the official group map (if needed):

```powershell
python scripts/prepare_opportunity_official_map.py
```

## Sanity check (small batch)

```powershell
python scripts/run_opportunity_sanity_one.py
```

## Loader notes

- Default: on-the-fly sliding windows (`seq_len`, `step`).
- Time-based windows: provide `window_ms` (milliseconds); the loader uses the
  `MILLISEC` column to infer the required `seq_len`.
- To extract labels from the dataset, pass `label_col` (0-based) to
  `OpportunityDataset`.

Example Python:

```python
from dataloaders.opportunity_pytorch_dataset import OpportunityDataset

ds = OpportunityDataset(
    root='dataset/Opportunity/extracted/OpportunityUCIDataset/dataset',
    window_ms=500,  # 0.5s windows
    step=15,
    group_map=None,
)
print(len(ds))
```

## Notes

This repository contains helpers and dataset scaffolding. See original dataset
licenses for redistribution and usage restrictions.
