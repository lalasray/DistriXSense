# DistriXSense

Datasets
--------

This project recommends placing downloaded datasets under the `dataset/` subfolder.

Multimodal datasets we reference in this repository:

- MM‑Fi (MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset): main archive hosted on Google Drive (~77 GB). Project page and repo:
  - https://ntu-aiot-lab.github.io/mm-fi
  - https://github.com/ybhbingo/MMFi_dataset
  - Google Drive folder: https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz?usp=sharing

Automated helper
----------------

Helper scripts are provided to help fetch or prepare multimodal datasets:

- `scripts/download_opportunity.py` — run with Python 3 to attempt downloading the Opportunity dataset into `dataset/Opportunity/` (may require manual download depending on host).
- For MM‑Fi the project provides its own toolbox; automatic download may be attempted with `gdown` but manual download is common due to size.

Recommended multimodal alternatives
-----------------------------------

If you need multimodal wearable / sensor datasets (accelerometers, gyros, magnetometers, body sensors) consider the Opportunity Activity Recognition dataset (widely used for multimodal activity recognition) and the MM‑Fi dataset (wireless multimodal). Links:

- Opportunity: http://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition or http://www.opportunity-project.eu/dataset.html
- MM‑Fi toolbox and data: https://github.com/ybhbingo/MMFi_dataset and https://ntu-aiot-lab.github.io/mm-fi

This repository includes a small scaffold to help integrate Opportunity: see `scripts/download_opportunity.py` and `dataloaders/opportunity_pytorch_dataset.py`.

Future work
-----------

- Integrate MM‑Fi toolbox and provide an MM‑Fi dataloader/wrapper (large dataset; planned).

Quick commands
--------------

Install recommended helpers and run the Opportunity downloader:

```
python -m pip install --upgrade pip
pip install requests tqdm
python scripts/download_opportunity.py --out dataset/Opportunity
```

Notes
-----

- MM‑Fi Google Drive is large (~77 GB). If you prefer, download it manually from the Drive web UI.
- The MM‑Fi anonymized RGB images are hosted on Aliyun (OSS) — see the project README for that link.

Python environment (Windows)
---------------------------

A PowerShell helper to create a venv and install base requirements is provided at `scripts/setup_venv.ps1`.

Steps (PowerShell):

```
powershell -ExecutionPolicy Bypass -File scripts/setup_venv.ps1 -EnvName .venv
```

Activate the venv before running commands:

```
# PowerShell
. .venv\Scripts\Activate.ps1

# CMD
.venv\Scripts\activate.bat
```

Install base requirements and (optionally) a GPU-capable PyTorch wheel. Replace the PyTorch install command with the CUDA version that matches your system; example for CUDA 13.2:

```
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu132 torch torchvision -U
```

Using the Opportunity Dataset loader
-----------------------------------

A minimal `torch.utils.data.Dataset` implementation for Opportunity CSV files is included at `dataloaders/opportunity_pytorch_dataset.py`. It expects extracted CSV files under `dataset/Opportunity/`.

Example usage after activating the venv (point to the extracted CSV folder):

```
python -c "from dataloaders.opportunity_pytorch_dataset import OpportunityDataset; ds=OpportunityDataset('dataset/Opportunity', seq_len=128); print(len(ds))"
```

Run DataLoader example
----------------------

After the venv is activated, run the example script to fetch one batch for Opportunity (uses GPU if available):

```
python -c "from dataloaders.opportunity_pytorch_dataset import OpportunityDataset; ds=OpportunityDataset('dataset/Opportunity', seq_len=128); print(len(ds))"
```


