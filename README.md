# DistriXSense

Datasets
--------

This project uses two external datasets. Place downloads under the `dataset/` subfolder.

- CASAS Smart Home (Zenodo record): total ~3.0 GB. Direct file URLs (use a browser or curl/wget):
	- https://zenodo.org/record/15708568/files/data.zip?download=1
	- https://zenodo.org/record/15708568/files/labeled_data.zip?download=1
	- https://zenodo.org/record/15708568/files/floorplans.zip?download=1

- MM‑Fi (MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset): main archive hosted on Google Drive (~77 GB). Project page and repo:
	- https://ntu-aiot-lab.github.io/mm-fi
	- https://github.com/ybhbingo/MMFi_dataset
	- Google Drive folder: https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz?usp=sharing

Automated helper
----------------

A helper script is provided to download CASAS automatically and to attempt MM‑Fi download via `gdown` if available:

- `scripts/download_datasets.py` — run with Python 3. It will download CASAS files into `dataset/CASAS/` and will try to download the MM‑Fi Google Drive folder into `dataset/MMFi/` when `gdown` is installed.

Quick commands
--------------

Install recommended helpers and run the downloader:

```
python -m pip install --upgrade pip
pip install requests tqdm gdown
python scripts/download_datasets.py
```

Notes
-----

- MM‑Fi Google Drive is large (~77 GB). If you prefer, download it manually from the Drive web UI.
- The MM‑Fi anonymized RGB images are hosted on Aliyun (OSS) — see the project README for that link.
