#!/usr/bin/env python3
import os
import sys
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except Exception:
    print('Please install required packages: pip install requests tqdm')
    sys.exit(1)


def download_file(url, dest_path, chunk_size=32768):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    with open(dest_path, 'wb') as fd, tqdm(total=total, unit='B', unit_scale=True, desc=dest_path.name) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            fd.write(chunk)
            bar.update(len(chunk))


def main():
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / 'dataset'
    dataset_dir.mkdir(exist_ok=True)

    # CASAS (Zenodo)
    casas_dir = dataset_dir / 'CASAS'
    casas_dir.mkdir(exist_ok=True)
    print('Downloading CASAS metadata and archives into', casas_dir)
    casas_urls = {
        'data.zip': 'https://zenodo.org/record/15708568/files/data.zip?download=1',
        'labeled_data.zip': 'https://zenodo.org/record/15708568/files/labeled_data.zip?download=1',
        'floorplans.zip': 'https://zenodo.org/record/15708568/files/floorplans.zip?download=1',
    }
    for name, url in casas_urls.items():
        dest = casas_dir / name
        if dest.exists():
            print(dest, 'already exists, skipping')
            continue
        try:
            download_file(url, dest)
        except Exception as e:
            print('Failed to download', url, e)

    # MM-Fi (Google Drive)
    mmfi_dir = dataset_dir / 'MMFi'
    mmfi_dir.mkdir(exist_ok=True)
    drive_folder = 'https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz'
    print('\nMM-Fi dataset: the repository links a Google Drive folder (~77 GB).')
    print('Attempting automatic download using gdown (if installed).')
    try:
        import gdown
        print('Found gdown, downloading folder into', mmfi_dir)
        # gdown can download folders with download_folder
        gdown.download_folder(drive_folder, output=str(mmfi_dir), quiet=False)
        print('MM-Fi download attempted. Verify contents in', mmfi_dir)
    except Exception:
        print('gdown not available or download failed. To download MM-Fi manually:')
        print(' - Open', drive_folder)
        print(' - Download the folder contents (large: ~77 GB) and place them in', mmfi_dir)

    print('\nDone. Extract archives as needed.')


if __name__ == '__main__':
    main()
