"""Helper to download Opportunity dataset (best-effort).

The Opportunity dataset hosting varies; this script attempts to fetch the UCI
archive copy or falls back to printing manual instructions.

Usage:
    python scripts/download_opportunity.py --out dataset/Opportunity

Note: the UCI archive may require manual acceptance of license; if download
fails, visit the links in README and place the extracted files under
`dataset/Opportunity`.
"""
import argparse
import os
import sys
from urllib.parse import urljoin
import requests
from tqdm import tqdm

UCI_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/'
FILES = [
    'OpportunityUCIDataset.zip',
]


def download_file(url, dest):
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(dest)) as p:
        for chunk in r.iter_content(1024 * 32):
            if not chunk:
                continue
            f.write(chunk)
            p.update(len(chunk))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', '-o', default='dataset/Opportunity')
    args = p.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)

    print('Attempting to download Opportunity dataset from UCI archive...')
    for fname in FILES:
        url = urljoin(UCI_URL, fname)
        dest = os.path.join(out, fname)
        try:
            download_file(url, dest)
            print('Downloaded', dest)
        except Exception as e:
            print('Failed to download', url, '-', e)
            print('Please download manually from the UCI/Opportunity project pages and place the extracted files under', out)
            sys.exit(1)

    print('Done. Extract the zip files into', out)


if __name__ == '__main__':
    main()
