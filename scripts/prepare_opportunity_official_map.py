"""Extract Opportunity zip (if present), infer official group map, and write JSON.

Writes `dataset/Opportunity/group_map_official.json` based on the first .dat file
found in the extracted archive or dataset folder. Uses conservative triaxial grouping.
"""
import zipfile
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
opp_zip = ROOT / 'dataset' / 'Opportunity' / 'opportunity+activity+recognition.zip'
extracted = ROOT / 'dataset' / 'Opportunity' / 'extracted'
out_json = ROOT / 'dataset' / 'Opportunity' / 'group_map_official.json'

def extract_if_needed():
    if not extracted.exists():
        extracted.mkdir(parents=True, exist_ok=True)
    # if dataset already contains dat files at top, skip
    has_dat = any((ROOT / 'dataset' / 'Opportunity').rglob('*.dat'))
    if has_dat and not any(extracted.rglob('*.dat')):
        # assume dat files are already in dataset/Opportunity
        return
    if not opp_zip.exists():
        print('Zip not found at', opp_zip)
        return
    print('Extracting', opp_zip, '->', extracted)
    with zipfile.ZipFile(opp_zip, 'r') as z:
        z.extractall(path=extracted)

def find_first_dat() -> Path:
    # check extracted first, then dataset root
    for p in extracted.rglob('*.dat'):
        return p
    for p in (ROOT / 'dataset' / 'Opportunity').rglob('*.dat'):
        return p
    return None

def infer_and_write(dat_path: Path):
    # simple inspect: read first non-empty line and count columns
    with open(dat_path, 'r', errors='ignore') as fh:
        for line in fh:
            if line.strip():
                cols = line.strip().split()
                n_cols = len(cols)
                break
        else:
            print('No data lines found in', dat_path)
            return

    # default grouping: triaxial groups when divisible
    preferred_axis = 3
    has_label = False
    # in Opportunity, labels may be included at the end; we conservatively assume no label
    feature_cols = n_cols - (1 if has_label else 0)
    group_map = {}
    if feature_cols % preferred_axis == 0:
        n_groups = feature_cols // preferred_axis
        for i in range(n_groups):
            start = i * preferred_axis
            group_map[f'sensor_{i}'] = list(range(start, start+preferred_axis))
    else:
        for i in range(feature_cols):
            group_map[f'sensor_{i}'] = [i]

    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(group_map, fh, indent=2)
    print('Wrote', out_json)

def main():
    extract_if_needed()
    dat = find_first_dat()
    if not dat:
        print('No .dat files found. Place the Opportunity .dat files under dataset/Opportunity or the extracted folder and retry.')
        sys.exit(1)
    print('Using dat file:', dat)
    infer_and_write(dat)

if __name__ == '__main__':
    main()
