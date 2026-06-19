#!/usr/bin/env python3
"""Check MILLISEC sampling regularity for Opportunity .dat files.

Prints per-file stats (rows, median_dt, mean_dt, std_dt, min_dt, max_dt, cv, outliers)
and flags files that likely need interpolation/resampling.
"""
import glob
import os
import sys
import numpy as np


def analyze_file(path):
    times = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                t = float(parts[0])
            except Exception:
                continue
            times.append(t)
    if len(times) < 2:
        return None
    a = np.array(times, dtype=np.float64)
    dt = np.diff(a)
    median = float(np.median(dt))
    mean = float(np.mean(dt))
    std = float(np.std(dt))
    mn = float(np.min(dt))
    mx = float(np.max(dt))
    cv = std / mean if mean != 0 else float('inf')
    outliers = int(np.sum((dt > 2 * median) | (dt < 0.5 * median)))
    return dict(rows=len(a), median_dt=median, mean_dt=mean, std_dt=std, min_dt=mn, max_dt=mx, cv=cv, outliers=outliers)


def main(root=None):
    if root is None:
        root = os.path.join('dataset', 'Opportunity', 'extracted', 'OpportunityUCIDataset', 'dataset')
    pattern = os.path.join(root, '*.dat')
    files = sorted(glob.glob(pattern))
    if not files:
        print('No .dat files found at', pattern, file=sys.stderr)
        return 2
    print('Checking', len(files), 'files in', root)
    irregular = []
    for p in files:
        stats = analyze_file(p)
        if stats is None:
            print(os.path.basename(p), '-> too few rows')
            continue
        flag = False
        if stats['cv'] > 0.02 or stats['outliers'] > 0 or stats['min_dt'] <= 0:
            flag = True
        print(f"{os.path.basename(p):30s} rows={stats['rows']:6d} med_dt={stats['median_dt']:7.3f}ms mean={stats['mean_dt']:7.3f}ms std={stats['std_dt']:7.3f}ms min={stats['min_dt']:7.3f}ms max={stats['max_dt']:7.3f}ms cv={stats['cv']:.4f} outliers={stats['outliers']:3d} {'IRREGULAR' if flag else ''}")
        if flag:
            irregular.append((p, stats))
    print('\nSummary: {} irregular files out of {}'.format(len(irregular), len(files)))
    if irregular:
        print('Irregular files:')
        for p, s in irregular:
            print(' -', os.path.basename(p), 'med_dt={:.3f}ms cv={:.4f} outliers={}'.format(s['median_dt'], s['cv'], s['outliers']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
