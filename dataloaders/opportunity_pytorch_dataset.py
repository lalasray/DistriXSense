"""Opportunity PyTorch Dataset with .dat parsing and multimodal collate.

This loader accepts the original Opportunity .dat files (whitespace-separated
float columns), or CSV files with numeric columns. It builds sliding-window
sequences and returns per-window tensors. The file-level scan is lightweight
and windows are read on-demand to avoid loading entire datasets into memory.

Features:
- Support for `.dat` and `.csv` files found under `root`
- Optional `label_col` (0-based) to extract labels from a dedicated column
- `seq_len` and `step` sliding-window generation
- `opportunity_collate` helper: pads sequences, returns `sequences`, `mask`,
  `lengths`, and `labels` (majority vote per-window when available)

Usage:
```
from dataloaders.opportunity_pytorch_dataset import OpportunityDataset, opportunity_collate
ds = OpportunityDataset('dataset/Opportunity', seq_len=128, step=64, label_col=None)
dl = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=opportunity_collate)
for batch in dl:
    print(batch['sequences'].shape, batch['mask'].shape)
    break
```
"""

from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple


def _is_dat_file(path: Path) -> bool:
    return path.suffix.lower() == '.dat'


def _parse_line_floats(line: str) -> List[float]:
    parts = line.strip().split()
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            vals.append(0.0)
    return vals


def create_group_map_by_axes(n_cols: int, label_col: Optional[int] = None, preferred_axis: int = 3) -> dict:
    """Create a default group_map by assuming sensor axes (e.g. triaxial -> 3).

    - If (n_cols - has_label) is divisible by `preferred_axis`, group every
      `preferred_axis` columns into one sensor stream (e.g., accel x/y/z).
    - Otherwise fall back to single-column groups.

    Returns dict name->list(indices).
    """
    has_label = 1 if label_col is not None else 0
    feature_cols = n_cols - has_label
    gm = {}
    if feature_cols <= 0:
        return gm
    if feature_cols % preferred_axis == 0:
        n_sensors = feature_cols // preferred_axis
        for i in range(n_sensors):
            start = i * preferred_axis
            gm[f'sensor_{i}'] = list(range(start, start + preferred_axis))
        return gm
    # fallback: group by single columns
    for i in range(feature_cols):
        gm[f'sensor_{i}'] = [i]
    return gm


def infer_group_map_from_first_file(root: str = 'dataset/Opportunity', label_col: Optional[int] = None, preferred_axis: int = 3) -> dict:
    """Inspect the first data file under `root` and infer a group_map.

    This is a convenience wrapper that reads the first `.dat` or `.csv` file,
    determines the column count and returns a `group_map` using
    `create_group_map_by_axes`.
    """
    p = Path(root)
    if not p.exists():
        return {}
    files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in ('.dat', '.csv')])
    if not files:
        return {}
    f = files[0]
    n_cols = None
    with open(f, 'r', errors='ignore') as fh:
        for line in fh:
            if not line.strip():
                continue
            vals = _parse_line_floats(line)
            n_cols = len(vals)
            break
    if n_cols is None:
        return {}
    return create_group_map_by_axes(n_cols, label_col=label_col, preferred_axis=preferred_axis)


class OpportunityDataset(Dataset):
    def __init__(self, root: str = 'dataset/Opportunity', seq_len: int = 128, step: int = 64,
                 files: Optional[List[str]] = None, label_col: Optional[int] = None,
                 normalize: bool = False, group_map: Optional[dict] = None, window_ms: Optional[float] = None,
                 resample_on_drift: bool = False, resample_cv_threshold: float = 0.02, resample_outlier_factor: float = 2.0):
        self.root = Path(root)
        self.seq_len = int(seq_len)
        self.step = int(step)
        self.label_col = label_col
        self.normalize = bool(normalize)
        # group_map: optional dict mapping group_name -> list of column indices (0-based, after removing label_col)
        # if None, each column becomes its own sensor stream named s0, s1, ...
        self.group_map = group_map
        self.window_ms = float(window_ms) if window_ms is not None else None
        # resampling options: if True, check per-window timing and interpolate when irregular
        self.resample_on_drift = bool(resample_on_drift)
        self.resample_cv_threshold = float(resample_cv_threshold)
        self.resample_outlier_factor = float(resample_outlier_factor)

        if files is None:
            self.files = sorted([p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in ('.dat', '.csv')])
        else:
            self.files = [self.root / f for f in files]

        # seq index entries: (file_path, start_row, n_features)
        self.seqs: List[Tuple[Path, int, int]] = []
        self._file_meta = {}  # path -> dict with n_rows, n_cols
        # if user requested time-based windowing, compute seq_len from first file's MILLISEC column
        if self.window_ms is not None and len(self.files) > 0:
            # try to compute sampling rate (samples per second) from first file
            first = self.files[0]
            try:
                with open(first, 'r', errors='ignore') as fh:
                    times = []
                    for line in fh:
                        line=line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        try:
                            t = float(parts[0])
                            times.append(t)
                        except Exception:
                            continue
                        if len(times) >= 50:
                            break
                if len(times) >= 2:
                    diffs = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:]) if t2 > t1]
                    if len(diffs) > 0:
                        # median delta in milliseconds
                        import statistics
                        median_dt = statistics.median(diffs)
                        if median_dt > 0:
                            sampling_hz = 1000.0 / median_dt
                            computed = max(1, int(round((self.window_ms / 1000.0) * sampling_hz)))
                            self.seq_len = computed
            except Exception:
                pass

        self._scan_files()

    def _scan_files(self):
        for f in self.files:
            # determine number of rows and columns by sampling
            n_rows = 0
            n_cols = None
            with open(f, 'r', errors='ignore') as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    vals = _parse_line_floats(line)
                    if n_cols is None:
                        n_cols = len(vals)
                    n_rows += 1
            if n_cols is None or n_rows == 0:
                continue
            self._file_meta[f] = {'n_rows': n_rows, 'n_cols': n_cols}
            for start in range(0, max(0, n_rows - self.seq_len + 1), self.step):
                self.seqs.append((f, start, n_cols))

        # if no explicit group_map provided, create default per-column groups based on first file
        if self.group_map is None and len(self.files) > 0 and self.files[0] in self._file_meta:
            ncols = self._file_meta[self.files[0]]['n_cols'] - (1 if self.label_col is not None else 0)
            self.group_map = {f's{j}': [j] for j in range(ncols)}
    def __len__(self):
        return len(self.seqs)

    def _read_window(self, f: Path, start: int, count: int) -> Tuple[np.ndarray, Optional[List[int]]]:
        """Read `count` rows starting at `start` from file `f`.

        Returns (data: np.ndarray shape (count, n_cols), labels: Optional[list])
        """
        rows = []
        labels = [] if self.label_col is not None else None
        with open(f, 'r', errors='ignore') as fh:
            # skip
            for _ in range(start):
                next(fh, None)
            for _ in range(count):
                line = fh.readline()
                if not line:
                    break
                vals = _parse_line_floats(line)
                if self.label_col is not None and self.label_col < len(vals):
                    labels.append(int(vals[self.label_col]))
                    # remove label from features
                    vals = vals[:self.label_col] + vals[self.label_col+1:]
                rows.append(vals)
        if len(rows) == 0:
            data = np.zeros((0, self._file_meta[f]['n_cols'] - (1 if self.label_col is not None else 0)), dtype=np.float32)
        else:
            # pad/truncate columns to expected n_cols
            exp_cols = self._file_meta[f]['n_cols'] - (1 if self.label_col is not None else 0)
            arr = np.zeros((len(rows), exp_cols), dtype=np.float32)
            for i, r in enumerate(rows):
                for j in range(min(len(r), exp_cols)):
                    arr[i, j] = r[j]
            data = arr
        return data, labels

    def _maybe_resample_window(self, data: np.ndarray) -> np.ndarray:
        """Check MILLISEC regularity in `data` and linearly interpolate to regular grid if needed.

        Assumes MILLISEC is the first column. Returns possibly-modified data of same shape.
        """
        if not self.resample_on_drift or data.size == 0 or data.shape[1] == 0:
            return data
        # need at least two samples to compute dt
        if data.shape[0] < 2:
            return data
        times = data[:, 0].astype(np.float64)
        # if times are all zeros or identical, skip
        if np.all(times == times[0]):
            return data
        dt = np.diff(times)
        # ignore non-positive diffs when computing stats, but mark as irregular if any
        positive = dt[dt > 0]
        if positive.size == 0:
            return data
        mean = float(positive.mean())
        std = float(positive.std())
        cv = std / mean if mean != 0 else float('inf')
        outliers = int(np.sum((positive > self.resample_outlier_factor * np.median(positive)) | (positive < 0.5 * np.median(positive))))
        need = (cv > self.resample_cv_threshold) or (outliers > 0) or np.any(dt <= 0)
        if not need:
            return data

        # perform linear interpolation onto regular grid based on median dt
        median_dt = float(np.median(positive))
        L = data.shape[0]
        target_times = np.arange(L) * median_dt + times[0]
        # Prepare output array
        out = np.empty_like(data, dtype=np.float32)
        out[:, 0] = target_times.astype(np.float32)
        # interpolate each other column
        for c in range(1, data.shape[1]):
            col = data[:, c]
            # mask NaNs
            x = times
            y = col
            # For interpolation, need increasing x; if duplicates exist, use first occurrence
            # Use numpy.interp which requires increasing x
            # Build unique x indices
            try:
                # remove NaNs in y for interpolation
                valid = ~np.isnan(y)
                if np.sum(valid) < 2:
                    out[:, c] = np.nan_to_num(y, nan=0.0).astype(np.float32)
                    continue
                x_valid = x[valid]
                y_valid = y[valid]
                # ensure strictly increasing x_valid
                # if not, perturb slightly
                if np.any(np.diff(x_valid) <= 0):
                    x_valid = np.cumsum(np.maximum(1e-3, np.diff(np.concatenate(([0.0], x_valid))))) + x_valid[0]
                yi = np.interp(target_times, x_valid, y_valid).astype(np.float32)
                out[:, c] = yi
            except Exception:
                out[:, c] = np.nan_to_num(col, nan=0.0).astype(np.float32)
        return out

    def __getitem__(self, idx):
        f, start, n_cols = self.seqs[idx]
        data, labels = self._read_window(f, start, self.seq_len)
        # optionally resample this window if timing is irregular
        if self.resample_on_drift and data.size:
            try:
                data = self._maybe_resample_window(data)
            except Exception:
                pass
        # optional per-window normalization
        if self.normalize and data.size > 0:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True) + 1e-6
            data = (data - mean) / std

        seq_tensor = torch.from_numpy(data.astype(np.float32)) if data.size else torch.empty((0, 0), dtype=torch.float32)

        # create per-group streams
        streams = {}
        if self.group_map:
            for gname, cols in self.group_map.items():
                cols = [int(c) for c in cols]
                # guard against out-of-range
                cols_filtered = [c for c in cols if c < seq_tensor.size(1)]
                if len(cols_filtered) == 0:
                    streams[gname] = torch.zeros((seq_tensor.size(0), 0), dtype=torch.float32)
                else:
                    streams[gname] = seq_tensor[:, cols_filtered]
        else:
            streams = {'s_all': seq_tensor}

        label_out = -1
        if labels:
            # majority vote ignoring zeros (0 often means null label in Opportunity)
            c = Counter([l for l in labels if l != 0])
            if len(c) == 0:
                label_out = 0
            else:
                label_out = c.most_common(1)[0][0]

        return {
            'sequence': seq_tensor,  # shape (L, C)
            'streams': streams,      # dict: group_name -> Tensor (L, C_g)
            'length': seq_tensor.size(0),
            'label': int(label_out) if labels is not None else None,
            'file': str(f.name),
            'start': start,
        }


def opportunity_collate(batch: List[dict], pad_value: float = 0.0) -> dict:
    """Collate function that pads variable-length sequences and returns masks.

    Returns dict with keys:
      - sequences: Tensor (B, T, C)
      - mask: BoolTensor (B, T) where True indicates valid timesteps
      - lengths: LongTensor (B,)
      - labels: LongTensor (B,) or None
      - files, starts
    """
    # batch streams: collect group names
    group_names = set()
    for b in batch:
        # prefer 'streams' key, fall back to single 'sequence'
        if 'streams' in b:
            group_names.update(b['streams'].keys())
        else:
            group_names.add('s_all')
    group_names = sorted(list(group_names))

    B = len(batch)
    lengths = [b['length'] for b in batch]
    max_len = max(lengths) if lengths else 0

    streams_out = {}
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    for g in group_names:
        # determine channels for this group from first non-empty occurrence
        C = 0
        for b in batch:
            if 'streams' in b:
                t = b['streams'].get(g)
            else:
                t = b.get('sequence')
            if t is not None and t.ndim == 2:
                C = t.size(1)
                break
        out_g = torch.full((B, max_len, C), float(pad_value), dtype=torch.float32)
        for i, b in enumerate(batch):
            if 'streams' in b:
                t = b['streams'].get(g)
            else:
                # fallback: use whole sequence as s_all
                t = b.get('sequence') if g == 's_all' else None
            if t is None:
                continue
            L = t.size(0)
            if C > 0 and t.size(1) > 0:
                out_g[i, :L, :t.size(1)] = t
            mask[i, :L] = True
        streams_out[g] = out_g

    labels = None
    if any(b.get('label') is not None for b in batch):
        labels = torch.tensor([b.get('label', -1) if b.get('label') is not None else -1 for b in batch], dtype=torch.long)

    return {
        'streams': streams_out,   # dict group_name -> Tensor (B, T, C_g)
        'mask': mask,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'labels': labels,
        'files': [b['file'] for b in batch],
        'starts': [b['start'] for b in batch],
    }


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='dataset/Opportunity')
    p.add_argument('--seq_len', type=int, default=32)
    args = p.parse_args()
    ds = OpportunityDataset(root=args.root, seq_len=args.seq_len)
    print('files', [p.name for p in ds.files])
    print('seqs', len(ds))
    if len(ds) > 0:
        item = ds[0]
        print('sequence', item['sequence'].shape, 'label', item['label'])
