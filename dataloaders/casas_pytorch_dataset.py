from pathlib import Path
from typing import List, Optional
import os
import csv
import datetime

import torch
from torch.utils.data import Dataset


class CASASSequenceDataset(Dataset):
    """A minimal PyTorch Dataset for CASAS CSV files.

    Expects per-home CSV files with lines: date time sensor1 sensor2 message
    This loader converts sensors to integer indices and returns sequences of
    sensor events as tensors. Labels (activity) are returned when present.
    """

    def __init__(self, root: str, homes: Optional[List[str]] = None, seq_len: int = 256):
        self.root = Path(root)
        self.seq_len = seq_len
        # find CSV files
        all_files = sorted(self.root.glob('*.csv'))
        if homes:
            files = [f for f in all_files if f.stem in homes or any(h in f.stem for h in homes)]
        else:
            files = all_files
        self.files = files

        # build sensor vocabulary from files (simple pass)
        self.sensor2idx = {}
        self.idx2sensor = []
        for f in self.files:
            with f.open('r', encoding='utf-8', errors='ignore') as fd:
                reader = csv.reader(fd)
                for row in reader:
                    if len(row) < 5:
                        continue
                    sensor = row[2].strip()
                    if sensor not in self.sensor2idx:
                        self.sensor2idx[sensor] = len(self.idx2sensor)
                        self.idx2sensor.append(sensor)

        # prepare in-memory index of sequences: (file, start_line)
        self.seqs = []
        for f in self.files:
            # count lines and create sliding windows
            with f.open('r', encoding='utf-8', errors='ignore') as fd:
                lines = sum(1 for _ in fd)
            for start in range(0, max(1, lines - self.seq_len + 1), self.seq_len):
                self.seqs.append((f, start))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        f, start = self.seqs[idx]
        events = []
        labels = []
        with f.open('r', encoding='utf-8', errors='ignore') as fd:
            for i, row in enumerate(csv.reader(fd)):
                if i < start:
                    continue
                if i >= start + self.seq_len:
                    break
                if len(row) < 5:
                    continue
                sensor = row[2].strip()
                msg = row[4].strip()
                sensor_idx = self.sensor2idx.get(sensor, -1)
                events.append(sensor_idx)
                labels.append(msg if msg else '')

        # convert to tensor: sensor indices
        events_tensor = torch.tensor(events, dtype=torch.long)
        return {
            'events': events_tensor,
            'labels': labels,
            'file': str(f.name),
            'start': start,
        }


if __name__ == '__main__':
    # quick smoke test (does not require GPU)
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--root', default='dataset/CASAS')
    args = p.parse_args()

    ds = CASASSequenceDataset(args.root, seq_len=128)
    print('Found', len(ds), 'sequences from', len(ds.files), 'files and', len(ds.idx2sensor) if hasattr(ds, 'idx2sensor') else 0, 'sensors')
    item = ds[0]
    print('Example:', item['file'], item['start'], item['events'].shape)
