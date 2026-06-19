import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Ensure project root is on PYTHONPATH so `dataloaders` is importable
proj_root = str(Path(__file__).resolve().parents[1])
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from dataloaders.casas_pytorch_dataset import CASASSequenceDataset


def collate_batch(batch):
    # batch: list of dicts
    events = [b['events'] for b in batch]
    max_len = max([e.size(0) for e in events])
    padded = torch.zeros(len(events), max_len, dtype=torch.long)
    for i, e in enumerate(events):
        padded[i, : e.size(0)] = e
    return {'events': padded, 'files': [b['file'] for b in batch]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='dataset/CASAS/data/data')
    p.add_argument('--seq_len', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=8)
    args = p.parse_args()

    ds = CASASSequenceDataset(args.root, seq_len=args.seq_len)
    print('Dataset: files=', len(ds.files), 'sequences=', len(ds), 'vocab=', len(ds.idx2sensor))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_batch, pin_memory=True)

    for batch in dl:
        events = batch['events'].to(device)
        print('Batch events shape', events.shape)
        break


if __name__ == '__main__':
    main()
