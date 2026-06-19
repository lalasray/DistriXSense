"""Train MultiModalSharedVQVAE on Opportunity streams with checkpointing and basic logging."""
import argparse
import os
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.opportunity_pytorch_dataset import OpportunityDataset, opportunity_collate
from Code.vqvae.models import MultiModalSharedVQVAE


def parse_modality_str(s: str):
    d = {}
    for item in s.split(','):
        if not item:
            continue
        k, v = item.split(':')
        d[k] = int(v)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='dataset/Opportunity/extracted/OpportunityUCIDataset/dataset')
    p.add_argument('--modalities', default='acc:3,quat:4,reed:1')
    p.add_argument('--codebook', default='acc:128,quat:128,reed:64')
    p.add_argument('--seq_len', type=int, default=32)
    p.add_argument('--step', type=int, default=64)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--checkpoint_dir', default='checkpoints')
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--amp', action='store_true', help='Use CUDA automatic mixed precision.')
    p.add_argument('--num_workers', type=int, default=0)
    args = p.parse_args()

    modalities = parse_modality_str(args.modalities)
    codebook = parse_modality_str(args.codebook)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested with --device cuda, but torch.cuda.is_available() is False.')

    device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device_name == 'auto':
        device_name = 'cpu'
    device = torch.device(device_name)
    amp_enabled = bool(args.amp and device.type == 'cuda')

    ds = OpportunityDataset(root=args.root, seq_len=args.seq_len, step=args.step)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=opportunity_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
        print(f'AMP: {amp_enabled}')
    print(f'Dataset windows: {len(ds)}')

    model = MultiModalSharedVQVAE(modality_dims=modalities, modality_codebook_sizes=codebook, hidden=64, latent_dim=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        for i, batch in enumerate(dl):
            streams = batch['streams']
            inputs = {}
            for m in modalities:
                if m not in streams:
                    continue
                x = streams[m].float().to(device, non_blocking=(device.type == 'cuda'))
                inputs[m] = x
            if not inputs:
                continue
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(inputs)
                total_loss = out.get('total_loss') if 'total_loss' in out else sum(v['loss'] for v in out.values())
            scaler.scale(total_loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(total_loss.detach())
            step += 1
            if step % 10 == 0:
                print(f'Epoch {epoch} step {step} batch {i} loss {float(total_loss):.4f}')
            # periodic codebook usage
            if step % 100 == 0:
                for m in modalities:
                    if m in out:
                        perc = out[m].get('perplexity')
                        print(f'  usage {m} perplexity={perc}')
        t1 = time.time()
        print(f'Epoch {epoch} finished, avg loss {epoch_loss / max(1, i+1):.4f}, time {t1-t0:.1f}s')
        torch.save({'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, os.path.join(args.checkpoint_dir, f'model_epoch{epoch}.pt'))
        print('Saved checkpoint', os.path.join(args.checkpoint_dir, f'model_epoch{epoch}.pt'))


if __name__ == '__main__':
    main()
