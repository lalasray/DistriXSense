"""Train MultiModalSharedVQVAE on Opportunity streams with checkpointing and basic logging."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm.auto import tqdm

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


def mean_metric(out: dict, modalities: dict, key: str):
    values = []
    for m in modalities:
        metric = out.get(m, {}).get(key)
        if metric is None:
            continue
        if torch.is_tensor(metric):
            metric = float(metric.detach())
        values.append(float(metric))
    if not values:
        return None
    return sum(values) / len(values)


def drop_random_frames(x: torch.Tensor, drop_prob: float, min_keep: int):
    B, T, C = x.shape
    if drop_prob <= 0:
        lengths = torch.full((B,), T, dtype=torch.long)
        return x, lengths

    min_keep = max(1, min(int(min_keep), T))
    sequences = []
    lengths = []
    for i in range(B):
        keep = torch.rand(T) > drop_prob
        if int(keep.sum()) < min_keep:
            chosen = torch.randperm(T)[:min_keep]
            keep = torch.zeros(T, dtype=torch.bool)
            keep[chosen] = True
        idx = torch.nonzero(keep, as_tuple=False).flatten().sort().values
        seq = x[i, idx, :]
        sequences.append(seq)
        lengths.append(seq.size(0))

    max_len = max(lengths)
    out = x.new_zeros((B, max_len, C))
    for i, seq in enumerate(sequences):
        out[i, :seq.size(0), :] = seq
    return out, torch.tensor(lengths, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='dataset/Opportunity/extracted/OpportunityUCIDataset/dataset')
    p.add_argument('--modalities', default='acc:3,quat:4,reed:1')
    p.add_argument('--codebook', default='acc:128,quat:128,reed:64')
    p.add_argument('--seq_len', type=int, default=32)
    p.add_argument('--step', type=int, default=64)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--checkpoint_dir', default='checkpoints')
    p.add_argument('--group_map', default='dataset/Opportunity/group_map_official.json')
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--amp', action='store_true', help='Use CUDA automatic mixed precision.')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of dataset windows to train on, in (0, 1].')
    p.add_argument('--normalize_batch', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--use_temporal_interpolator', action='store_true')
    p.add_argument('--frame_drop_prob', type=float, default=0.0)
    p.add_argument('--min_keep_frames', type=int, default=8)
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

    if not 0 < args.data_fraction <= 1:
        raise ValueError('--data_fraction must be greater than 0 and less than or equal to 1.')
    if args.frame_drop_prob < 0 or args.frame_drop_prob >= 1:
        raise ValueError('--frame_drop_prob must be greater than or equal to 0 and less than 1.')
    if args.frame_drop_prob > 0 and not args.use_temporal_interpolator:
        raise ValueError('--frame_drop_prob requires --use_temporal_interpolator.')

    group_map = None
    if args.group_map:
        with open(args.group_map, 'r', encoding='utf-8') as fh:
            group_map = json.load(fh)

    full_ds = OpportunityDataset(root=args.root, seq_len=args.seq_len, step=args.step, group_map=group_map)
    available_modalities = set(full_ds.group_map.keys()) if full_ds.group_map else set()
    missing_modalities = sorted(set(modalities) - available_modalities)
    if missing_modalities:
        preview = ', '.join(sorted(available_modalities)[:12])
        raise ValueError(
            f'Requested modalities are missing from the dataset/group map: {missing_modalities}. '
            f'Available examples: {preview}'
        )

    ds = full_ds
    if args.data_fraction < 1:
        subset_len = max(1, int(len(full_ds) * args.data_fraction))
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(full_ds), generator=generator)[:subset_len].tolist()
        ds = Subset(full_ds, indices)
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
    print(f'Dataset windows: {len(ds)} / {len(full_ds)} ({args.data_fraction:.1%})')

    model = MultiModalSharedVQVAE(
        modality_dims=modalities,
        modality_codebook_sizes=codebook,
        hidden=64,
        latent_dim=32,
        input_len=args.seq_len,
        use_temporal_interpolator=args.use_temporal_interpolator,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        progress = tqdm(dl, desc=f'Epoch {epoch}/{args.epochs}', unit='batch')
        for i, batch in enumerate(progress):
            streams = batch['streams']
            inputs = {}
            input_lengths = {}
            targets = {}
            for m in modalities:
                if m not in streams:
                    continue
                x = streams[m].float()
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                if args.normalize_batch:
                    mean = x.mean(dim=(0, 1), keepdim=True)
                    std = x.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
                    x = (x - mean) / std
                target = x.to(device, non_blocking=(device.type == 'cuda'))
                x_input, lengths = drop_random_frames(x, args.frame_drop_prob, args.min_keep_frames)
                x_input = x_input.to(device, non_blocking=(device.type == 'cuda'))
                inputs[m] = x_input
                input_lengths[m] = lengths.to(device, non_blocking=(device.type == 'cuda'))
                targets[m] = target
            if not inputs:
                continue
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(inputs, input_lengths=input_lengths, targets=targets)
                total_loss = out.get('total_loss') if 'total_loss' in out else sum(v['loss'] for v in out.values())
            if not torch.isfinite(total_loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}, batch {i}, step {step + 1}. Stopping before saving a bad checkpoint.')
            scaler.scale(total_loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(total_loss.detach())
            step += 1
            recon_loss = mean_metric(out, modalities, 'recon_loss')
            codebook_loss = mean_metric(out, modalities, 'codebook_loss')
            commitment_loss = mean_metric(out, modalities, 'commitment_loss')
            perplexity = mean_metric(out, modalities, 'perplexity')
            postfix = {'loss': f'{float(total_loss.detach()):.4f}', 'step': step}
            if recon_loss is not None:
                postfix['recon'] = f'{recon_loss:.4f}'
            if codebook_loss is not None:
                postfix['codebook'] = f'{codebook_loss:.4f}'
            if commitment_loss is not None:
                postfix['commit'] = f'{commitment_loss:.4f}'
            if perplexity is not None:
                postfix['ppl'] = f'{perplexity:.2f}'
            progress.set_postfix(postfix)
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
