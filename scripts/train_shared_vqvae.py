"""Train MultiModalSharedVQVAE on Opportunity streams with checkpointing and basic logging."""
import argparse
import csv
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


def metric_to_float(metric):
    if metric is None:
        return None
    if torch.is_tensor(metric):
        return float(metric.detach())
    return float(metric)


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


def majority_labels(label_stream: torch.Tensor) -> torch.Tensor:
    # label_stream: (B, T, 1) or (B, T); zeros are treated as null/background.
    labels = label_stream.squeeze(-1).long()
    out = []
    for row in labels:
        valid = row[row != 0]
        if valid.numel() == 0:
            out.append(row.new_tensor(0))
        else:
            out.append(torch.mode(valid).values)
    return torch.stack(out)


def compute_or_load_norm_stats(ds, modalities, collate_fn, stats_path: Path, batch_size: int, num_workers: int, std_floor: float):
    expected_modalities = list(modalities.keys())
    if stats_path.exists():
        with open(stats_path, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        saved_modalities = list(raw.get('modalities', {}).keys())
        saved_std_floor = float(raw.get('std_floor', -1.0))
        if saved_modalities == expected_modalities and saved_std_floor == float(std_floor):
            return {
                m: {
                    'mean': torch.tensor(raw['modalities'][m]['mean'], dtype=torch.float32),
                    'std': torch.tensor(raw['modalities'][m]['std'], dtype=torch.float32).clamp_min(std_floor),
                }
                for m in modalities
            }
        print(f'Ignoring stale normalization stats: {stats_path}')

    stats_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = stats_path.with_suffix('.stale.json')
    if stats_path.exists():
        stats_path.replace(tmp_path)
        print(f'Moved stale normalization stats to: {tmp_path}')

    sums = {m: None for m in modalities}
    sq_sums = {m: None for m in modalities}
    counts = {m: 0 for m in modalities}
    stats_dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    for batch in tqdm(stats_dl, desc='Computing normalization stats', unit='batch'):
        streams = batch['streams']
        for m in modalities:
            x = torch.nan_to_num(streams[m].float(), nan=0.0, posinf=0.0, neginf=0.0)
            flat = x.reshape(-1, x.size(-1)).double()
            if sums[m] is None:
                sums[m] = flat.sum(dim=0)
                sq_sums[m] = (flat * flat).sum(dim=0)
            else:
                sums[m] += flat.sum(dim=0)
                sq_sums[m] += (flat * flat).sum(dim=0)
            counts[m] += flat.size(0)

    stats = {}
    serializable = {'modalities': {}, 'num_windows': len(ds), 'std_floor': float(std_floor)}
    for m in modalities:
        mean = sums[m] / max(1, counts[m])
        var = (sq_sums[m] / max(1, counts[m])) - (mean * mean)
        std = torch.sqrt(var.clamp_min(1e-12))
        stats[m] = {'mean': mean.float(), 'std': std.float().clamp_min(std_floor)}
        serializable['modalities'][m] = {
            'mean': stats[m]['mean'].tolist(),
            'std': stats[m]['std'].tolist(),
            'count': counts[m],
        }

    with open(stats_path, 'w', encoding='utf-8') as fh:
        json.dump(serializable, fh, indent=2)
    return stats


def normalize_with_stats(x: torch.Tensor, stats: dict, modality: str, std_floor: float) -> torch.Tensor:
    mean = stats[modality]['mean'].view(1, 1, -1)
    std = stats[modality]['std'].view(1, 1, -1).clamp_min(std_floor)
    return (x - mean) / std


def init_wandb(args, modalities: dict, codebook: dict, dataset_windows: int, full_dataset_windows: int, device: torch.device, amp_enabled: bool):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            'Weights & Biases logging was requested with --wandb, but wandb is not installed. '
            'Install it with: pip install wandb'
        ) from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
    config = vars(args).copy()
    config.update({
        'modalities': modalities,
        'codebook': codebook,
        'num_modalities': len(modalities),
        'dataset_windows': dataset_windows,
        'full_dataset_windows': full_dataset_windows,
        'device_resolved': str(device),
        'cuda_device': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'amp_enabled': amp_enabled,
    })
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        tags=tags or None,
        config=config,
        dir=args.wandb_dir or None,
        mode=args.wandb_mode,
    )
    return run


def wandb_log(run, payload: dict, step: int = None):
    if run is None:
        return
    clean = {key: value for key, value in payload.items() if value is not None}
    run.log(clean, step=step)


def wandb_log_artifact(run, artifact_path: Path, artifact_type: str, aliases=None):
    if run is None or not artifact_path.exists():
        return
    import wandb
    artifact = wandb.Artifact(artifact_path.stem, type=artifact_type)
    artifact.add_file(str(artifact_path))
    run.log_artifact(artifact, aliases=aliases)


def wandb_watch_model(run, model, log_interval: int):
    if run is None:
        return
    import wandb
    wandb.watch(model, log='gradients', log_freq=max(1, log_interval))


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
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--checkpoint_dir', default='checkpoints')
    p.add_argument('--group_map', default='dataset/Opportunity/group_map_official.json')
    p.add_argument('--quantizer', choices=('standard', 'ema'), default='ema')
    p.add_argument('--no-ema', action='store_true', help='Use the standard gradient-updated quantizer instead of EMA.')
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--amp', action='store_true', help='Use CUDA automatic mixed precision.')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of dataset windows to train on, in (0, 1].')
    p.add_argument('--normalize_batch', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--norm_stats', default=None, help='Path to dataset normalization stats JSON. Computed if missing.')
    p.add_argument('--norm_std_floor', type=float, default=1.0, help='Minimum std used for dataset normalization.')
    p.add_argument('--no_dataset_norm', action='store_true')
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--use_temporal_interpolator', action='store_true')
    p.add_argument('--frame_drop_prob', type=float, default=0.0)
    p.add_argument('--min_keep_frames', type=int, default=8)
    p.add_argument('--activity_contrastive_loss', action=argparse.BooleanOptionalAction, default=True, help='Use labels only as a training-time auxiliary contrastive loss.')
    p.add_argument('--label_stream', default='label_HL_Activity')
    p.add_argument('--label_vocab_size', type=int, default=4096)
    p.add_argument('--label_embedding_dim', type=int, default=32)
    p.add_argument('--label_contrastive_weight', type=float, default=0.02)
    p.add_argument('--encoder_res_blocks', type=int, default=1)
    p.add_argument('--decoder_res_blocks', type=int, default=1)
    p.add_argument('--stft_loss_weight', type=float, default=0.03)
    p.add_argument('--wavelet_loss_weight', type=float, default=0.05)
    p.add_argument('--reed_transition_loss_weight', type=float, default=0.20)
    p.add_argument('--learnable_loss_weights', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--modality_loss_reduction', choices=('mean', 'sum'), default='mean')
    p.add_argument('--wandb', action='store_true', help='Log training config, losses, alphas, and optional artifacts to Weights & Biases.')
    p.add_argument('--wandb_project', default='DistriXSense-vqvae')
    p.add_argument('--wandb_entity', default='')
    p.add_argument('--wandb_run_name', default='')
    p.add_argument('--wandb_tags', default='')
    p.add_argument('--wandb_mode', choices=('online', 'offline', 'disabled'), default='online')
    p.add_argument('--wandb_dir', default='')
    p.add_argument('--wandb_log_interval', type=int, default=10, help='Log batch metrics to W&B every N optimizer steps.')
    p.add_argument('--wandb_watch', action='store_true', help='Ask W&B to watch model gradients/parameters.')
    p.add_argument('--wandb_log_artifacts', action='store_true', help='Upload metrics.csv and checkpoints as W&B artifacts.')
    args = p.parse_args()
    use_activity_contrastive_loss = bool(args.activity_contrastive_loss)
    if args.no_ema:
        args.quantizer = 'standard'

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
    if use_activity_contrastive_loss and args.label_stream not in available_modalities:
        raise ValueError(f'Label stream {args.label_stream!r} is missing from the dataset/group map.')

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

    wandb_run = init_wandb(
        args=args,
        modalities=modalities,
        codebook=codebook,
        dataset_windows=len(ds),
        full_dataset_windows=len(full_ds),
        device=device,
        amp_enabled=amp_enabled,
    )

    norm_stats = None
    if not args.no_dataset_norm and not args.normalize_batch:
        stats_path = Path(args.norm_stats) if args.norm_stats else Path(args.checkpoint_dir) / 'norm_stats.json'
        norm_stats = compute_or_load_norm_stats(
            ds=ds,
            modalities=modalities,
            collate_fn=opportunity_collate,
            stats_path=stats_path,
            batch_size=args.batch,
            num_workers=args.num_workers,
            std_floor=args.norm_std_floor,
        )
        print(f'Normalization stats: {stats_path}')

    model = MultiModalSharedVQVAE(
        modality_dims=modalities,
        modality_codebook_sizes=codebook,
        hidden=64,
        latent_dim=32,
        beta=args.beta,
        input_len=args.seq_len,
        use_temporal_interpolator=args.use_temporal_interpolator,
        quantizer_type=args.quantizer,
        activity_contrastive_loss=use_activity_contrastive_loss,
        label_vocab_size=args.label_vocab_size,
        label_embedding_dim=args.label_embedding_dim,
        label_contrastive_weight=args.label_contrastive_weight,
        encoder_res_blocks=args.encoder_res_blocks,
        decoder_res_blocks=args.decoder_res_blocks,
        stft_loss_weight=args.stft_loss_weight,
        wavelet_loss_weight=args.wavelet_loss_weight,
        reed_transition_loss_weight=args.reed_transition_loss_weight,
        learnable_loss_weights=args.learnable_loss_weights,
        modality_loss_reduction=args.modality_loss_reduction,
    ).to(device)
    if wandb_run is not None and args.wandb_watch:
        wandb_watch_model(wandb_run, model, args.wandb_log_interval)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metrics_path = Path(args.checkpoint_dir) / 'metrics.csv'
    metric_fields = [
        'epoch',
        'avg_total_loss',
        'avg_recon_loss',
        'avg_codebook_loss',
        'avg_commitment_loss',
        'avg_label_contrastive_loss',
        'avg_stft_loss',
        'avg_wavelet_loss',
        'avg_reed_transition_loss',
        'avg_perplexity',
        'lr',
        'beta',
        'quantizer',
        'modality_loss_reduction',
        'alpha_recon',
        'alpha_vq',
        'alpha_label',
        'alpha_stft',
        'alpha_wavelet',
        'alpha_reed_transition',
    ]
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as fh:
            first_line = fh.readline().strip()
        if first_line.split(',') != metric_fields:
            stale_metrics_path = metrics_path.with_suffix('.stale.csv')
            metrics_path.replace(stale_metrics_path)
            print(f'Moved stale metrics to: {stale_metrics_path}')
    if not metrics_path.exists():
        with open(metrics_path, 'w', newline='', encoding='utf-8') as fh:
            csv.DictWriter(fh, fieldnames=metric_fields).writeheader()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        epoch_metrics = {
            'recon_loss': [],
            'codebook_loss': [],
            'commitment_loss': [],
            'label_contrastive_loss': [],
            'stft_loss': [],
            'wavelet_loss': [],
            'reed_transition_loss': [],
            'perplexity': [],
        }
        epoch_loss_weights = {
            'recon': [],
            'vq': [],
            'label': [],
            'stft': [],
            'wavelet': [],
            'reed_transition': [],
        }
        progress = tqdm(dl, desc=f'Epoch {epoch}/{args.epochs}', unit='batch')
        for i, batch in enumerate(progress):
            streams = batch['streams']
            inputs = {}
            input_lengths = {}
            targets = {}
            condition_labels = None
            if use_activity_contrastive_loss:
                condition_labels = majority_labels(streams[args.label_stream]).to(device, non_blocking=(device.type == 'cuda'))
            for m in modalities:
                if m not in streams:
                    continue
                x = streams[m].float()
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                if args.normalize_batch:
                    mean = x.mean(dim=(0, 1), keepdim=True)
                    std = x.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
                    x = (x - mean) / std
                elif norm_stats is not None:
                    x = normalize_with_stats(x, norm_stats, m, args.norm_std_floor)
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
                out = model(inputs, input_lengths=input_lengths, targets=targets, labels=condition_labels)
                total_loss = out.get('total_loss') if 'total_loss' in out else sum(v['loss'] for v in out.values())
            if not torch.isfinite(total_loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}, batch {i}, step {step + 1}. Stopping before saving a bad checkpoint.')
            scaler.scale(total_loss).backward()
            grad_norm = None
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(total_loss.detach())
            step += 1
            recon_loss = mean_metric(out, modalities, 'recon_loss')
            codebook_loss = mean_metric(out, modalities, 'codebook_loss')
            commitment_loss = mean_metric(out, modalities, 'commitment_loss')
            label_contrastive_loss = mean_metric(out, modalities, 'label_contrastive_loss')
            stft_loss = mean_metric(out, modalities, 'stft_loss')
            wavelet_loss = mean_metric(out, modalities, 'wavelet_loss')
            reed_transition_loss = mean_metric(out, modalities, 'reed_transition_loss')
            perplexity = mean_metric(out, modalities, 'perplexity')
            postfix = {'loss': f'{float(total_loss.detach()):.4f}', 'step': step}
            if recon_loss is not None:
                postfix['recon'] = f'{recon_loss:.4f}'
            if args.quantizer == 'ema':
                postfix['ema'] = 'on'
            elif codebook_loss is not None:
                postfix['codebook'] = f'{codebook_loss:.4f}'
            if commitment_loss is not None:
                postfix['commit'] = f'{commitment_loss:.4f}'
            if label_contrastive_loss is not None:
                postfix['label'] = f'{label_contrastive_loss:.4f}'
            if stft_loss is not None:
                postfix['stft'] = f'{stft_loss:.4f}'
            if wavelet_loss is not None:
                postfix['wavelet'] = f'{wavelet_loss:.4f}'
            if reed_transition_loss is not None:
                postfix['reed_d'] = f'{reed_transition_loss:.4f}'
            if perplexity is not None:
                postfix['ppl'] = f'{perplexity:.2f}'
            loss_weights = out.get('loss_weights')
            if loss_weights:
                postfix['a_recon'] = f"{loss_weights.get('recon', 0):.2f}"
                postfix['a_vq'] = f"{loss_weights.get('vq', 0):.2f}"
                for key in epoch_loss_weights:
                    if key in loss_weights:
                        epoch_loss_weights[key].append(float(loss_weights[key]))
            progress.set_postfix(postfix)
            if wandb_run is not None and args.wandb_log_interval > 0 and step % args.wandb_log_interval == 0:
                log_payload = {
                    'train/step_loss': float(total_loss.detach()),
                    'train/recon_loss': recon_loss,
                    'train/codebook_loss': codebook_loss if args.quantizer != 'ema' else None,
                    'train/commitment_loss': commitment_loss,
                    'train/label_contrastive_loss': label_contrastive_loss,
                    'train/stft_loss': stft_loss,
                    'train/wavelet_loss': wavelet_loss,
                    'train/reed_transition_loss': reed_transition_loss,
                    'train/perplexity': perplexity,
                    'train/grad_norm': metric_to_float(grad_norm),
                    'train/lr': args.lr,
                    'train/epoch': epoch,
                    'train/batch_index': i,
                }
                if loss_weights:
                    for alpha_name, alpha_value in loss_weights.items():
                        log_payload[f'alpha/{alpha_name}'] = alpha_value
                for m in modalities:
                    if m not in out:
                        continue
                    for metric_name in [
                        'recon_loss',
                        'codebook_loss',
                        'commitment_loss',
                        'label_contrastive_loss',
                        'stft_loss',
                        'wavelet_loss',
                        'reed_transition_loss',
                        'perplexity',
                    ]:
                        value = metric_to_float(out[m].get(metric_name))
                        if value is not None:
                            log_payload[f'modality/{m}/{metric_name}'] = value
                wandb_log(wandb_run, log_payload, step=step)
            for key, value in [
                ('recon_loss', recon_loss),
                ('codebook_loss', codebook_loss),
                ('commitment_loss', commitment_loss),
                ('label_contrastive_loss', label_contrastive_loss),
                ('stft_loss', stft_loss),
                ('wavelet_loss', wavelet_loss),
                ('reed_transition_loss', reed_transition_loss),
                ('perplexity', perplexity),
            ]:
                if value is not None:
                    epoch_metrics[key].append(float(value))
            # periodic codebook usage
            if step % 100 == 0:
                for m in modalities:
                    if m in out:
                        perc = out[m].get('perplexity')
                        print(f'  usage {m} perplexity={perc}')
        t1 = time.time()
        avg_total = epoch_loss / max(1, i+1)
        avg_metrics = {
            key: (sum(values) / len(values) if values else None)
            for key, values in epoch_metrics.items()
        }
        avg_loss_weights = {
            key: (sum(values) / len(values) if values else None)
            for key, values in epoch_loss_weights.items()
        }
        print(
            f"Epoch {epoch} finished, avg loss {avg_total:.4f}, "
            f"recon {avg_metrics['recon_loss'] or 0:.4f}, "
            f"commit {avg_metrics['commitment_loss'] or 0:.4f}, "
            f"ppl {avg_metrics['perplexity'] or 0:.2f}, "
            f"time {t1-t0:.1f}s"
        )
        with open(metrics_path, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=metric_fields)
            metrics_row = {
                'epoch': epoch,
                'avg_total_loss': avg_total,
                'avg_recon_loss': avg_metrics['recon_loss'],
                'avg_codebook_loss': avg_metrics['codebook_loss'] if args.quantizer != 'ema' else None,
                'avg_commitment_loss': avg_metrics['commitment_loss'],
                'avg_label_contrastive_loss': avg_metrics['label_contrastive_loss'],
                'avg_stft_loss': avg_metrics['stft_loss'],
                'avg_wavelet_loss': avg_metrics['wavelet_loss'],
                'avg_reed_transition_loss': avg_metrics['reed_transition_loss'],
                'avg_perplexity': avg_metrics['perplexity'],
                'lr': args.lr,
                'beta': args.beta,
                'quantizer': args.quantizer,
                'modality_loss_reduction': args.modality_loss_reduction,
                'alpha_recon': avg_loss_weights['recon'],
                'alpha_vq': avg_loss_weights['vq'],
                'alpha_label': avg_loss_weights['label'],
                'alpha_stft': avg_loss_weights['stft'],
                'alpha_wavelet': avg_loss_weights['wavelet'],
                'alpha_reed_transition': avg_loss_weights['reed_transition'],
            }
            writer.writerow(metrics_row)
        wandb_log(
            wandb_run,
            {
                'epoch/total_loss': avg_total,
                'epoch/recon_loss': avg_metrics['recon_loss'],
                'epoch/codebook_loss': avg_metrics['codebook_loss'] if args.quantizer != 'ema' else None,
                'epoch/commitment_loss': avg_metrics['commitment_loss'],
                'epoch/label_contrastive_loss': avg_metrics['label_contrastive_loss'],
                'epoch/stft_loss': avg_metrics['stft_loss'],
                'epoch/wavelet_loss': avg_metrics['wavelet_loss'],
                'epoch/reed_transition_loss': avg_metrics['reed_transition_loss'],
                'epoch/perplexity': avg_metrics['perplexity'],
                'epoch/seconds': t1 - t0,
                'epoch/index': epoch,
                'alpha_epoch/recon': avg_loss_weights['recon'],
                'alpha_epoch/vq': avg_loss_weights['vq'],
                'alpha_epoch/label': avg_loss_weights['label'],
                'alpha_epoch/stft': avg_loss_weights['stft'],
                'alpha_epoch/wavelet': avg_loss_weights['wavelet'],
                'alpha_epoch/reed_transition': avg_loss_weights['reed_transition'],
            },
            step=step,
        )
        checkpoint_path = Path(args.checkpoint_dir) / f'model_epoch{epoch}.pt'
        torch.save({'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, checkpoint_path)
        print('Saved checkpoint', checkpoint_path)
        print('Saved metrics', metrics_path)
        if args.wandb_log_artifacts:
            wandb_log_artifact(wandb_run, checkpoint_path, artifact_type='model', aliases=[f'epoch-{epoch}', 'latest'])
            wandb_log_artifact(wandb_run, metrics_path, artifact_type='metrics', aliases=['latest'])

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
