"""Build and launch a full-scale VQ-VAE training command from the Opportunity group map."""
import argparse
import math
import json
import subprocess
import sys
from pathlib import Path


def is_requested_family(name: str) -> bool:
    if name == 'MILLISEC' or name.startswith('label_') or name.startswith('LOCATION_TAG'):
        return False
    if name.startswith('REED_'):
        return True
    if 'SHOE' in name:
        return True
    if 'quat' in name:
        return True
    if 'mag' in name or 'COMPASS' in name:
        return True
    if 'gyro' in name or 'ANGVEL' in name:
        return True
    if 'acc' in name or 'accX' in name:
        return True
    return False


def fallback_codebook_size(name: str) -> int:
    if name.startswith('REED_'):
        return 32
    if 'quat' in name:
        return 128
    if 'SHOE' in name:
        return 96
    return 64


def family_bounds(name: str):
    if name.startswith('REED_'):
        return 8, 32
    if 'COMPASS' in name:
        return 8, 32
    if 'quat' in name:
        return 64, 160
    if 'SHOE' in name:
        return 32, 128
    if 'mag' in name:
        return 32, 96
    return 32, 128


def round_codebook_size(value: float, minimum: int, maximum: int) -> int:
    value = max(minimum, min(maximum, int(round(value / 8.0) * 8)))
    return int(value)


def estimate_stream_complexity(root: Path, group_map: dict, modalities: dict, max_files: int, max_rows_per_file: int):
    files = sorted(root.glob('*.dat'))[:max_files]
    stats = {}
    for name, channels in modalities.items():
        stats[name] = {
            'sum': [0.0] * channels,
            'sum_sq': [0.0] * channels,
            'count': 0,
            'delta_sum': 0.0,
            'delta_count': 0,
            'prev': None,
        }

    for file_path in files:
        with open(file_path, 'r', errors='ignore') as fh:
            row_count = 0
            for line in fh:
                if row_count >= max_rows_per_file:
                    break
                if not line.strip():
                    continue
                parts = line.split()
                row_count += 1
                for name in modalities:
                    cols = group_map[name]
                    values = []
                    for col in cols:
                        if col >= len(parts):
                            values.append(0.0)
                            continue
                        try:
                            value = float(parts[col])
                        except Exception:
                            value = 0.0
                        if not math.isfinite(value):
                            value = 0.0
                        values.append(value)

                    s = stats[name]
                    for i, value in enumerate(values):
                        s['sum'][i] += value
                        s['sum_sq'][i] += value * value
                    s['count'] += 1
                    if s['prev'] is not None:
                        s['delta_sum'] += sum(abs(a - b) for a, b in zip(values, s['prev'])) / max(1, len(values))
                        s['delta_count'] += 1
                    s['prev'] = values

    complexity = {}
    for name, s in stats.items():
        count = max(1, s['count'])
        variances = []
        for total, total_sq in zip(s['sum'], s['sum_sq']):
            mean = total / count
            var = max(0.0, total_sq / count - mean * mean)
            variances.append(var)
        mean_std = math.sqrt(sum(variances) / max(1, len(variances)))
        mean_delta = s['delta_sum'] / max(1, s['delta_count'])
        complexity[name] = {'mean_std': mean_std, 'mean_delta': mean_delta}
    return complexity


def adaptive_codebook_sizes(root: Path, group_map: dict, modalities: dict, max_files: int, max_rows_per_file: int):
    complexity = estimate_stream_complexity(root, group_map, modalities, max_files, max_rows_per_file)
    std_values = [v['mean_std'] for v in complexity.values()]
    delta_values = [v['mean_delta'] for v in complexity.values()]
    std_ref = sorted(std_values)[len(std_values) // 2] if std_values else 1.0
    delta_ref = sorted(delta_values)[len(delta_values) // 2] if delta_values else 1.0
    std_ref = max(std_ref, 1e-6)
    delta_ref = max(delta_ref, 1e-6)

    sizes = {}
    allocation = {}
    for name, channels in modalities.items():
        minimum, maximum = family_bounds(name)
        stats = complexity[name]
        channel_factor = math.sqrt(max(1, channels) / 3.0)
        std_factor = math.sqrt(max(0.25, min(4.0, stats['mean_std'] / std_ref)))
        delta_factor = math.sqrt(max(0.25, min(4.0, stats['mean_delta'] / delta_ref)))
        base = fallback_codebook_size(name)
        size = round_codebook_size(base * channel_factor * 0.5 * (std_factor + delta_factor), minimum, maximum)
        sizes[name] = size
        allocation[name] = {
            'channels': channels,
            'codebook_size': size,
            'mean_std': stats['mean_std'],
            'mean_delta': stats['mean_delta'],
            'fallback_size': base,
            'min': minimum,
            'max': maximum,
        }
    return sizes, allocation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--group_map', default='dataset/Opportunity/group_map_official.json')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--data_fraction', type=float, default=0.10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--checkpoint_dir', default='checkpoints/vqvae_full_scale')
    p.add_argument('--root', default='dataset/Opportunity/extracted/OpportunityUCIDataset/dataset')
    p.add_argument('--device', default='cuda', choices=('auto', 'cuda', 'cpu'))
    p.add_argument('--quantizer', default='ema', choices=('standard', 'ema'))
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--stft_loss_weight', type=float, default=0.03)
    p.add_argument('--wavelet_loss_weight', type=float, default=0.05)
    p.add_argument('--reed_transition_loss_weight', type=float, default=0.20)
    p.add_argument('--activity_contrastive_loss', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--label_contrastive_weight', type=float, default=0.02)
    p.add_argument('--adaptive_codebook', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--complexity_max_files', type=int, default=8)
    p.add_argument('--complexity_max_rows_per_file', type=int, default=5000)
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.group_map, 'r', encoding='utf-8') as fh:
        group_map = json.load(fh)

    modalities = {name: len(cols) for name, cols in group_map.items() if is_requested_family(name)}
    if not modalities:
        raise RuntimeError('No requested sensor groups found in group map.')

    repo_root = Path(__file__).resolve().parents[1]
    data_root = (repo_root / args.root).resolve()
    if args.adaptive_codebook:
        codebook, allocation = adaptive_codebook_sizes(
            root=data_root,
            group_map=group_map,
            modalities=modalities,
            max_files=args.complexity_max_files,
            max_rows_per_file=args.complexity_max_rows_per_file,
        )
    else:
        codebook = {name: fallback_codebook_size(name) for name in modalities}
        allocation = {
            name: {'channels': channels, 'codebook_size': codebook[name], 'fallback_size': codebook[name]}
            for name, channels in modalities.items()
        }

    print(f'Full-scale selected groups: {len(modalities)}')
    print('Families included: acceleration, gyro/angular velocity, reed/contact, mag/compass, shoe, quaternion')
    print('Excluded: labels, MILLISEC, LOCATION_TAG*')
    print(f'Codebook entries total: {sum(codebook.values())}')

    checkpoint_dir = repo_root / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    allocation_path = checkpoint_dir / 'codebook_allocation.json'
    with open(allocation_path, 'w', encoding='utf-8') as fh:
        json.dump({'adaptive': args.adaptive_codebook, 'modalities': allocation}, fh, indent=2)
    print(f'Codebook allocation: {allocation_path}')

    train_script = repo_root / 'scripts' / 'train_shared_vqvae.py'
    cmd = [
        sys.executable,
        str(train_script),
        '--device', args.device,
        '--modalities', ','.join(f'{k}:{v}' for k, v in modalities.items()),
        '--codebook', ','.join(f'{k}:{v}' for k, v in codebook.items()),
        '--batch', str(args.batch),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--beta', str(args.beta),
        '--quantizer', args.quantizer,
        '--num_workers', str(args.num_workers),
        '--data_fraction', str(args.data_fraction),
        '--grad_clip', '1.0',
        '--stft_loss_weight', str(args.stft_loss_weight),
        '--wavelet_loss_weight', str(args.wavelet_loss_weight),
        '--reed_transition_loss_weight', str(args.reed_transition_loss_weight),
        '--encoder_res_blocks', '1',
        '--decoder_res_blocks', '1',
        '--checkpoint_dir', args.checkpoint_dir,
    ]
    if args.activity_contrastive_loss:
        cmd.extend([
            '--activity_contrastive_loss',
            '--label_stream', 'label_HL_Activity',
            '--label_contrastive_weight', str(args.label_contrastive_weight),
        ])
    else:
        cmd.append('--no-activity_contrastive_loss')

    if args.dry_run:
        print(' '.join(cmd))
        return

    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == '__main__':
    main()
