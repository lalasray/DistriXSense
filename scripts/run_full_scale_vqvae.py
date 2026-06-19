"""Build and launch a full-scale VQ-VAE training command from the Opportunity group map."""
import argparse
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


def codebook_size(name: str) -> int:
    if name.startswith('REED_'):
        return 32
    if 'quat' in name:
        return 128
    if 'SHOE' in name:
        return 96
    return 64


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--group_map', default='dataset/Opportunity/group_map_official.json')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--data_fraction', type=float, default=0.10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--checkpoint_dir', default='checkpoints/vqvae_full_scale')
    p.add_argument('--device', default='cuda', choices=('auto', 'cuda', 'cpu'))
    p.add_argument('--quantizer', default='ema', choices=('standard', 'ema'))
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--stft_loss_weight', type=float, default=0.03)
    p.add_argument('--wavelet_loss_weight', type=float, default=0.05)
    p.add_argument('--reed_transition_loss_weight', type=float, default=0.20)
    p.add_argument('--activity_contrastive_loss', action='store_true')
    p.add_argument('--label_contrastive_weight', type=float, default=0.02)
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.group_map, 'r', encoding='utf-8') as fh:
        group_map = json.load(fh)

    modalities = {name: len(cols) for name, cols in group_map.items() if is_requested_family(name)}
    if not modalities:
        raise RuntimeError('No requested sensor groups found in group map.')

    codebook = {name: codebook_size(name) for name in modalities}

    print(f'Full-scale selected groups: {len(modalities)}')
    print('Families included: acceleration, gyro/angular velocity, reed/contact, mag/compass, shoe, quaternion')
    print('Excluded: labels, MILLISEC, LOCATION_TAG*')

    repo_root = Path(__file__).resolve().parents[1]
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
        '--checkpoint_dir', args.checkpoint_dir,
    ]
    if args.activity_contrastive_loss:
        cmd.extend([
            '--activity_contrastive_loss',
            '--label_stream', 'label_HL_Activity',
            '--label_contrastive_weight', str(args.label_contrastive_weight),
        ])

    if args.dry_run:
        print(' '.join(cmd))
        return

    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == '__main__':
    main()
