"""Collect encoder outputs from OpportunityDataset and initialize shared quantizer codebooks."""
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.opportunity_pytorch_dataset import OpportunityDataset
from Code.vqvae.models import MultiModalSharedVQVAE


def collect_encoder_samples(root, modalities, sample_per_mod=5000, batch_size=8, max_batches=200):
    ds = OpportunityDataset(root=root, seq_len=32)
    dl = DataLoader(ds, batch_size=batch_size)
    collectors = {m: [] for m in modalities}
    collected = {m: 0 for m in modalities}
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        streams = batch['streams']
        for m, ch in modalities.items():
            if m not in streams:
                continue
            # take first batch example across time and channels as samples
            # we'll run encoder below; here just collect raw streams
            t = streams[m]  # Tensor (B, T, C)
            # convert to float
            t = t.float()
            collectors[m].append(t)
            collected[m] += t.size(0) * t.size(1)
            if collected[m] >= sample_per_mod:
                pass
        if all(collected[m] >= sample_per_mod for m in modalities):
            break
    # concatenate and reduce
    samples = {}
    for m in modalities:
        if len(collectors[m]) == 0:
            samples[m] = None
            continue
        x = torch.cat(collectors[m], dim=0)  # (N, T, C)
        B, T, C = x.shape
        z = x.reshape(-1, C)
        # subsample
        N = z.size(0)
        M = min(sample_per_mod, N)
        idx = torch.randperm(N)[:M]
        samples[m] = z[idx]
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='dataset/Opportunity/extracted/OpportunityUCIDataset/dataset')
    p.add_argument('--modalities', default='acc:3,quat:4,reed:1')
    p.add_argument('--codebook', default='acc:128,quat:128,reed:64')
    p.add_argument('--out', default='checkpoints/init_codebook.pt')
    args = p.parse_args()

    modalities = {}
    for item in args.modalities.split(','):
        k, v = item.split(':')
        modalities[k] = int(v)
    codebook = {}
    for item in args.codebook.split(','):
        k, v = item.split(':')
        codebook[k] = int(v)

    mm = MultiModalSharedVQVAE(modality_dims=modalities, modality_codebook_sizes=codebook, hidden=64, latent_dim=32)

    samples = collect_encoder_samples(args.root, modalities)
    # run encoders on samples to get latent z_e
    for m in modalities:
        if samples.get(m) is None:
            print('No samples for', m)
            continue
        # samples[m] is (M, C) raw; reshape to (B, T, C) with T=1 for encoder
        data = samples[m].unsqueeze(1)  # (M,1,C)
        with torch.no_grad():
            z = mm.encoders[m](data)  # (M, T', D)
            zflat = z.reshape(-1, z.size(2))
        print('Initializing codebook for', m, 'with', zflat.size(0), 'samples')
        mm.quantizer.init_embeddings_kmeans(m, zflat, n_iter=20)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'embeddings': mm.quantizer.embeddings}, args.out)
    print('Saved initialized embeddings to', args.out)


if __name__ == '__main__':
    main()
