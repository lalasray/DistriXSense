"""Initialize shared quantizer codebooks using encoder samples and k-means.

This demo uses random data; replace `sample_inputs` with real encoder outputs
collected from your dataset for better initialization.
"""
import torch
from .models import MultiModalSharedVQVAE


def demo_init():
    modalities = {'acc': 3, 'quat': 4, 'reed': 1}
    codebook = {'acc': 128, 'quat': 128, 'reed': 64}
    mm = MultiModalSharedVQVAE(modality_dims=modalities, modality_codebook_sizes=codebook, hidden=64, latent_dim=32)
    # collect samples for each modality by running the encoder on random data
    samples_per_mod = {}
    Batches = 20
    B = 8
    T = 64
    for _ in range(Batches):
        inputs = {
            'acc': torch.randn(B, T, 3),
            'quat': torch.randn(B, T, 4),
            'reed': torch.randn(B, T, 1),
        }
        for name, x in inputs.items():
            z = mm.encoders[name](x)  # (B, T', D)
            bt = z.reshape(-1, z.size(2)).detach()
            samples_per_mod.setdefault(name, []).append(bt)
    for name, chunks in samples_per_mod.items():
        samples = torch.cat(chunks, dim=0)
        # randomly subsample to at most 10000 points
        N = samples.size(0)
        M = min(10000, N)
        idx = torch.randperm(N)[:M]
        mm.quantizer.init_embeddings_kmeans(name, samples[idx], n_iter=20)
        print(f'Initialized codebook for {name} with {M} samples')


if __name__ == '__main__':
    demo_init()
