"""Small demo that creates a MultiModalVQVAE and runs a forward pass with random tensors."""
import torch
from .models import MultiModalVQVAE


def demo():
    # define some example modalities (channel dimensions)
    modalities = {
        'acc': 3,
        'quat': 4,
        'reed': 1,
    }
    mm = MultiModalVQVAE(modality_dims=modalities, hidden=64, latent_dim=32, num_embeddings=128)
    B = 2
    T = 64
    inputs = {
        'acc': torch.randn(B, T, 3),
        'quat': torch.randn(B, T, 4),
        'reed': torch.randn(B, T, 1),
    }
    out = mm(inputs)
    for name, info in out.items():
        print(name, 'recon.shape', info['recon'].shape, 'loss', float(info['loss']))


if __name__ == '__main__':
    demo()
