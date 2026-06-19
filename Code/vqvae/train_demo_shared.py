"""Demo for MultiModalSharedVQVAE using a shared partitioned quantizer."""
import torch
from .models import MultiModalSharedVQVAE


def demo():
    modalities = {'acc': 3, 'quat': 4, 'reed': 1}
    # allocate codebook sizes per modality
    codebook = {'acc': 128, 'quat': 128, 'reed': 64}
    mm = MultiModalSharedVQVAE(modality_dims=modalities, modality_codebook_sizes=codebook, hidden=64, latent_dim=32)
    B = 2
    T = 64
    inputs = {
        'acc': torch.randn(B, T, 3),
        'quat': torch.randn(B, T, 4),
        'reed': torch.randn(B, T, 1),
    }
    out = mm(inputs)
    for name, info in out.items():
        if name == 'total_loss':
            print('total_loss', float(info))
            continue
        print(name, 'recon.shape', info['recon'].shape, 'loss', float(info['loss']), 'indices.shape', info['indices'].shape)


if __name__ == '__main__':
    demo()
