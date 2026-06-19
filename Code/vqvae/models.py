import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class Encoder1D(nn.Module):
    """Very simple 1D encoder: Conv1d -> ReLU -> Conv1d -> projection to latent_dim."""
    def __init__(self, in_channels: int, hidden: int = 64, latent_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(hidden, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        z = self.conv3(x)
        # output (B, latent_dim, T') -> (B, T', D)
        z = z.permute(0, 2, 1)
        return z


class Decoder1D(nn.Module):
    """Very simple decoder: maps latent (B, T', D) -> (B, T, C)"""
    def __init__(self, out_channels: int, hidden: int = 64, latent_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(latent_dim, hidden, kernel_size=1)
        self.deconv = nn.ConvTranspose1d(hidden, hidden, kernel_size=4, stride=2, padding=1)
        self.conv_out = nn.Conv1d(hidden, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, target_len: int = None) -> torch.Tensor:
        # z: (B, T', D) -> (B, D, T')
        x = z.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.deconv(x))
        x = self.conv_out(x)
        # x: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        if target_len is not None and x.size(1) != target_len:
            # simple interpolation in time
            x = F.interpolate(x.permute(0,2,1), size=target_len, mode='linear', align_corners=False).permute(0,2,1)
        return x


class VectorQuantizer(nn.Module):
    """A simple (non-EMA) vector quantizer with straight-through estimator."""
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 32, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # z_e: (B, T, D)
        B, T, D = z_e.shape
        flat = z_e.reshape(-1, D)  # (B*T, D)
        # compute distances
        dist = torch.sum(flat**2, dim=1, keepdim=True) - 2 * flat @ self.embeddings.t() + torch.sum(self.embeddings**2, dim=1)
        idx = torch.argmin(dist, dim=1)
        z_q = self.embeddings[idx].view(B, T, D)
        # losses
        commitment = F.mse_loss(z_q.detach(), z_e)
        codebook = F.mse_loss(z_q, z_e.detach())
        loss = codebook + self.beta * commitment
        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        losses = {'vq_loss': loss, 'codebook_loss': codebook, 'commitment_loss': commitment}
        return z_q_st, loss, idx.view(B, T), losses


class SharedVectorQuantizer(nn.Module):
    """Shared quantizer with modality-specific contiguous slices of a global codebook.

    `modality_codebook_sizes` is a dict mapping modality name -> number of embeddings.
    Embeddings are stored in a single parameter; each modality uses a contiguous slice.
    """
    def __init__(self, modality_codebook_sizes: Dict[str, int], embedding_dim: int = 32, beta: float = 0.25):
        super().__init__()
        self.modality_codebook_sizes = dict(modality_codebook_sizes)
        self.embedding_dim = embedding_dim
        self.beta = beta
        # compute offsets
        self.offsets = {}
        total = 0
        for k, v in self.modality_codebook_sizes.items():
            self.offsets[k] = total
            total += int(v)
        self.total_embeddings = total
        self.embeddings = nn.Parameter(torch.randn(self.total_embeddings, embedding_dim))

    def quantize(self, z_e: torch.Tensor, modality: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Quantize `z_e` (B, T, D) using the codebook slice for `modality`.

        Returns (z_q_st, loss, indices_global)
        """
        if modality not in self.modality_codebook_sizes:
            raise KeyError(f"Modality {modality} not configured in SharedVectorQuantizer")
        start = self.offsets[modality]
        size = int(self.modality_codebook_sizes[modality])
        emb_slice = self.embeddings[start:start+size]  # (M, D)
        B, T, D = z_e.shape
        flat = z_e.reshape(-1, D)  # (B*T, D)
        # distances: using (a-b)^2 = a^2 - 2ab + b^2
        dist = torch.sum(flat**2, dim=1, keepdim=True) - 2 * flat @ emb_slice.t() + torch.sum(emb_slice**2, dim=1)
        idx_local = torch.argmin(dist, dim=1)
        idx_global = idx_local + start
        z_q = emb_slice[idx_local].view(B, T, D)
        commitment = F.mse_loss(z_q.detach(), z_e)
        codebook = F.mse_loss(z_q, z_e.detach())
        loss = codebook + self.beta * commitment
        z_q_st = z_e + (z_q - z_e).detach()
        # compute usage/perplexity for this slice
        # histogram over local indices
        counts = torch.bincount(idx_local, minlength=size).float()
        probs = counts / counts.sum().clamp(min=1.0)
        # perplexity = exp(-sum(p log p))
        perplexity = float(torch.exp(-torch.sum(probs * torch.log(probs + 1e-10))))
        stats = {
            'perplexity': perplexity,
            'counts': counts,
            'vq_loss': loss,
            'codebook_loss': codebook,
            'commitment_loss': commitment,
        }
        return z_q_st, loss, idx_global.view(B, T), stats

    def init_embeddings_kmeans(self, modality: str, samples: torch.Tensor, n_iter: int = 20, seed: int = 0):
        """Initialize the modality's codebook slice using k-means on `samples`.

        `samples`: (N, D) torch tensor or numpy array. This runs a simple Lloyd's k-means.
        """
        import numpy as _np
        if modality not in self.modality_codebook_sizes:
            raise KeyError(f"Modality {modality} not configured")
        start = self.offsets[modality]
        K = int(self.modality_codebook_sizes[modality])
        if isinstance(samples, torch.Tensor):
            samples_np = samples.detach().cpu().numpy()
        else:
            samples_np = _np.asarray(samples)
        N, D = samples_np.shape
        if D != self.embedding_dim:
            raise ValueError(f"Samples dim {D} doesn't match embedding_dim {self.embedding_dim}")
        rng = _np.random.RandomState(seed)
        # initialize centers by random sampling from data
        if N < K:
            # pad centers by random normal
            centers = rng.normal(size=(K, D)).astype(_np.float32)
            centers[:N] = samples_np[:N]
        else:
            idx = rng.choice(N, K, replace=False)
            centers = samples_np[idx].astype(_np.float32)
        for it in range(n_iter):
            # assign
            d = _np.sum((samples_np[:, None, :] - centers[None, :, :])**2, axis=2)
            assign = _np.argmin(d, axis=1)
            new_centers = _np.zeros_like(centers)
            for k in range(K):
                members = samples_np[assign == k]
                if len(members) == 0:
                    new_centers[k] = centers[k]
                else:
                    new_centers[k] = members.mean(axis=0)
            centers = new_centers
        # write centers into embedding slice
        with torch.no_grad():
            self.embeddings[start:start+K].copy_(torch.from_numpy(centers))

    def usage_stats(self, indices: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Return usage stats for the entire codebook or for provided indices.

        If `indices` is provided (B, T), computes bincount over global indices.
        """
        if indices is None:
            # no usage available
            return {'total_embeddings': self.total_embeddings}
        flat = indices.view(-1)
        counts = torch.bincount(flat, minlength=self.total_embeddings).float()
        return {'counts': counts}


class EMAVectorQuantizer(nn.Module):
    """EMA vector quantizer (van den Oord et al.) single-modality version."""
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 32, decay: float = 0.99, eps: float = 1e-5, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        self.beta = beta
        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embeddings', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', torch.zeros(num_embeddings, embedding_dim))

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: (B, T, D)
        B, T, D = z_e.shape
        flat = z_e.reshape(-1, D)  # (N, D)
        # compute distances
        dist = torch.sum(flat**2, dim=1, keepdim=True) - 2 * flat @ self.embeddings.t() + torch.sum(self.embeddings**2, dim=1)
        idx = torch.argmin(dist, dim=1)
        z_q = self.embeddings[idx].view(B, T, D)
        # commitment loss only
        commitment = F.mse_loss(z_q.detach(), z_e)
        loss = self.beta * commitment
        z_q_st = z_e + (z_q - z_e).detach()

        # EMA updates (only in training mode)
        if self.training:
            # one-hot assignments
            one_hot = F.one_hot(idx, num_classes=self.num_embeddings).float()  # (N, M)
            counts = one_hot.sum(dim=0)
            embed_sum = one_hot.t() @ flat  # (M, D)
            # update buffers
            self.cluster_size.mul_(self.decay).add_(counts * (1 - self.decay))
            self.embed_avg.mul_(self.decay).add_(embed_sum * (1 - self.decay))
            n = self.cluster_size.sum()
            # normalize to get embeddings
            n_cluster = (self.cluster_size + self.eps)
            embeddings = self.embed_avg / n_cluster.unsqueeze(1)
            self.embeddings.copy_(embeddings)

        return z_q_st, loss, idx.view(B, T)


class SharedEMAVectorQuantizer(nn.Module):
    """Shared EMA quantizer partitioned per-modality (contiguous slices)."""
    def __init__(self, modality_codebook_sizes: Dict[str, int], embedding_dim: int = 32, decay: float = 0.99, eps: float = 1e-5, beta: float = 0.25):
        super().__init__()
        self.modality_codebook_sizes = dict(modality_codebook_sizes)
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        self.beta = beta
        self.offsets = {}
        total = 0
        for k, v in self.modality_codebook_sizes.items():
            self.offsets[k] = total
            total += int(v)
        self.total_embeddings = total
        embed = torch.randn(self.total_embeddings, embedding_dim)
        self.register_buffer('embeddings', embed)
        self.register_buffer('cluster_size', torch.zeros(self.total_embeddings))
        self.register_buffer('embed_avg', torch.zeros(self.total_embeddings, embedding_dim))

    def quantize(self, z_e: torch.Tensor, modality: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if modality not in self.modality_codebook_sizes:
            raise KeyError(f"Modality {modality} not configured in SharedEMAVectorQuantizer")
        start = self.offsets[modality]
        size = int(self.modality_codebook_sizes[modality])
        emb_slice = self.embeddings[start:start+size]
        B, T, D = z_e.shape
        flat = z_e.reshape(-1, D)
        dist = torch.sum(flat**2, dim=1, keepdim=True) - 2 * flat @ emb_slice.t() + torch.sum(emb_slice**2, dim=1)
        idx_local = torch.argmin(dist, dim=1)
        idx_global = idx_local + start
        z_q = emb_slice[idx_local].view(B, T, D)
        commitment = F.mse_loss(z_q.detach(), z_e)
        loss = self.beta * commitment
        z_q_st = z_e + (z_q - z_e).detach()

        if self.training:
            one_hot = F.one_hot(idx_local, num_classes=size).float()
            counts = one_hot.sum(dim=0)
            embed_sum = one_hot.t() @ flat
            # update global buffers
            self.cluster_size[start:start+size].mul_(self.decay).add_(counts * (1 - self.decay))
            self.embed_avg[start:start+size].mul_(self.decay).add_(embed_sum * (1 - self.decay))
            n_cluster = (self.cluster_size[start:start+size] + self.eps)
            new_emb = self.embed_avg[start:start+size] / n_cluster.unsqueeze(1)
            self.embeddings[start:start+size].copy_(new_emb)

        counts = torch.bincount(idx_local, minlength=size).float()
        probs = counts / counts.sum().clamp(min=1.0)
        perplexity = float(torch.exp(-torch.sum(probs * torch.log(probs + 1e-10))))
        return z_q_st, loss, idx_global.view(B, T), {'perplexity': perplexity, 'counts': counts}

    def init_embeddings_kmeans(self, modality: str, samples: torch.Tensor, n_iter: int = 20, seed: int = 0):
        # reuse previous implementation for k-means init
        if modality not in self.modality_codebook_sizes:
            raise KeyError(f"Modality {modality} not configured")
        start = self.offsets[modality]
        K = int(self.modality_codebook_sizes[modality])
        if isinstance(samples, torch.Tensor):
            samples_np = samples.detach().cpu().numpy()
        else:
            import numpy as _np
            samples_np = _np.asarray(samples)
        N, D = samples_np.shape
        import numpy as _np
        rng = _np.random.RandomState(seed)
        if N < K:
            centers = rng.normal(size=(K, D)).astype(_np.float32)
            centers[:N] = samples_np[:N]
        else:
            idx = rng.choice(N, K, replace=False)
            centers = samples_np[idx].astype(_np.float32)
        for it in range(n_iter):
            d = _np.sum((samples_np[:, None, :] - centers[None, :, :])**2, axis=2)
            assign = _np.argmin(d, axis=1)
            new_centers = _np.zeros_like(centers)
            for k in range(K):
                members = samples_np[assign == k]
                if len(members) == 0:
                    new_centers[k] = centers[k]
                else:
                    new_centers[k] = members.mean(axis=0)
            centers = new_centers
        with torch.no_grad():
            self.embeddings[start:start+K].copy_(torch.from_numpy(centers))



class VQVAE(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, latent_dim: int = 32, num_embeddings: int = 256):
        super().__init__()
        self.encoder = Encoder1D(in_channels, hidden=hidden, latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim)
        self.decoder = Decoder1D(out_channels=in_channels, hidden=hidden, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, T, C)
        z_e = self.encoder(x)  # (B, T', D)
        z_q, qloss, idx, qstats = self.quantizer(z_e)
        recon = self.decoder(z_q, target_len=x.size(1))
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + qloss
        return {
            'recon': recon,
            'loss': loss,
            'recon_loss': recon_loss,
            'vq_loss': qloss,
            'codebook_loss': qstats['codebook_loss'],
            'commitment_loss': qstats['commitment_loss'],
            'indices': idx,
        }


class MultiModalVQVAE(nn.Module):
    """Original multimodal container (each modality has its own VQVAE instance).

    Kept for backward compatibility.
    """
    def __init__(self, modality_dims: Dict[str, int], hidden: int = 64, latent_dim: int = 32, num_embeddings: int = 256):
        super().__init__()
        self.modalities = nn.ModuleDict({
            name: VQVAE(in_ch, hidden=hidden, latent_dim=latent_dim, num_embeddings=num_embeddings)
            for name, in_ch in modality_dims.items()
        })

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        out = {}
        for name, x in inputs.items():
            if name not in self.modalities:
                raise KeyError(f"Unknown modality {name}")
            out[name] = self.modalities[name](x)
        return out


class MultiModalSharedVQVAE(nn.Module):
    """Multimodal VQ-VAE with multiple encoders, a single shared quantizer (partitioned per modality), and multiple decoders.

    `modality_codebook_sizes` maps modality name -> number of embeddings in its slice.
    """
    def __init__(self, modality_dims: Dict[str, int], modality_codebook_sizes: Dict[str, int],
                 hidden: int = 64, latent_dim: int = 32, beta: float = 0.25):
        super().__init__()
        # encoders and decoders
        self.encoders = nn.ModuleDict({
            name: Encoder1D(in_ch, hidden=hidden, latent_dim=latent_dim)
            for name, in_ch in modality_dims.items()
        })
        self.decoders = nn.ModuleDict({
            name: Decoder1D(out_ch, hidden=hidden, latent_dim=latent_dim)
            for name, out_ch in modality_dims.items()
        })
        # shared quantizer
        self.quantizer = SharedVectorQuantizer(modality_codebook_sizes, embedding_dim=latent_dim, beta=beta)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        out = {}
        total_loss = 0.0
        for name, x in inputs.items():
            if name not in self.encoders:
                raise KeyError(f"Unknown modality {name}")
            z_e = self.encoders[name](x)  # (B, T', D)
            z_q, qloss, idx, stats = self.quantizer.quantize(z_e, modality=name)
            recon = self.decoders[name](z_q, target_len=x.size(1))
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + qloss
            total_loss = total_loss + loss
            out[name] = {
                'recon': recon,
                'loss': loss,
                'recon_loss': recon_loss,
                'vq_loss': qloss,
                'codebook_loss': stats.get('codebook_loss') if isinstance(stats, dict) else None,
                'commitment_loss': stats.get('commitment_loss') if isinstance(stats, dict) else None,
                'indices': idx,
                'perplexity': stats.get('perplexity') if isinstance(stats, dict) else None,
                'counts': stats.get('counts') if isinstance(stats, dict) else None,
            }
        out['total_loss'] = total_loss
        return out
