import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class LearnableTemporalInterpolator(nn.Module):
    """Resample variable-length sequences to a fixed length, then learn a small refinement."""
    def __init__(self, channels: int, target_len: int):
        super().__init__()
        self.target_len = int(target_len)
        self.refine = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, C), lengths: valid frames before padding
        B, T, C = x.shape
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        else:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=T)

        if T == self.target_len and torch.all(lengths == T):
            base = x
        else:
            pieces = []
            for i in range(B):
                valid = x[i:i + 1, :int(lengths[i].item()), :]
                if valid.size(1) == 1:
                    resized = valid.expand(-1, self.target_len, -1)
                else:
                    resized = F.interpolate(
                        valid.permute(0, 2, 1),
                        size=self.target_len,
                        mode='linear',
                        align_corners=False,
                    ).permute(0, 2, 1)
                pieces.append(resized)
            base = torch.cat(pieces, dim=0)

        residual = self.refine(base.permute(0, 2, 1)).permute(0, 2, 1)
        return base + residual


class Encoder1D(nn.Module):
    """Very simple 1D encoder: Conv1d -> ReLU -> Conv1d -> projection to latent_dim."""
    def __init__(self, in_channels: int, hidden: int = 64, latent_dim: int = 32, res_blocks: int = 0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, stride=2)
        self.res_blocks = nn.Sequential(*[ResidualBlock1D(hidden) for _ in range(int(res_blocks))])
        self.conv3 = nn.Conv1d(hidden, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.res_blocks(x)
        z = self.conv3(x)
        # output (B, latent_dim, T') -> (B, T', D)
        z = z.permute(0, 2, 1)
        return z


class Decoder1D(nn.Module):
    """Very simple decoder: maps latent (B, T', D) -> (B, T, C)"""
    def __init__(self, out_channels: int, hidden: int = 64, latent_dim: int = 32, res_blocks: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(latent_dim, hidden, kernel_size=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock1D(hidden) for _ in range(int(res_blocks))])
        self.deconv = nn.ConvTranspose1d(hidden, hidden, kernel_size=4, stride=2, padding=1)
        self.conv_out = nn.Conv1d(hidden, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, target_len: int = None) -> torch.Tensor:
        # z: (B, T', D) -> (B, D, T')
        x = z.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.res_blocks(x)
        x = F.relu(self.deconv(x))
        x = self.conv_out(x)
        # x: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        if target_len is not None and x.size(1) != target_len:
            # simple interpolation in time
            x = F.interpolate(x.permute(0,2,1), size=target_len, mode='linear', align_corners=False).permute(0,2,1)
        return x


class TransformFeatureDecoder(nn.Module):
    """Auxiliary decoder head that predicts transform-domain features from quantized latents."""
    def __init__(self, latent_dim: int, hidden: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T', D)
        return self.net(z.permute(0, 2, 1))


class ResidualBlock1D(nn.Module):
    """Small residual block for 1D temporal features."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LabelConditioner(nn.Module):
    """Small training-only alignment head for sensor latents and activity labels."""
    def __init__(self, modality_names, label_vocab_size: int, latent_dim: int, label_embedding_dim: int = None):
        super().__init__()
        label_embedding_dim = int(label_embedding_dim or latent_dim)
        self.label_embedding = nn.Embedding(label_vocab_size, label_embedding_dim)
        self.latent_projection = nn.Linear(latent_dim, label_embedding_dim)

    def contrastive_loss(self, pooled_latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(dtype=torch.long).clamp(min=0, max=self.label_embedding.num_embeddings - 1)
        sensor_vec = F.normalize(self.latent_projection(pooled_latent), dim=-1)
        label_vec = F.normalize(self.label_embedding(labels), dim=-1)
        logits = sensor_vec @ label_vec.t()
        targets = torch.arange(labels.size(0), device=labels.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


class LearnableLossWeights(nn.Module):
    """Softmax-normalized learnable loss alphas over active loss terms."""
    def __init__(self, names):
        super().__init__()
        self.logits = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(()))
            for name in names
        })

    def alphas(self, names=None):
        names = list(names) if names is not None else list(self.logits.keys())
        if not names:
            return {}
        values = torch.stack([self.logits[name] for name in names])
        weights = torch.softmax(values, dim=0)
        return {name: weights[i] for i, name in enumerate(names)}

    def weighted_sum(self, losses: Dict[str, torch.Tensor]):
        alphas = self.alphas(losses.keys())
        weighted = sum(alphas[name] * loss for name, loss in losses.items())
        return weighted, alphas

    def current_weights(self, names=None):
        alphas = self.alphas(names)
        return {
            name: float(value.detach().cpu())
            for name, value in alphas.items()
        }


def stft_feature_dim(channels: int, seq_len: int, n_fft: int = 16) -> int:
    n_fft = min(int(n_fft), int(seq_len))
    hop_length = max(1, n_fft // 4)
    frames = 1 + max(0, (int(seq_len) - n_fft) // hop_length)
    freqs = n_fft // 2 + 1
    return int(channels) * freqs * frames


def wavelet_feature_dim(channels: int, seq_len: int, levels: int = 2) -> int:
    length = int(seq_len)
    total = 0
    for _ in range(int(levels)):
        if length < 2:
            break
        even_len = length - (length % 2)
        coeff_len = even_len // 2
        total += 2 * coeff_len * int(channels)
        length = coeff_len
    return total


def transition_feature_dim(channels: int, seq_len: int) -> int:
    return max(0, int(seq_len) - 1) * int(channels)


def stft_magnitude_features(x: torch.Tensor, n_fft: int = 16) -> torch.Tensor:
    # STFT magnitudes computed from raw target signal.
    B, T, C = x.shape
    if T < 4:
        return x.new_zeros((B, 0))
    n_fft = min(int(n_fft), T)
    hop_length = max(1, n_fft // 4)
    window = torch.hann_window(n_fft, device=x.device, dtype=torch.float32)
    flat = x.float().permute(0, 2, 1).reshape(B * C, T)
    spec = torch.stft(flat, n_fft=n_fft, hop_length=hop_length, window=window, center=False, return_complex=True)
    mag = torch.abs(spec).reshape(B, C, -1)
    return mag.reshape(B, -1)


def haar_wavelet_features(x: torch.Tensor, levels: int = 2) -> torch.Tensor:
    # Simple Haar approximation/detail coefficients across temporal scales.
    coeff = x.float()
    features = []
    scale = math.sqrt(2.0)
    for _ in range(int(levels)):
        length = coeff.size(1)
        if length < 2:
            break
        even_len = length - (length % 2)
        y = coeff[:, :even_len, :]
        y_even, y_odd = y[:, 0::2, :], y[:, 1::2, :]
        approx, detail = (y_even + y_odd) / scale, (y_even - y_odd) / scale
        features.extend([approx.reshape(x.size(0), -1), detail.reshape(x.size(0), -1)])
        coeff = approx
    if not features:
        return x.new_zeros((x.size(0), 0))
    return torch.cat(features, dim=1)


def transition_features(x: torch.Tensor) -> torch.Tensor:
    # Useful for sparse reed/contact streams: target changes.
    if x.size(1) < 2:
        return x.new_zeros((x.size(0), 0))
    delta = x[:, 1:, :] - x[:, :-1, :]
    return delta.reshape(x.size(0), -1)


class VectorQuantizer(nn.Module):
    """A simple (non-EMA) vector quantizer with straight-through estimator."""
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 32, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.uniform_(self.embeddings, -1.0 / num_embeddings, 1.0 / num_embeddings)

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
        self.embeddings = nn.Parameter(torch.empty(self.total_embeddings, embedding_dim))
        nn.init.uniform_(self.embeddings, -1.0 / self.total_embeddings, 1.0 / self.total_embeddings)

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
        embed = torch.empty(num_embeddings, embedding_dim)
        nn.init.uniform_(embed, -1.0 / num_embeddings, 1.0 / num_embeddings)
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
        embed = torch.empty(self.total_embeddings, embedding_dim)
        nn.init.uniform_(embed, -1.0 / self.total_embeddings, 1.0 / self.total_embeddings)
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
            embed_sum = one_hot.t() @ flat.detach()
            with torch.no_grad():
                cluster = self.cluster_size[start:start+size]
                avg = self.embed_avg[start:start+size]
                cluster.mul_(self.decay).add_(counts * (1 - self.decay))
                avg.mul_(self.decay).add_(embed_sum * (1 - self.decay))
                used = cluster > self.eps
                updated = avg[used] / cluster[used].unsqueeze(1)
                self.embeddings[start:start+size][used].copy_(updated)

        counts = torch.bincount(idx_local, minlength=size).float()
        probs = counts / counts.sum().clamp(min=1.0)
        perplexity = float(torch.exp(-torch.sum(probs * torch.log(probs + 1e-10))))
        zero = commitment.new_zeros(())
        stats = {
            'perplexity': perplexity,
            'counts': counts,
            'vq_loss': loss,
            'codebook_loss': zero,
            'commitment_loss': commitment,
        }
        return z_q_st, loss, idx_global.view(B, T), stats

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
                 hidden: int = 64, latent_dim: int = 32, beta: float = 0.25,
                 input_len: int = None, use_temporal_interpolator: bool = False,
                 quantizer_type: str = 'standard', activity_contrastive_loss: bool = False,
                 label_vocab_size: int = 512, label_embedding_dim: int = None,
                 label_contrastive_weight: float = 0.0,
                 encoder_res_blocks: int = 0, decoder_res_blocks: int = 1,
                 stft_loss_weight: float = 0.0, wavelet_loss_weight: float = 0.0,
                 reed_transition_loss_weight: float = 0.0,
                 learnable_loss_weights: bool = False,
                 modality_loss_reduction: str = 'mean'):
        super().__init__()
        self.input_len = input_len
        self.use_temporal_interpolator = bool(use_temporal_interpolator)
        self.quantizer_type = quantizer_type
        self.activity_contrastive_loss = bool(activity_contrastive_loss)
        self.label_contrastive_weight = float(label_contrastive_weight)
        self.stft_loss_weight = float(stft_loss_weight)
        self.wavelet_loss_weight = float(wavelet_loss_weight)
        self.reed_transition_loss_weight = float(reed_transition_loss_weight)
        self.learnable_loss_weights_enabled = bool(learnable_loss_weights)
        if modality_loss_reduction not in ('mean', 'sum'):
            raise ValueError("modality_loss_reduction must be 'mean' or 'sum'")
        self.modality_loss_reduction = modality_loss_reduction
        self.loss_weighter = None
        if self.learnable_loss_weights_enabled:
            self.loss_weighter = LearnableLossWeights([
                'recon',
                'vq',
                'label',
                'stft',
                'wavelet',
                'reed_transition',
            ])
        # encoders and decoders
        self.temporal_interpolators = nn.ModuleDict()
        if self.use_temporal_interpolator:
            if input_len is None:
                raise ValueError('input_len is required when use_temporal_interpolator=True')
            self.temporal_interpolators = nn.ModuleDict({
                name: LearnableTemporalInterpolator(in_ch, target_len=input_len)
                for name, in_ch in modality_dims.items()
            })
        self.encoders = nn.ModuleDict({
            name: Encoder1D(in_ch, hidden=hidden, latent_dim=latent_dim, res_blocks=encoder_res_blocks)
            for name, in_ch in modality_dims.items()
        })
        self.decoders = nn.ModuleDict({
            name: Decoder1D(out_ch, hidden=hidden, latent_dim=latent_dim, res_blocks=decoder_res_blocks)
            for name, out_ch in modality_dims.items()
        })
        self.stft_decoders = nn.ModuleDict()
        self.wavelet_decoders = nn.ModuleDict()
        self.reed_transition_decoders = nn.ModuleDict()
        if input_len is not None:
            for name, channels in modality_dims.items():
                if name.startswith('REED_'):
                    if self.reed_transition_loss_weight > 0:
                        out_dim = transition_feature_dim(channels, input_len)
                        self.reed_transition_decoders[name] = TransformFeatureDecoder(latent_dim, hidden, out_dim)
                else:
                    if self.stft_loss_weight > 0:
                        out_dim = stft_feature_dim(channels, input_len)
                        self.stft_decoders[name] = TransformFeatureDecoder(latent_dim, hidden, out_dim)
                    if self.wavelet_loss_weight > 0:
                        out_dim = wavelet_feature_dim(channels, input_len)
                        self.wavelet_decoders[name] = TransformFeatureDecoder(latent_dim, hidden, out_dim)
        self.label_conditioner = None
        if self.activity_contrastive_loss:
            self.label_conditioner = LabelConditioner(
                modality_names=list(modality_dims.keys()),
                label_vocab_size=label_vocab_size,
                latent_dim=latent_dim,
                label_embedding_dim=label_embedding_dim,
            )
        if quantizer_type == 'standard':
            self.quantizer = SharedVectorQuantizer(modality_codebook_sizes, embedding_dim=latent_dim, beta=beta)
        elif quantizer_type == 'ema':
            self.quantizer = SharedEMAVectorQuantizer(modality_codebook_sizes, embedding_dim=latent_dim, beta=beta)
        else:
            raise ValueError(f'Unknown quantizer_type: {quantizer_type}')

    def forward(self, inputs: Dict[str, torch.Tensor], input_lengths: Dict[str, torch.Tensor] = None,
                targets: Dict[str, torch.Tensor] = None, labels: torch.Tensor = None) -> Dict[str, Dict]:
        out = {}
        total_loss = 0.0
        active_modalities = 0
        alpha_totals = {}
        for name, x in inputs.items():
            if name not in self.encoders:
                raise KeyError(f"Unknown modality {name}")
            lengths = input_lengths.get(name) if input_lengths else None
            if self.use_temporal_interpolator:
                x = self.temporal_interpolators[name](x, lengths=lengths)
            target = targets.get(name, x) if targets else x
            z_e = self.encoders[name](x)  # (B, T', D)
            label_contrastive_loss = None
            if self.activity_contrastive_loss and labels is not None and self.label_contrastive_weight > 0:
                label_contrastive_loss = self.label_conditioner.contrastive_loss(z_e.mean(dim=1), labels)
            z_q, qloss, idx, stats = self.quantizer.quantize(z_e, modality=name)
            recon = self.decoders[name](z_q, target_len=target.size(1))
            recon_loss = F.mse_loss(recon, target)
            loss_terms = {
                'recon': recon_loss,
                'vq': qloss,
            }
            stft_loss = None
            wavelet_loss = None
            reed_transition = None
            if name.startswith('REED_'):
                if self.reed_transition_loss_weight > 0 and name in self.reed_transition_decoders:
                    pred_transition = self.reed_transition_decoders[name](z_q)
                    target_transition = transition_features(target).detach()
                    reed_transition = F.smooth_l1_loss(pred_transition, target_transition)
                    if self.learnable_loss_weights_enabled:
                        loss_terms['reed_transition'] = self.reed_transition_loss_weight * reed_transition
                    else:
                        loss_terms['reed_transition'] = self.reed_transition_loss_weight * reed_transition
            else:
                if self.stft_loss_weight > 0 and name in self.stft_decoders:
                    pred_stft = self.stft_decoders[name](z_q)
                    target_stft = stft_magnitude_features(target).detach()
                    stft_loss = F.l1_loss(pred_stft, target_stft)
                    if self.learnable_loss_weights_enabled:
                        loss_terms['stft'] = self.stft_loss_weight * stft_loss
                    else:
                        loss_terms['stft'] = self.stft_loss_weight * stft_loss
                if self.wavelet_loss_weight > 0 and name in self.wavelet_decoders:
                    pred_wavelet = self.wavelet_decoders[name](z_q)
                    target_wavelet = haar_wavelet_features(target).detach()
                    wavelet_loss = F.l1_loss(pred_wavelet, target_wavelet)
                    if self.learnable_loss_weights_enabled:
                        loss_terms['wavelet'] = self.wavelet_loss_weight * wavelet_loss
                    else:
                        loss_terms['wavelet'] = self.wavelet_loss_weight * wavelet_loss
            if label_contrastive_loss is not None:
                if self.learnable_loss_weights_enabled:
                    loss_terms['label'] = self.label_contrastive_weight * label_contrastive_loss
                else:
                    loss_terms['label'] = self.label_contrastive_weight * label_contrastive_loss
            if self.learnable_loss_weights_enabled:
                loss, active_alphas = self.loss_weighter.weighted_sum(loss_terms)
                for alpha_name, alpha_value in active_alphas.items():
                    alpha_totals[alpha_name] = alpha_totals.get(alpha_name, 0.0) + alpha_value.detach()
            else:
                loss = sum(loss_terms.values())
            total_loss = total_loss + loss
            active_modalities += 1
            out[name] = {
                'recon': recon,
                'loss': loss,
                'recon_loss': recon_loss,
                'vq_loss': qloss,
                'codebook_loss': stats.get('codebook_loss') if isinstance(stats, dict) else None,
                'commitment_loss': stats.get('commitment_loss') if isinstance(stats, dict) else None,
                'label_contrastive_loss': label_contrastive_loss,
                'stft_loss': stft_loss,
                'wavelet_loss': wavelet_loss,
                'reed_transition_loss': reed_transition,
                'indices': idx,
                'perplexity': stats.get('perplexity') if isinstance(stats, dict) else None,
                'counts': stats.get('counts') if isinstance(stats, dict) else None,
            }
        if self.modality_loss_reduction == 'mean':
            total_loss = total_loss / max(1, active_modalities)
        out['total_loss'] = total_loss
        if self.learnable_loss_weights_enabled:
            out['loss_weights'] = {
                name: float((alpha_totals.get(name, 0.0) / max(1, active_modalities)).detach().cpu())
                for name in self.loss_weighter.logits.keys()
            }
        return out
