import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import torch
from torch import optim

from dataloaders.opportunity_pytorch_dataset import OpportunityDataset
from dataloaders.opportunity_pytorch_dataset import opportunity_collate
from Code.vqvae.models import MultiModalSharedVQVAE


def stack_batch(items, modalities):
    # items: list of dataset items; modalities: list of modality names
    B = len(items)
    inputs = {}
    for m in modalities:
        arrs = [it['streams'][m] for it in items]
        inputs[m] = torch.stack(arrs, dim=0)  # (B, T, C)
    return inputs


def main():
    root = Path('dataset/Opportunity/extracted/OpportunityUCIDataset/dataset')
    if not root.exists():
        print('Dataset path not found:', root)
        return
    ds = OpportunityDataset(root=str(root), seq_len=32)
    if len(ds) == 0:
        print('No sequences found in dataset')
        return

    # pick a few modalities from first item
    first = ds[0]
    stream_keys = list(first['streams'].keys())
    if len(stream_keys) < 1:
        print('No streams available')
        return
    modalities = stream_keys[:3]
    modality_dims = {m: int(first['streams'][m].size(1)) for m in modalities}
    print('Using modalities:', modalities, 'dims:', modality_dims)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalSharedVQVAE(modality_dims=modality_dims, modality_codebook_sizes={m:64 for m in modalities}, hidden=64, latent_dim=32).to(device)
    # use smaller lr for stability and enable grad clipping
    opt = optim.Adam(model.parameters(), lr=1e-4)

    # prepare initial batch
    B = 4
    items = [ds[i] for i in range(B)]
    inputs = stack_batch(items, modalities)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.train()
    out = model(inputs)
    initial_loss = float(out['total_loss'].detach()) if 'total_loss' in out else sum(float(v['loss'].detach()) for v in out.values())
    print('Initial loss:', initial_loss)

    steps = 50
    for step in range(steps):
        # random batch
        idxs = [random.randrange(len(ds)) for _ in range(B)]
        items = [ds[i] for i in idxs]
        batch_inputs = stack_batch(items, modalities)
        # per-batch per-modality normalization (channels are last dim)
        for k in list(batch_inputs.keys()):
            x = batch_inputs[k]
            mean = x.mean(dim=(0,1), keepdim=True)
            std = x.std(dim=(0,1), keepdim=True) + 1e-6
            batch_inputs[k] = ((x - mean) / std)
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        out = model(batch_inputs)
        loss = out['total_loss'] if 'total_loss' in out else sum(v['loss'] for v in out.values())
        opt.zero_grad()
        loss.backward()
        # gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if (step+1) % 10 == 0:
            print(f'step {step+1}/{steps} loss {float(loss.detach()):.4f}')

    # eval on same initial batch
    model.eval()
    with torch.no_grad():
        out = model(inputs)
        final_loss = float(out['total_loss']) if 'total_loss' in out else sum(float(v['loss']) for v in out.values())
    print('Final loss:', final_loss)


if __name__ == '__main__':
    main()
