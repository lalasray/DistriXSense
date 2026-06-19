# DistriXSense

Utilities for loading the Opportunity Activity Recognition dataset and training a
multimodal VQ-VAE on sensor streams.

## Setup

Create a virtual environment and install dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1 -EnvName .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA training, make sure your PyTorch install is CUDA-enabled:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Download Data

Try the helper downloader:

```powershell
python scripts\download_opportunity.py --out dataset\Opportunity
```

If the download fails, download the Opportunity dataset manually and extract it
so the `.dat` files are here:

```text
dataset\Opportunity\extracted\OpportunityUCIDataset\dataset\
```

Then run a quick loader check:

```powershell
python scripts\run_opportunity_sanity_one.py
```

## Train VQ-VAE

Architecture sketch and suggested next changes:

```text
docs\vqvae_architecture.md
```

Full CUDA run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_vqvae_cuda.ps1 -Epochs 2 -Batch 8
```

Small CUDA run using 10% of the data and 4 workers:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_vqvae_cuda_10pct.ps1 -Epochs 2 -Batch 8
```

10% run with random frame-drop augmentation and learnable temporal interpolation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_vqvae_cuda_10pct_interp.ps1 -Epochs 2 -Batch 8
```

Add `-Amp` to either command to use CUDA mixed precision.

Checkpoints are written under `checkpoints\`.
