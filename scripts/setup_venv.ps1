<#
Windows PowerShell script to create a Python virtual environment and install requirements.
Usage (PowerShell):
  .\setup_venv.ps1 -EnvName .venv
#>

param(
    [string]$EnvName = '.venv'
)

Write-Host "Creating virtual environment: $EnvName"
python -m venv $EnvName
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create venv"; exit 1 }

Write-Host "Activating venv and installing packages"
& "$EnvName\Scripts\python.exe" -m pip install --upgrade pip
& "$EnvName\Scripts\python.exe" -m pip install -r ..\requirements.txt

Write-Host "NOTE: For GPU-enabled PyTorch, install the appropriate torch+cuda wheel per https://pytorch.org/"
Write-Host "Example (CUDA 11.8):"
Write-Host "& $EnvName\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"

Write-Host "Done. Activate the venv with: .\$EnvName\Scripts\Activate.ps1"
