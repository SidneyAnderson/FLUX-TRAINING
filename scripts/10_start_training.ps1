###############################################################################
# Start FLUX Training (Windows 11)
###############################################################################

$ErrorActionPreference = "Stop"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "STARTING RTX 5090 NATIVE FLUX TRAINING" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VENV_DIR = Join-Path $PROJECT_ROOT "venv"
$CONFIG_FILE = Join-Path $PROJECT_ROOT "config\rtx5090_native.toml"
$SD_SCRIPTS_DIR = Join-Path $PROJECT_ROOT "sd-scripts"

# Pre-flight checks
Write-Host ""
Write-Host "Pre-flight checks..." -ForegroundColor Yellow

if (-not (Test-Path $CONFIG_FILE)) {
    Write-Host "✗ Config file not found: $CONFIG_FILE" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Config file found" -ForegroundColor Green

if (-not (Test-Path "$PROJECT_ROOT\dataset")) {
    Write-Host "✗ Dataset directory not found" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dataset directory found" -ForegroundColor Green

if (-not (Test-Path "$PROJECT_ROOT\models")) {
    Write-Host "✗ Models directory not found" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Models directory found" -ForegroundColor Green

# Set environment
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,garbage_collection_threshold:0.95"
$env:CUDNN_BENCHMARK = "1"

# Create output directories
New-Item -ItemType Directory -Path "$PROJECT_ROOT\output" -Force | Out-Null
New-Item -ItemType Directory -Path "$PROJECT_ROOT\logs" -Force | Out-Null

Write-Host ""
Write-Host "Starting training..." -ForegroundColor Yellow
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host ""

Push-Location $SD_SCRIPTS_DIR
& "$VENV_DIR\Scripts\python.exe" flux_train_network.py --config_file $CONFIG_FILE --sample_at_first --highvram
Pop-Location

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "TRAINING COMPLETED" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "Finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host ""
Write-Host "Output: $PROJECT_ROOT\output" -ForegroundColor Cyan
Write-Host "Logs: $PROJECT_ROOT\logs" -ForegroundColor Cyan
