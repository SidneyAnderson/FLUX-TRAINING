#Requires -RunAsAdministrator
###############################################################################
# Build xformers from Source (Windows 11)
# Memory-efficient attention with sm_120 support
###############################################################################

$ErrorActionPreference = "Stop"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "BUILDING XFORMERS - NATIVE SM_120 (WINDOWS 11)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Time estimate: 30-60 minutes" -ForegroundColor Yellow
Write-Host ""

$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$BUILD_DIR = Join-Path $PROJECT_ROOT "build"
$XFORMERS_DIR = Join-Path $BUILD_DIR "xformers"
$VENV_DIR = Join-Path $PROJECT_ROOT "venv"
$PYTHON_EXE = Join-Path $VENV_DIR "Scripts\python.exe"
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"

# Check prerequisites
if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "✗ Python venv not found" -ForegroundColor Red
    exit 1
}

# Setup environment
$env:CUDA_HOME = $CUDA_PATH
$env:TORCH_CUDA_ARCH_LIST = "8.9;9.0;12.0"
$env:MAX_JOBS = [Environment]::ProcessorCount

# Clone xformers
if (-not (Test-Path $XFORMERS_DIR)) {
    Write-Host "Cloning xformers..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $BUILD_DIR -Force | Out-Null
    Push-Location $BUILD_DIR
    git clone https://github.com/facebookresearch/xformers.git
    Pop-Location
}

Push-Location $XFORMERS_DIR
git checkout v0.0.28
git submodule update --init --recursive

Write-Host "Building xformers..." -ForegroundColor Yellow
& $PYTHON_EXE -m pip install -r requirements.txt
& $PYTHON_EXE setup.py build_ext --inplace
& $PYTHON_EXE setup.py install

Pop-Location

# Verify
& $PYTHON_EXE -c "import xformers; print(f'✓ xformers {xformers.__version__}')"

Write-Host ""
Write-Host "✅ XFORMERS BUILD COMPLETE" -ForegroundColor Green
Write-Host "Next: .\scripts\07_build_blackwell_kernels.ps1" -ForegroundColor Cyan
