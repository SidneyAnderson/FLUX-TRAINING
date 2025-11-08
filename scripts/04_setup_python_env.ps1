#Requires -RunAsAdministrator
###############################################################################
# Python Virtual Environment Setup (Windows 11)
# Creates isolated environment for RTX 5090 FLUX training
###############################################################################

$ErrorActionPreference = "Stop"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "PYTHON ENVIRONMENT SETUP (WINDOWS 11)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$PYTHON_EXE = "C:\Python311\python.exe"
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VENV_DIR = Join-Path $PROJECT_ROOT "venv"

Write-Host "Project Root: $PROJECT_ROOT" -ForegroundColor Gray
Write-Host "Virtual Environment: $VENV_DIR" -ForegroundColor Gray
Write-Host ""

# Verify Python 3.11.9
Write-Host "Verifying Python installation..." -ForegroundColor Yellow

if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "✗ Python not found at: $PYTHON_EXE" -ForegroundColor Red
    Write-Host "  Run: .\scripts\01_install_python.ps1" -ForegroundColor Yellow
    exit 1
}

$pythonVersion = & $PYTHON_EXE --version 2>&1
Write-Host "  $pythonVersion" -ForegroundColor Gray

if ($pythonVersion -notmatch "3\.11\.9") {
    Write-Host "✗ Python 3.11.9 required, found: $pythonVersion" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Python 3.11.9 verified" -ForegroundColor Green
Write-Host ""

# Check if venv already exists
if (Test-Path $VENV_DIR) {
    Write-Host "Virtual environment already exists at: $VENV_DIR" -ForegroundColor Yellow
    $answer = Read-Host "Recreate? (Y/N)"

    if ($answer -eq "Y" -or $answer -eq "y") {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        Remove-Item $VENV_DIR -Recurse -Force
        Write-Host "✓ Removed" -ForegroundColor Green
    } else {
        Write-Host "Using existing environment." -ForegroundColor Green
        Write-Host ""
        Write-Host "To activate:" -ForegroundColor Cyan
        Write-Host "  $VENV_DIR\Scripts\Activate.ps1" -ForegroundColor White
        Write-Host ""
        exit 0
    }
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
& $PYTHON_EXE -m venv $VENV_DIR

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Virtual environment created" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Host "✗ Activation script not found" -ForegroundColor Red
    exit 1
}

# Execute activation in current session
. $activateScript

Write-Host "✓ Activated" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ pip upgrade failed" -ForegroundColor Red
    exit 1
}

$pipVersion = python -m pip --version
Write-Host "  $pipVersion" -ForegroundColor Gray
Write-Host "✓ pip upgraded" -ForegroundColor Green
Write-Host ""

# Install essential build tools
Write-Host "Installing essential build dependencies..." -ForegroundColor Yellow
Write-Host "  This may take 2-3 minutes..." -ForegroundColor Gray
Write-Host ""

$packages = @(
    "setuptools==69.0.3"
    "wheel==0.42.0"
    "ninja==1.11.1.1"
    "cmake==3.28.1"
    "numpy==1.26.4"
    "pyyaml==6.0.1"
    "typing_extensions==4.9.0"
    "Pillow==10.2.0"
)

foreach ($package in $packages) {
    Write-Host "  Installing $package..." -ForegroundColor Gray
    python -m pip install $package --quiet

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Failed to install $package" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✓ Build dependencies installed" -ForegroundColor Green
Write-Host ""

# Install Intel MKL for optimized BLAS
Write-Host "Installing Intel MKL (optimized math library)..." -ForegroundColor Yellow
python -m pip install mkl mkl-include mkl-static mkl-devel --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Intel MKL installed" -ForegroundColor Green
} else {
    Write-Host "⚠ MKL installation failed (not critical)" -ForegroundColor Yellow
}

Write-Host ""

# Verify installation
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "VERIFICATION" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$verifyScript = @'
import sys
import platform
import numpy
import yaml
import PIL

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("NumPy:", numpy.__version__)
print("PyYAML:", yaml.__version__)
print("Pillow:", PIL.__version__)

# Check if in venv
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("\n✓ Running in virtual environment")
else:
    print("\n⚠ Not in virtual environment")

print("\nEnvironment ready for PyTorch compilation!")
'@

python -c $verifyScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host "✅ PYTHON ENVIRONMENT SETUP COMPLETE" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host ""
    Write-Host "Virtual Environment: $VENV_DIR" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To activate in future sessions:" -ForegroundColor White
    Write-Host "  $VENV_DIR\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or from Command Prompt:" -ForegroundColor White
    Write-Host "  $VENV_DIR\Scripts\activate.bat" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "⚠  IMPORTANT: Next steps must be run in" -ForegroundColor Yellow
    Write-Host "   'x64 Native Tools Command Prompt for VS 2022'" -ForegroundColor Yellow
    Write-Host "   with this virtual environment activated" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Next step: .\scripts\05_build_pytorch.ps1" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "✗ Verification failed" -ForegroundColor Red
    exit 1
}
