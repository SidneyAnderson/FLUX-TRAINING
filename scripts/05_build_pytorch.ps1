#Requires -RunAsAdministrator
###############################################################################
# Build PyTorch from Source with Native sm_120 Support (Windows 11)
# This is the most critical script - compiles PyTorch for Blackwell
###############################################################################

$ErrorActionPreference = "Stop"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "BUILDING PYTORCH FROM SOURCE - NATIVE SM_120 (WINDOWS 11)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠  THIS WILL TAKE 2-4 HOURS" -ForegroundColor Yellow
Write-Host ""

$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$BUILD_DIR = Join-Path $PROJECT_ROOT "build"
$PYTORCH_DIR = Join-Path $BUILD_DIR "pytorch"
$VENV_DIR = Join-Path $PROJECT_ROOT "venv"

# Verify prerequisites
Write-Host "Verifying prerequisites..." -ForegroundColor Yellow
Write-Host ""

# Check Visual Studio
$VS_PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if (-not (Test-Path $VS_PATH)) {
    Write-Host "✗ Visual Studio 2022 Build Tools not found" -ForegroundColor Red
    Write-Host "  Run: .\scripts\03_install_vs2022.ps1" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Visual Studio 2022 Build Tools found" -ForegroundColor Green

# Check CUDA
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
if (-not (Test-Path $CUDA_PATH)) {
    Write-Host "✗ CUDA 13.0 not found" -ForegroundColor Red
    Write-Host "  Run: .\scripts\02_verify_cuda.ps1" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ CUDA 13.0 found" -ForegroundColor Green

# Check Python venv
$PYTHON_EXE = Join-Path $VENV_DIR "Scripts\python.exe"
if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "✗ Python virtual environment not found" -ForegroundColor Red
    Write-Host "  Run: .\scripts\04_setup_python_env.ps1" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Python virtual environment found" -ForegroundColor Green

Write-Host ""

# Check if PyTorch already built
$sitePackages = & $PYTHON_EXE -c "import site; print(site.getsitepackages()[0])"
$torchPath = Join-Path $sitePackages "torch"

if (Test-Path $torchPath) {
    Write-Host "PyTorch already installed. Checking sm_120 support..." -ForegroundColor Yellow
    $hasSmarch = & $PYTHON_EXE -c "import torch; print('sm_120' in str(torch.cuda.get_arch_list()))" 2>$null

    if ($hasSmarch -eq "True") {
        Write-Host "✓ PyTorch with sm_120 already installed" -ForegroundColor Green
        Write-Host ""
        $answer = Read-Host "Rebuild anyway? (Y/N)"

        if ($answer -ne "Y" -and $answer -ne "y") {
            Write-Host "Keeping existing PyTorch installation." -ForegroundColor Green
            Write-Host ""
            Write-Host "Next step: .\scripts\06_build_xformers.ps1" -ForegroundColor Cyan
            exit 0
        }

        Write-Host "Uninstalling existing PyTorch..." -ForegroundColor Yellow
        & $PYTHON_EXE -m pip uninstall -y torch
    } else {
        Write-Host "⚠ Existing PyTorch missing sm_120 support" -ForegroundColor Yellow
        Write-Host "  Uninstalling and rebuilding..." -ForegroundColor Yellow
        & $PYTHON_EXE -m pip uninstall -y torch
    }
}

# Create build directory
New-Item -ItemType Directory -Path $BUILD_DIR -Force | Out-Null

# Clone PyTorch if not already cloned
Write-Host ""
Write-Host "Preparing PyTorch source..." -ForegroundColor Yellow

if (-not (Test-Path $PYTORCH_DIR)) {
    Write-Host "  Cloning PyTorch (this may take 10-15 minutes)..." -ForegroundColor Gray
    Push-Location $BUILD_DIR
    git clone --recursive https://github.com/pytorch/pytorch.git 2>&1 | Out-Null
    Pop-Location

    if (-not (Test-Path $PYTORCH_DIR)) {
        Write-Host "✗ Clone failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ Cloned" -ForegroundColor Green
} else {
    Write-Host "  ✓ PyTorch source already exists" -ForegroundColor Green
}

Push-Location $PYTORCH_DIR

# Checkout specific version
Write-Host "  Checking out v2.5.1..." -ForegroundColor Gray
git checkout v2.5.1 2>&1 | Out-Null
git submodule sync 2>&1 | Out-Null
git submodule update --init --recursive 2>&1 | Out-Null
Write-Host "  ✓ Version v2.5.1 ready" -ForegroundColor Green

Write-Host ""

# Apply sm_120 patches
Write-Host "Applying Blackwell (sm_120) patches..." -ForegroundColor Yellow
Write-Host ""

# Patch cmake/public/cuda.cmake
$cudaCmake = "cmake\public\cuda.cmake"
$cudaCmakeContent = Get-Content $cudaCmake -Raw

if ($cudaCmakeContent -notmatch "sm_120") {
    Write-Host "  Patching cuda.cmake..." -ForegroundColor Gray
    $patch = @"

# Add Blackwell (sm_120) support for RTX 5090
if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
  list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_120,code=sm_120")
endif()
"@
    Add-Content -Path $cudaCmake -Value $patch
    Write-Host "  ✓ cuda.cmake patched" -ForegroundColor Green
} else {
    Write-Host "  ✓ cuda.cmake already patched" -ForegroundColor Green
}

# Patch torch/utils/cpp_extension.py
$cppExtension = "torch\utils\cpp_extension.py"
if (Test-Path $cppExtension) {
    $cppExtContent = Get-Content $cppExtension -Raw

    if ($cppExtContent -notmatch "12.0") {
        Write-Host "  Patching cpp_extension.py..." -ForegroundColor Gray
        $cppExtContent = $cppExtContent -replace "(arch_list\.append\('9\.0\+PTX'\))", "`$1`n    if int(cuda_version_major) >= 13:`n        arch_list.append('12.0')"
        Set-Content -Path $cppExtension -Value $cppExtContent
        Write-Host "  ✓ cpp_extension.py patched" -ForegroundColor Green
    } else {
        Write-Host "  ✓ cpp_extension.py already patched" -ForegroundColor Green
    }
}

Write-Host ""

# Setup build environment
Write-Host "Configuring build environment..." -ForegroundColor Yellow
Write-Host ""

# Visual Studio environment
Write-Host "  Setting up Visual Studio environment..." -ForegroundColor Gray
$vsVarsPath = "$VS_PATH\VC\Auxiliary\Build\vcvars64.bat"

# Call vcvars64.bat and capture environment
$vcvarsCmd = "`"$vsVarsPath`" && set"
$envVars = cmd /c $vcvarsCmd

# Parse and set environment variables
foreach ($line in $envVars) {
    if ($line -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

Write-Host "  ✓ Visual Studio environment configured" -ForegroundColor Green

# CUDA environment
Write-Host "  Setting up CUDA environment..." -ForegroundColor Gray
$env:CUDA_HOME = $CUDA_PATH
$env:CUDA_PATH = $CUDA_PATH
$env:CUDA_PATH_V13_0 = $CUDA_PATH
$env:PATH = "$CUDA_PATH\bin;$CUDA_PATH\libnvvp;" + $env:PATH
$env:CUDNN_LIB_DIR = "$CUDA_PATH\lib\x64"
$env:CUDNN_INCLUDE_DIR = "$CUDA_PATH\include"
Write-Host "  ✓ CUDA environment configured" -ForegroundColor Green

# PyTorch build configuration
Write-Host "  Setting build flags..." -ForegroundColor Gray

$env:BUILD_TEST = "0"
$env:BUILD_CAFFE2 = "0"
$env:USE_CUDA = "1"
$env:USE_CUDNN = "1"
$env:USE_CUBLAS = "1"
$env:USE_CUFFT = "1"
$env:USE_CURAND = "1"
$env:USE_CUSPARSE = "1"
$env:USE_CUSOLVER = "1"
$env:USE_NCCL = "1"
$env:USE_MKLDNN = "1"
$env:TORCH_CUDA_ARCH_LIST = "8.9;9.0;12.0"
$env:TORCH_NVCC_FLAGS = "-Xcompiler /MD -gencode=arch=compute_120,code=sm_120 --allow-unsupported-compiler"
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_BUILD_TYPE = "Release"
$env:MAX_JOBS = [Environment]::ProcessorCount
$env:DISTUTILS_USE_SDK = "1"
$env:MSSdk = "1"

Write-Host "  ✓ Build flags configured" -ForegroundColor Green

Write-Host ""
Write-Host "Build Configuration:" -ForegroundColor Cyan
Write-Host "  CUDA: $env:CUDA_HOME" -ForegroundColor Gray
Write-Host "  CUDA Architectures: $env:TORCH_CUDA_ARCH_LIST" -ForegroundColor Gray
Write-Host "  Max Jobs: $env:MAX_JOBS" -ForegroundColor Gray
Write-Host "  Generator: $env:CMAKE_GENERATOR" -ForegroundColor Gray
Write-Host ""

# Start build
Write-Host "=" * 80 -ForegroundColor Yellow
Write-Host "STARTING PYTORCH BUILD" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Yellow
Write-Host ""
Write-Host "⏰ This will take 2-4 hours depending on your CPU" -ForegroundColor Yellow
Write-Host "☕ Good time for a coffee break!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host ""

$answer = Read-Host "Start PyTorch compilation? (Y/N)"
if ($answer -ne "Y" -and $answer -ne "y") {
    Write-Host "Build cancelled." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Building..." -ForegroundColor Yellow
Write-Host ""

# Clean previous builds
& $PYTHON_EXE setup.py clean 2>&1 | Out-Null

# Build and install
& $PYTHON_EXE setup.py develop

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ PyTorch build failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  • Not enough RAM (need 64GB)" -ForegroundColor Gray
    Write-Host "  • Not enough disk space" -ForegroundColor Gray
    Write-Host "  • Visual Studio environment not set" -ForegroundColor Gray
    Write-Host "  • CUDA mismatch" -ForegroundColor Gray
    Write-Host ""
    exit 1
}

Pop-Location

# Verify build
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "VERIFICATION" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$verifyScript = @'
import torch
import sys

print("=" * 80)
print("PYTORCH BUILD VERIFICATION")
print("=" * 80)
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Architecture list: {torch.cuda.get_arch_list()}")

    # Check for sm_120
    arch_list = str(torch.cuda.get_arch_list())
    if 'sm_120' in arch_list:
        print("\n✅ Native sm_120 support confirmed!")

        # Additional checks
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {cap[0]}.{cap[1]}")

        if cap == (12, 0):
            print("✓ Blackwell (sm_120) GPU detected")

        # Test BF16
        try:
            a = torch.randn(1000, 1000, dtype=torch.bfloat16, device='cuda')
            b = torch.randn(1000, 1000, dtype=torch.bfloat16, device='cuda')
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            print("✓ BF16 computation working")
        except Exception as e:
            print(f"⚠ BF16 test failed: {e}")

        sys.exit(0)
    else:
        print("\n❌ sm_120 NOT in architecture list!")
        print(f"Found: {arch_list}")
        sys.exit(1)
else:
    print("\n❌ CUDA not available in PyTorch!")
    sys.exit(1)
'@

& $PYTHON_EXE -c $verifyScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host "✅ PYTORCH BUILD COMPLETE WITH NATIVE SM_120!" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host ""
    Write-Host "Finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""
    Write-Host "PyTorch built with:" -ForegroundColor Cyan
    Write-Host "  • Native Blackwell (sm_120) support" -ForegroundColor White
    Write-Host "  • CUDA 13.0 integration" -ForegroundColor White
    Write-Host "  • BF16 tensor cores enabled" -ForegroundColor White
    Write-Host ""
    Write-Host "Next step: .\scripts\06_build_xformers.ps1" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "✗ PyTorch verification failed" -ForegroundColor Red
    Write-Host "  Build completed but sm_120 support not detected" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
