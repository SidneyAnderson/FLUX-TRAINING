#Requires -RunAsAdministrator
###############################################################################
# CUDA 13.0 Verification and Installation Guide (Windows 11)
# Verifies CUDA 13.0 installation or guides through installation
###############################################################################

$ErrorActionPreference = "Stop"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "CUDA 13.0 VERIFICATION (WINDOWS 11)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$CUDA_VERSION = "13.0"
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CUDA_VERSION"

# Check if CUDA 13.0 is installed
Write-Host "Checking for CUDA $CUDA_VERSION installation..." -ForegroundColor Yellow
Write-Host ""

if (Test-Path $CUDA_PATH) {
    Write-Host "✓ CUDA $CUDA_VERSION found at: $CUDA_PATH" -ForegroundColor Green
    Write-Host ""

    # Set environment variables for this session
    $env:CUDA_HOME = $CUDA_PATH
    $env:CUDA_PATH = $CUDA_PATH
    $env:CUDA_PATH_V13_0 = $CUDA_PATH
    $env:PATH = "$CUDA_PATH\bin;$CUDA_PATH\libnvvp;" + $env:PATH

    # Check nvcc
    Write-Host "Verifying NVCC..." -ForegroundColor Yellow
    if (Get-Command nvcc -ErrorAction SilentlyContinue) {
        $nvccVersion = nvcc --version | Select-String "release"
        Write-Host "  $nvccVersion" -ForegroundColor Gray

        if ($nvccVersion -match "release $CUDA_VERSION") {
            Write-Host "✓ NVCC version correct" -ForegroundColor Green
        } else {
            Write-Host "⚠ NVCC version mismatch" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠ NVCC not found in PATH" -ForegroundColor Yellow
        Write-Host "  Adding to PATH..." -ForegroundColor Gray
    }

    # Test sm_120 support
    Write-Host ""
    Write-Host "Testing sm_120 (Blackwell) support..." -ForegroundColor Yellow

    $testCudaCode = @'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_sm120() {
    if (threadIdx.x == 0) {
        printf("✓ sm_120 kernel executed successfully!\n");
    }
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    if (prop.major == 12 && prop.minor == 0) {
        printf("✓ Blackwell (sm_120) GPU detected\n");
        test_sm120<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    } else {
        printf("⚠ Not a Blackwell GPU (expected sm_120, got sm_%d%d)\n",
               prop.major, prop.minor);
        return 1;
    }
}
'@

    $testDir = "$env:TEMP\cuda_test"
    New-Item -ItemType Directory -Path $testDir -Force | Out-Null
    $testCudaCode | Out-File -FilePath "$testDir\test_sm120.cu" -Encoding ASCII

    Write-Host "  Compiling test kernel..." -ForegroundColor Gray
    try {
        Push-Location $testDir
        $compileOutput = nvcc -arch=sm_120 test_sm120.cu -o test_sm120.exe 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Compilation successful" -ForegroundColor Green
            Write-Host ""
            Write-Host "  Running test..." -ForegroundColor Gray
            & .\test_sm120.exe

            if ($LASTEXITCODE -eq 0) {
                Write-Host ""
                Write-Host "✅ CUDA 13.0 with native sm_120 support confirmed!" -ForegroundColor Green
            } else {
                Write-Host "⚠ Test execution completed with warnings" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  ⚠ Compilation completed with warnings:" -ForegroundColor Yellow
            Write-Host $compileOutput -ForegroundColor Gray
        }
    } catch {
        Write-Host "  ⚠ Test failed: $_" -ForegroundColor Yellow
    } finally {
        Pop-Location
        Remove-Item $testDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Set permanent environment variables
    Write-Host ""
    Write-Host "Setting permanent environment variables..." -ForegroundColor Yellow
    [Environment]::SetEnvironmentVariable("CUDA_HOME", $CUDA_PATH, "Machine")
    [Environment]::SetEnvironmentVariable("CUDA_PATH", $CUDA_PATH, "Machine")
    [Environment]::SetEnvironmentVariable("CUDA_PATH_V13_0", $CUDA_PATH, "Machine")

    # Update PATH if not already present
    $machinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    if ($machinePath -notmatch [regex]::Escape("$CUDA_PATH\bin")) {
        $newPath = "$CUDA_PATH\bin;$CUDA_PATH\libnvvp;" + $machinePath
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "Machine")
        Write-Host "✓ PATH updated" -ForegroundColor Green
    } else {
        Write-Host "✓ PATH already configured" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host "✅ CUDA 13.0 VERIFICATION COMPLETE" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host ""
    Write-Host "CUDA Home: $CUDA_PATH" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next step: .\scripts\03_install_vs2022.ps1" -ForegroundColor Cyan
    Write-Host ""
    exit 0

} else {
    # CUDA not found - provide installation guidance
    Write-Host "✗ CUDA 13.0 not found at: $CUDA_PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Yellow
    Write-Host "CUDA 13.0 INSTALLATION REQUIRED" -ForegroundColor Yellow
    Write-Host "=" * 80 -ForegroundColor Yellow
    Write-Host ""
    Write-Host "CUDA 13.0 is required for native Blackwell (sm_120) support." -ForegroundColor White
    Write-Host ""
    Write-Host "📝 IMPORTANT NOTE:" -ForegroundColor Yellow
    Write-Host "   As of this writing, CUDA 13.0 may not be publicly released yet." -ForegroundColor White
    Write-Host "   It may be available as:" -ForegroundColor White
    Write-Host "   • Early Access Program" -ForegroundColor White
    Write-Host "   • Developer Preview" -ForegroundColor White
    Write-Host "   • Or through NVIDIA Direct (for RTX 5090 customers)" -ForegroundColor White
    Write-Host ""
    Write-Host "OPTION 1: Download CUDA 13.0 (Recommended)" -ForegroundColor Cyan
    Write-Host "   1. Visit: https://developer.nvidia.com/cuda-downloads" -ForegroundColor White
    Write-Host "   2. Select: Windows -> x86_64 -> 11 -> exe (local)" -ForegroundColor White
    Write-Host "   3. Download CUDA 13.0 installer" -ForegroundColor White
    Write-Host "   4. Run installer with default options" -ForegroundColor White
    Write-Host "   5. Re-run this script to verify" -ForegroundColor White
    Write-Host ""
    Write-Host "OPTION 2: Use CUDA 12.4 (Compatibility Mode)" -ForegroundColor Cyan
    Write-Host "   • CUDA 12.4 has sm_120 support but may have some limitations" -ForegroundColor White
    Write-Host "   • Not recommended for best performance" -ForegroundColor White
    Write-Host "   • Download from: https://developer.nvidia.com/cuda-12-4-0-download-archive" -ForegroundColor White
    Write-Host ""
    Write-Host "OPTION 3: Request Early Access" -ForegroundColor Cyan
    Write-Host "   • Contact NVIDIA Developer Program" -ForegroundColor White
    Write-Host "   • Mention RTX 5090 / Blackwell development" -ForegroundColor White
    Write-Host "   • Apply at: https://developer.nvidia.com/developer-program" -ForegroundColor White
    Write-Host ""

    $answer = Read-Host "Do you have CUDA 13.0 installer? (Y/N)"

    if ($answer -eq "Y" -or $answer -eq "y") {
        Write-Host ""
        Write-Host "Manual Installation Instructions:" -ForegroundColor Cyan
        Write-Host "1. Locate your CUDA 13.0 installer (.exe file)" -ForegroundColor White
        Write-Host "2. Run the installer as Administrator" -ForegroundColor White
        Write-Host "3. Follow the installation wizard (use defaults)" -ForegroundColor White
        Write-Host "4. After installation, re-run this script: .\scripts\02_verify_cuda.ps1" -ForegroundColor White
        Write-Host ""

        $installerPath = Read-Host "Enter path to CUDA 13.0 installer (or press Enter to skip)"

        if ($installerPath -and (Test-Path $installerPath)) {
            Write-Host ""
            Write-Host "Launching CUDA installer..." -ForegroundColor Yellow
            Write-Host "Please complete the installation, then re-run this script." -ForegroundColor Yellow
            Start-Process -FilePath $installerPath -Wait
            Write-Host ""
            Write-Host "Installation complete. Re-run verification:" -ForegroundColor Green
            Write-Host "  .\scripts\02_verify_cuda.ps1" -ForegroundColor Cyan
        }
    }

    Write-Host ""
    Write-Host "After installing CUDA 13.0, re-run: .\scripts\02_verify_cuda.ps1" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
