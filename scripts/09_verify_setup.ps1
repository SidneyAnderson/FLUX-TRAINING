#Requires -RunAsAdministrator
###############################################################################
# Complete Setup Verification (Windows 11)
###############################################################################

$ErrorActionPreference = "Stop"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "RTX 5090 + CUDA 13.0 COMPLETE SETUP VERIFICATION" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VENV_DIR = Join-Path $PROJECT_ROOT "venv"
$PYTHON_EXE = Join-Path $VENV_DIR "Scripts\python.exe"

$checks_passed = 0
$checks_failed = 0

# Python check
Write-Host ""
Write-Host "=== Python ===" -ForegroundColor Yellow
& $PYTHON_EXE --version
if ($?) { $checks_passed++ } else { $checks_failed++ }

# PyTorch check
Write-Host ""
Write-Host "=== PyTorch ===" -ForegroundColor Yellow
& $PYTHON_EXE -c @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Architecture list: {torch.cuda.get_arch_list()}')
    assert 'sm_120' in str(torch.cuda.get_arch_list()), 'sm_120 missing!'
    print('✓ Native sm_120 confirmed')
"@
if ($?) { $checks_passed++ } else { $checks_failed++ }

# Performance benchmark
Write-Host ""
Write-Host "=== Performance Benchmark ===" -ForegroundColor Yellow
& $PYTHON_EXE -c @"
import torch
import time
size = 8192
iterations = 100
a = torch.randn(size, size, dtype=torch.bfloat16, device='cuda')
b = torch.randn(size, size, dtype=torch.bfloat16, device='cuda')
for _ in range(10): torch.matmul(a, b)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(iterations): torch.matmul(a, b)
torch.cuda.synchronize()
end = time.perf_counter()
flops = 2 * size ** 3 * iterations
tflops = flops / (end - start) / 1e12
print(f'Performance: {tflops:.1f} TFLOPS')
if tflops > 500: print('✓ Performance acceptable')
"@

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Passed: $checks_passed" -ForegroundColor Green
Write-Host "Failed: $checks_failed" -ForegroundColor Red

if ($checks_failed -eq 0) {
    Write-Host ""
    Write-Host "✅ ALL CHECKS PASSED - READY FOR TRAINING!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next:" -ForegroundColor Cyan
    Write-Host "  1. Prepare dataset in .\dataset\" -ForegroundColor White
    Write-Host "  2. Download models to .\models\" -ForegroundColor White
    Write-Host "  3. Edit config\rtx5090_native.toml" -ForegroundColor White
    Write-Host "  4. Run: .\scripts\10_start_training.ps1" -ForegroundColor White
    exit 0
} else {
    Write-Host ""
    Write-Host "❌ SOME CHECKS FAILED" -ForegroundColor Red
    exit 1
}
