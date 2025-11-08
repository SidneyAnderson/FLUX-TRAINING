#Requires -RunAsAdministrator
###############################################################################
# RTX 5090 Prerequisites Verification (Windows 11)
# Checks system requirements before starting setup
###############################################################################

$ErrorActionPreference = "Stop"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "RTX 5090 FLUX TRAINING - PREREQUISITES VERIFICATION (WINDOWS 11)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$checks_passed = 0
$checks_failed = 0
$checks_warning = 0

# Check Windows version
Write-Host "=== Operating System ===" -ForegroundColor Yellow
$os = Get-WmiObject Win32_OperatingSystem
Write-Host "OS: $($os.Caption)"
Write-Host "Version: $($os.Version)"
Write-Host "Build: $($os.BuildNumber)"

if ($os.Caption -match "Windows 11" -and [int]$os.BuildNumber -ge 22000) {
    Write-Host "✓ Windows 11 detected (Build $($os.BuildNumber))" -ForegroundColor Green
    $checks_passed++
} else {
    Write-Host "✗ Windows 11 (Build 22000+) required" -ForegroundColor Red
    $checks_failed++
}

# Check if running as Administrator
Write-Host ""
Write-Host "=== Administrator Privileges ===" -ForegroundColor Yellow
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if ($currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "✓ Running as Administrator" -ForegroundColor Green
    $checks_passed++
} else {
    Write-Host "✗ Must run as Administrator" -ForegroundColor Red
    $checks_failed++
}

# Check RAM
Write-Host ""
Write-Host "=== System Memory ===" -ForegroundColor Yellow
$ram = [math]::Round((Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Host "Total RAM: ${ram}GB"
if ($ram -ge 60) {
    Write-Host "✓ Sufficient RAM (${ram}GB >= 64GB recommended)" -ForegroundColor Green
    $checks_passed++
} else {
    Write-Host "⚠ RAM: ${ram}GB (64GB recommended for compilation)" -ForegroundColor Yellow
    $checks_warning++
}

# Check disk space
Write-Host ""
Write-Host "=== Disk Space ===" -ForegroundColor Yellow
$drive = Get-PSDrive -Name C
$freeSpaceGB = [math]::Round($drive.Free / 1GB)
Write-Host "C:\ Free Space: ${freeSpaceGB}GB"
if ($freeSpaceGB -ge 200) {
    Write-Host "✓ Sufficient disk space (${freeSpaceGB}GB >= 200GB required)" -ForegroundColor Green
    $checks_passed++
} else {
    Write-Host "✗ Insufficient disk space: ${freeSpaceGB}GB (200GB required)" -ForegroundColor Red
    $checks_failed++
}

# Check GPU
Write-Host ""
Write-Host "=== GPU Detection ===" -ForegroundColor Yellow
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader
    Write-Host "GPU Info: $gpuInfo"

    if ($gpuInfo -match "5090") {
        Write-Host "✓ RTX 5090 detected" -ForegroundColor Green
        $checks_passed++

        # Check compute capability
        $computeCap = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader).Trim()
        if ($computeCap -eq "12.0") {
            Write-Host "✓ Compute Capability: sm_120" -ForegroundColor Green
            $checks_passed++
        } else {
            Write-Host "✗ Wrong compute capability: $computeCap (expected 12.0)" -ForegroundColor Red
            $checks_failed++
        }
    } else {
        Write-Host "⚠ RTX 5090 not detected. Found: $gpuInfo" -ForegroundColor Yellow
        Write-Host "  This setup is optimized for RTX 5090" -ForegroundColor Yellow
        $checks_warning++
    }
} else {
    Write-Host "✗ nvidia-smi not found - GPU drivers not installed" -ForegroundColor Red
    Write-Host "  Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    $checks_failed++
}

# Check PowerShell version
Write-Host ""
Write-Host "=== PowerShell Version ===" -ForegroundColor Yellow
Write-Host "Version: $($PSVersionTable.PSVersion)"
if ($PSVersionTable.PSVersion.Major -ge 5) {
    Write-Host "✓ PowerShell $($PSVersionTable.PSVersion.Major) (5.1+ required)" -ForegroundColor Green
    $checks_passed++
} else {
    Write-Host "✗ PowerShell 5.1+ required" -ForegroundColor Red
    $checks_failed++
}

# Check for existing Python installations
Write-Host ""
Write-Host "=== Python Detection ===" -ForegroundColor Yellow
$pythonPaths = @(
    "C:\Python311",
    "$env:LOCALAPPDATA\Programs\Python",
    "$env:APPDATA\Python"
)

$pythonFound = $false
foreach ($path in $pythonPaths) {
    if (Test-Path $path) {
        Write-Host "⚠ Python installation found at: $path" -ForegroundColor Yellow
        $pythonFound = $true
    }
}

if ($pythonFound) {
    Write-Host "  Existing Python installations will be cleaned during setup" -ForegroundColor Yellow
    $checks_warning++
} else {
    Write-Host "✓ No conflicting Python installations" -ForegroundColor Green
    $checks_passed++
}

# Check internet connection
Write-Host ""
Write-Host "=== Internet Connection ===" -ForegroundColor Yellow
try {
    $response = Test-Connection -ComputerName google.com -Count 1 -Quiet
    if ($response) {
        Write-Host "✓ Internet connection active" -ForegroundColor Green
        $checks_passed++
    } else {
        Write-Host "✗ No internet connection" -ForegroundColor Red
        $checks_failed++
    }
} catch {
    Write-Host "✗ Cannot verify internet connection" -ForegroundColor Red
    $checks_failed++
}

# Check for Visual Studio
Write-Host ""
Write-Host "=== Visual Studio 2022 Build Tools ===" -ForegroundColor Yellow
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if (Test-Path $vsPath) {
    Write-Host "✓ Visual Studio 2022 Build Tools found" -ForegroundColor Green
    $checks_passed++
} else {
    Write-Host "○ Visual Studio 2022 Build Tools not found (will be installed)" -ForegroundColor Yellow
    Write-Host "  Location: $vsPath" -ForegroundColor Gray
    $checks_warning++
}

# Check for CUDA
Write-Host ""
Write-Host "=== CUDA Toolkit ===" -ForegroundColor Yellow
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
if (Test-Path $cudaPath) {
    Write-Host "✓ CUDA 13.0 found at: $cudaPath" -ForegroundColor Green
    $checks_passed++

    # Verify nvcc
    $env:PATH = "$cudaPath\bin;" + $env:PATH
    if (Get-Command nvcc -ErrorAction SilentlyContinue) {
        $nvccVersion = (nvcc --version | Select-String "release").ToString()
        Write-Host "  NVCC: $nvccVersion" -ForegroundColor Gray
    }
} else {
    Write-Host "○ CUDA 13.0 not found (will be verified/installed)" -ForegroundColor Yellow
    Write-Host "  Expected: $cudaPath" -ForegroundColor Gray
    $checks_warning++
}

# Summary
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "VERIFICATION SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Passed:   " -NoNewline
Write-Host $checks_passed -ForegroundColor Green
Write-Host "Warnings: " -NoNewline
Write-Host $checks_warning -ForegroundColor Yellow
Write-Host "Failed:   " -NoNewline
Write-Host $checks_failed -ForegroundColor Red
Write-Host ""

if ($checks_failed -eq 0) {
    Write-Host "✅ Prerequisites verification complete!" -ForegroundColor Green
    if ($checks_warning -gt 0) {
        Write-Host "⚠  Some components will be installed during setup" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Next step: .\scripts\01_install_python.ps1" -ForegroundColor Cyan
    exit 0
} else {
    Write-Host "❌ Some critical prerequisites are missing" -ForegroundColor Red
    Write-Host ""
    Write-Host "Required actions:" -ForegroundColor Yellow
    Write-Host "  1. Ensure running as Administrator" -ForegroundColor White
    Write-Host "  2. Install NVIDIA drivers if GPU not detected" -ForegroundColor White
    Write-Host "  3. Ensure sufficient disk space (200GB on C:\)" -ForegroundColor White
    Write-Host "  4. Verify internet connection" -ForegroundColor White
    Write-Host ""
    exit 1
}
